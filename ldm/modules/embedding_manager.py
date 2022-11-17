from typing import Union, Callable, Optional, cast

import torch
from torch import Tensor, nn

from ldm.util import default, as_list
from ldm.modules.embedding_shuffler import ShuffleMode, get_shuffler
from ldm.data.classic import extra_token_list
from functools import partial

DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

CLIP_ARGS = {
    "truncation": True,
    "max_length": 77,
    "return_length": True,
    "return_overflowing_tokens": False,
    "padding": "max_length",
    "return_tensors": "pt"
}

InitWords = Union[list[str], str]
WordsToTokenFn = Callable[[list[str]], Tensor]
TokenToEmbFn = Callable[[Tensor], Tensor]


class TokenSizeError(ValueError):
    """Indicates that no words mapped to a single token."""
    def __init__(self, words: list[str]):
        joined_words = ", ".join(words)
        message = "\n".join([
            f"All of the following string(s) map to more than one token: {joined_words}",
            "Please try another string."
        ])
        super().__init__(message)
        self.words = words


def get_clip_token_for_string(tokenizer, words: list[str]):
    for word in words:
        batch_encoding = tokenizer(word, **CLIP_ARGS)
        tokens = batch_encoding["input_ids"]
        if torch.count_nonzero(tokens - 49407) == 2: return tokens[0, 1]

    raise TokenSizeError(words)

def get_bert_token_for_string(tokenizer, words: list[str]):
    for word in words:
        token = tokenizer(word)
        if torch.count_nonzero(token) == 3: return token[0, 1]

    raise TokenSizeError(words)

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings: Optional[list[str]]=None,
            initializer_words: Optional[list[InitWords]]=None,
            shuffle_mode: Optional[Union[bool, ShuffleMode]]=None,
            per_image_tokens: bool=False,
            num_vectors_per_token: Optional[int]=None,
            subject_vectors: Optional[int]=None,
            quality_vectors: Optional[int]=None,
            progressive_words=False,
            **kwargs
    ):
        super().__init__()

        placeholder_strings = default(placeholder_strings, DEFAULT_PLACEHOLDER_TOKEN)
        # Ensure it is illegal to pass an empty list.
        assert len(placeholder_strings) > 0, "Need at least one placeholder string."

        extra_tokens = extra_token_list if per_image_tokens else []
        placeholder_strings = placeholder_strings + extra_tokens

        initializer_words = default(initializer_words, [])

        # `num_vectors_per_token` is for backward compatibility only.
        assert num_vectors_per_token is None or subject_vectors is None, "Cannot use `num_vectors_per_token` with `subject_vectors`."
        assert num_vectors_per_token is None or quality_vectors is None, "Cannot use `num_vectors_per_token` with `quality_vectors`."

        num_vectors_per_token = default(num_vectors_per_token, 1)
        self.subject_vectors = default(subject_vectors, num_vectors_per_token)
        self.quality_vectors = default(quality_vectors, self.subject_vectors)

        # The first placeholder is assumed to represent the subject.
        self.subject_placeholder = placeholder_strings[0]
        self.shuffle_embeddings = get_shuffler(default(shuffle_mode, "off"))
        self.progressive_words = progressive_words
        self.progressive_counter = 0

        if hasattr(embedder, "tokenizer"): # using Stable Diffusion's CLIP encoder
            as_token = cast(WordsToTokenFn, partial(get_clip_token_for_string, embedder.tokenizer))
            as_embedding = cast(TokenToEmbFn, partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings))
            token_dim = 768
        else: # using LDM's BERT encoder
            as_token = cast(WordsToTokenFn, partial(get_bert_token_for_string, embedder.tknz_fn))
            as_embedding = cast(TokenToEmbFn, embedder.transformer.token_emb)
            token_dim = 1280

        # Initialize the initial embeddings; they should not be optimized.
        self.initial_embeddings = nn.ParameterDict()
        for idx, placeholder_string in enumerate(placeholder_strings):
            num_vectors = self.subject_vectors if placeholder_string == self.subject_placeholder else self.quality_vectors
            init_words = as_list(initializer_words[idx]) if idx < len(initializer_words) else []
            if len(init_words) > 0:
                init_word_token = as_token(init_words)
                with torch.no_grad():
                    init_word_embedding = as_embedding(init_word_token.cpu())

                self.initial_embeddings[placeholder_string] = nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors, 1), requires_grad=False)
            else:
                # Fallback to a random start.
                self.initial_embeddings[placeholder_string] = nn.Parameter(torch.rand(size=(num_vectors, token_dim), requires_grad=False))

        # Initialize the optimized embeddings.
        self.string_to_token_dict: dict[str, Tensor] = {}
        self.string_to_param_dict = nn.ParameterDict()
        for placeholder_string in placeholder_strings:
            token = as_token(as_list(placeholder_string))
            init_token_params = self.initial_embeddings[placeholder_string]
            
            self.string_to_token_dict[placeholder_string] = token.detach()
            self.string_to_param_dict[placeholder_string] = nn.Parameter(init_token_params, requires_grad=True)

    def forward(
            self,
            tokenized_text: Tensor,
            embedded_text: Tensor,
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device

        if self.progressive_words:
            self.progressive_counter += 1

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            max_vectors = self.subject_vectors if placeholder_string == self.subject_placeholder else self.quality_vectors
            placeholder_token = placeholder_token.to(device)
            placeholder_embedding = cast(Tensor, self.string_to_param_dict[placeholder_string]).to(device)

            if max_vectors == 1: # If there's only one vector per token, we can do a simple replacement
                placeholder_idx = torch.where(tokenized_text == placeholder_token)
                embedded_text[placeholder_idx] = placeholder_embedding
            else: # otherwise, need to insert and keep track of changing indices
                if self.progressive_words:
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:
                    max_step_tokens = max_vectors

                num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)

                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token)

                if placeholder_rows.nelement() == 0:
                    continue

                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]

                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]

                    shuffle_view = self.shuffle_embeddings(placeholder_embedding, num_vectors_for_token)
                    new_token_row = torch.cat([tokenized_text[row][:col], tokenized_text[row][col].repeat(num_vectors_for_token), tokenized_text[row][col + 1:]], axis=0)[:n]
                    new_embed_row = torch.cat([embedded_text[row][:col], shuffle_view, embedded_text[row][col + 1:]], axis=0)[:n]

                    embedded_text[row]  = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text

    def save(self, ckpt_path):
        save_obj = {
            "subject_placeholder": self.subject_placeholder,
            "string_to_token": self.string_to_token_dict,
            "string_to_param": self.string_to_param_dict,
            "progressive_counter": self.progressive_counter
        }
        torch.save(save_obj, ckpt_path)

    def load(self, ckpt_path):
        ckpt: dict = torch.load(ckpt_path, map_location="cpu")

        # Primarily to allow other software to deal with multi-term embeddings.
        # This is the embedding of the intentionally trained subject.
        # All we can really do here is a config sanity check.
        loaded_placeholder = ckpt.get("subject_placeholder", self.subject_placeholder)
        assert loaded_placeholder == self.subject_placeholder, f"The subject placeholder changed from \"{loaded_placeholder}\" to \"{self.subject_placeholder}\" since the checkpoint was saved."

        # Allows `progressive_words` mode to resume properly.
        self.progressive_counter: int = ckpt.get("progressive_counter", 0)

        token_dict: dict[str, Tensor] = ckpt["string_to_token"]
        self.string_to_token_dict.update(token_dict)
        
        param_dict: dict[str, Tensor] = ckpt["string_to_param"]
        self.string_to_param_dict.update(param_dict)

        self.resize_after_load()

    def resize_after_load(self):
        """
        Checks to see if the loaded parameters need to have their size adjusted
        and does so as needed.

        It is possible the configuration was altered and the number of vectors
        has changed.  This could cause `embedding_to_coarse_loss` to throw errors
        when doing its subtraction.
        """
        for placeholder_string, in_param in self.string_to_param_dict.items():
            max_vectors = self.subject_vectors if placeholder_string == self.subject_placeholder else self.quality_vectors
            placeholder_embedding: Tensor = in_param
            loaded_size = placeholder_embedding.shape[0]
            
            # Same size, all good.
            if loaded_size == max_vectors:
                continue
            # Loaded is smaller, overlay on top of the initial embedding.
            if loaded_size < max_vectors:
                with torch.no_grad():
                    init_embedding: Tensor = self.initial_embeddings[placeholder_string]
                    to_add = max_vectors - loaded_size
                    resized = torch.cat([placeholder_embedding, init_embedding[-to_add:]], 0)
                self.string_to_param_dict[placeholder_string] = nn.Parameter(resized, requires_grad=True)
                print(f"Expanded `{placeholder_string}` from {loaded_size} to {resized.shape[0]} vectors.")
            # Loaded is larger, truncate the additional vectors.
            if loaded_size > max_vectors:
                with torch.no_grad():
                    to_remove = loaded_size - max_vectors
                    resized = placeholder_embedding[:-to_remove]
                self.string_to_param_dict[placeholder_string] = nn.Parameter(resized, requires_grad=True)
                print(f"Contracted `{placeholder_string}` from {loaded_size} to {resized.shape[0]} vectors.")

    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_param_dict.parameters()

    def embedding_to_coarse_loss(self):
        loss = torch.zeros(1, requires_grad=True)
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss