from typing import Optional, Sequence
from argparse import Namespace
from dataclasses import dataclass, field

import os
import random
import itertools
import re
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

import ldm.data.image_loader as di
import ldm.data.classic as dc
from ldm.config import DatasetKey, ConfigInstantiable, BasicDataModule, operate_on_config
from ldm.util import default, partition, compact_dict, exists

subject_conditions = [
    'a rendition',
    'a depiction',
    '[a|the] photo',
    'a [cropped|close-up|bright|dark|cropped|good] photo',
    'a rendering',
    'an illustration',
]

subject_templates = [
    '{condition} of [a|the|my|one] {subject}',
    '{condition} of [a|the|my|one] {subject} with {quality}',
    '{condition} of [a|the] [clean|dirty|cool|nice|weird|small|large] {subject}',
    '{condition} of [a|the] [clean|dirty|cool|nice|weird|small|large] {subject} with {quality}',
]

style_conditions = [
    'a picture',
    'a rendering',
    'a painting',
    'a [cropped|close-up|bright|dark|good|nice|small|large|weird] painting',
]

style_templates = [
    '{condition} in the style of {subject}',
    '{condition} in the style of {subject} with {quality}',
]


@dataclass
class MetadataConf:
    """Per-file metadata for each image."""
    enabled: bool = True
    conditions: list[str] = field(default_factory=list)
    qualities: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)


class ImageSrcWithMetadata(di.ImageSrc):
    """An image setup for processing, with metadata."""
    metadata: MetadataConf


class ImageDataWithMetadata(di.ImageData):
    """An image ready to use in the dataset, with metadata."""
    metadata: MetadataConf


def load_image_metadata(src_data: di.ImageSrc):
    config: MetadataConf = OmegaConf.structured(MetadataConf)

    try:
        config_path = os.path.splitext(src_data["path"])[0] + ".yaml"
        config = OmegaConf.merge(config, OmegaConf.load(config_path))
    except FileNotFoundError:
        pass

    return ImageSrcWithMetadata(metadata=config, **src_data)


class ImageLoaderWithMetadata(di.ImageLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load the per-image metadata, filtering out any disabled images.
        self.images = list(filter(
            lambda img: img["metadata"].enabled,
            map(load_image_metadata, self.images)
        ))
    
    def __getitem__(self, i):
        src_data = super().__getitem__(i)
        metadata = self.images[i]["metadata"]
        return ImageDataWithMetadata(metadata=metadata, **src_data)


def get_quality_tokens(qualities: list[str]):
    assert len(qualities) <= len(dc.extra_token_list), f"Loaded {len(qualities)} unique qualities, but only have {len(dc.extra_token_list)} extra tokens. Try adding more tokens to 'extra_token_list'."
    result: dict[str, str] = {}
    for i, v in enumerate(qualities):
        result[v] = dc.extra_token_list[i]
    return result

def is_conditional_template(template: str):
    return "{condition}" in template

def is_dual_template(template: str):
    return "{quality}" in template

def is_subjective_template(template: str):
    return "{subject}" in template

def do_replacement(match: re.Match[str]):
    return random.choice(match.group(1).split("|"))

def process_string_replacers(template: str):
    """
    Replaces substrings with the pattern `[str1|str2|str3]` with a random choice
    of `str1` or `str2` or so on.
    """
    return re.sub(r"\[((?:\|?[^\]|]*)+?)\]", do_replacement, template)

def pick_quality(qualities: list[str], mixing_prob: float, solo_count: int, dual_count: int):
    if dual_count == 0:
        assert solo_count > 0, "No templates were available to properly construct a conditioning caption."
        return None
    if len(qualities) == 0:
        assert solo_count > 0, "An image's metadata had an empty `qualities` array, but no template works without a quality."
        return None

    if mixing_prob == 0.0: return None
    if mixing_prob < 1.0 and np.random.uniform() >= mixing_prob: return None
    return random.choice(qualities)


class ManagedBase(Dataset, Sequence):
    def __init__(
                 self,
                 images: ImageLoaderWithMetadata,
                 multiplier: int=100,
                 **kwargs
                ):
        super().__init__()

        assert multiplier >= 0, "The `multiplier` argument must be greater than zero."
        self.multiplier = multiplier
        self.images = images

        # Generate a list of known qualities.
        # We have a resume issue I'm not sure how to resolve.  If the user adds
        # a new quality, it can change how the qualities are mapped to tokens.
        # Those tokens may already have a lot of training, which would be
        # upset by a sudden change in mappings.
        self.all_qualities: list[str] = list(set(itertools.chain(*map(
            lambda img: img["metadata"].qualities,
            self.images
        ))))
        self.all_qualities.sort()

        # Assign them each a placeholder string.
        self.quality_to_placeholder = get_quality_tokens(self.all_qualities)
        # When you do not use `{quality}`, you can reference specific qualities
        # directly, so let's cache a description map for formatting.
        self.quality_to_description = { quality: f"[{quality}]" for quality in self.all_qualities }
    
    def __len__(self):
        return len(self.images) * self.multiplier

    def __getitem__(self, i: int):
        if i < len(self): return self.images[i % len(self.images)]
        raise IndexError(f"Requested index {i} but dataset has size {len(self)}.")


class ManagedData(di.ImageData):
    """An image with caption and used placeholders."""
    caption: str
    human_caption: str


class ManagedStyle(ManagedBase):
    def __init__(self,
                 placeholder_token: str="*",
                 placeholder_desc: Optional[str]=None,
                 templates: Optional[list[str]]=None,
                 default_conditions: Optional[list[str]]=None,
                 extra_conditions: Optional[list[str]]=None,
                 mixing_prob: Optional[float]=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.placeholder_token = placeholder_token
        self.placeholder_desc = default(placeholder_desc, placeholder_token)
        self.default_conditions = default(default_conditions, style_conditions)
        self.extra_conditions = default(extra_conditions, [])
        self.base_templates = default(templates, style_templates)
        self.mixing_prob = mixing_prob

    def __getitem__(self, i: int):
        src_data = super().__getitem__(i)
        metadata = src_data["metadata"]

        base_conditions = metadata.conditions if len(metadata.conditions) > 0 else self.default_conditions
        extra_conditions = [] if "conditions" in metadata.overrides else self.extra_conditions
        all_conditions = list(set(base_conditions + extra_conditions))
        assert len(all_conditions) > 0, f"At least one condition is required for: {src_data['path']}"

        base_templates = [] if "templates" in metadata.overrides else self.base_templates
        all_templates = list(set(base_templates + metadata.templates))
        assert len(all_templates) > 0, f"At least one template is required for: {src_data['path']}"

        parted_templates = partition(all_templates, is_dual_template)
        solo_templates = list(parted_templates[0])
        dual_templates = list(parted_templates[1])
        solo_count = len(solo_templates)
        dual_count = len(dual_templates)
        mixing_prob = self.get_mixing_prob(solo_count, dual_count, len(metadata.qualities))

        # This also handles some assertions.  If this is not `None`,
        # it is safe to use `dual_templates`.
        quality_key = pick_quality(metadata.qualities, mixing_prob, len(solo_templates), len(dual_templates))
        condition = process_string_replacers(random.choice(all_conditions))

        if quality_key is not None:
            extra_placeholder = self.quality_to_placeholder[quality_key]
            text = process_string_replacers(random.choice(dual_templates))
            assert is_subjective_template(text), f"Expected a subject placeholder in: {text}"

            caption = text.format(
                condition=condition,
                subject=self.placeholder_token,
                quality=extra_placeholder
            )
            human_caption = text.format(
                condition=condition,
                subject=f"[{self.placeholder_desc}]",
                quality=f"[{quality_key}]"
            )
        else:
            text = process_string_replacers(random.choice(solo_templates))
            assert is_subjective_template(text), f"Expected a subject placeholder in: {text}"

            caption = text.format(
                condition=condition,
                subject=self.placeholder_token,
                **self.quality_to_placeholder
            )
            human_caption = text.format(
                condition=condition,
                subject=f"[{self.placeholder_desc}]",
                **self.quality_to_description
            )

        return ManagedData(
            caption=caption,
            human_caption=human_caption,
            image=src_data["image"],
            path=src_data["path"]
        )
    
    def get_mixing_prob(self, solo_count: int, dual_count: int, quality_count: int):
        if self.mixing_prob is not None: return self.mixing_prob
        dual_count = dual_count * quality_count
        if dual_count == 0: return 0.0
        if solo_count == 0: return 1.0
        return dual_count / (dual_count + solo_count)


class ManagedSubject(ManagedStyle):
    def __init__(self,
                 templates: Optional[list[str]]=None,
                 default_conditions: Optional[list[str]]=None,
                 **kwargs
                 ):
        # Use the subject variants instead of those of style.
        templates = default(templates, subject_templates)
        default_conditions = default(default_conditions, subject_conditions)
        super().__init__(templates=templates, default_conditions=default_conditions, **kwargs)


# constants for config parameter locations
SUPPORTED_KLASSES = [
    "ldm.data.managed.ManagedSubject",
    "ldm.data.managed.ManagedStyle"
]

PERS_CONFIG = ConfigInstantiable(
    path="model.params.personalization_config",
    klasses=["ldm.modules.embedding_manager.EmbeddingManager"]
)
DATA_CONFIG = ConfigInstantiable(
    path="data",
    klasses=["ldm.data.managed.ManagedDataModule"]
)
IMAGES_CONFIG = ConfigInstantiable(
    path=f"{DATA_CONFIG.params}.images",
    klasses=["ldm.data.managed.ImageLoaderWithMetadata"]
)
DATA_CONFIGS: dict[DatasetKey, ConfigInstantiable] = {
    "train": ConfigInstantiable(
        path=f"{DATA_CONFIG.params}.train",
        klasses=SUPPORTED_KLASSES
    ),
    "validation": ConfigInstantiable(
        path=f"{DATA_CONFIG.params}.validation",
        klasses=SUPPORTED_KLASSES
    ),
    "test": ConfigInstantiable(
        path=f"{DATA_CONFIG.params}.test",
        klasses=SUPPORTED_KLASSES
    ),
    "predict": ConfigInstantiable(
        path=f"{DATA_CONFIG.params}.predict",
        klasses=SUPPORTED_KLASSES
    )
}


class ManagedDataModule(BasicDataModule):
    def __init__(self,
                 images: DictConfig,
                 placeholder_token: str="*",
                 placeholder_desc: Optional[str]=None,
                 initializer_word: Optional[str]=None,
                 templates: Optional[list[str]]=None,
                 default_conditions: Optional[list[str]]=None,
                 extra_conditions: Optional[list[str]]=None,
                 mixing_prob: Optional[float]=None,
                 **kwargs
                ):
        super().__init__(**kwargs)

        self.extra_config_args = compact_dict({
            "images": IMAGES_CONFIG.direct().instantiate(images),
            "placeholder_token": placeholder_token,
            "placeholder_desc": default(placeholder_desc, initializer_word),
            "templates": templates,
            "default_conditions": default_conditions,
            "extra_conditions": extra_conditions,
            "mixing_prob": mixing_prob,
        })
    
    def instantiate_data(self, config_key: DatasetKey):
        return DATA_CONFIGS[config_key].direct().instantiate(
            self.dataset_configs[config_key],
            **self.extra_config_args
        )

    @staticmethod
    def normalize_config(config: DictConfig, opt: Namespace):
        pt = OmegaConf.select(config, PERS_CONFIG.target, default="")
        assert PERS_CONFIG.supports(pt), f"Requires {PERS_CONFIG.target} to be one of: {PERS_CONFIG.klasses}"

        pp = PERS_CONFIG.params
        dp = DATA_CONFIG.params
        ip = IMAGES_CONFIG.params

        operate_on_config(config,
            ("set_as", f"{dp}.placeholder_token", opt.placeholder_string),
            # Use the CLI argument if possible.
            ("set_as", f"{dp}.initializer_word", opt.init_word),
            # Otherwise, check in the personalization config.
            ("default_from", f"{dp}.initializer_word", f"{pp}.initializer_words[0]"),
            # Set the data root for the source images.
            ("set_as", f"{ip}.data_root", opt.data_root),
        )

        # We need the metadata to properly setup the embedding manager.
        managed = ManagedBase(IMAGES_CONFIG.instantiate(config), multiplier=1)
        init_word = OmegaConf.select(config, f"{dp}.initializer_word", default=None)

        placeholder_strings = [
            OmegaConf.select(config, f"{dp}.placeholder_token", default="*"),
            *managed.quality_to_placeholder.values(),
        ]

        initializer_words = [
            default(init_word, []),
            *map(
                lambda word: [word, init_word] if exists(init_word) else word,
                managed.all_qualities
            )
        ]

        operate_on_config(config,
            ("set_as", f"{pp}.embedding_manager_ckpt", opt.embedding_manager_ckpt),
            ("set_as", f"{pp}.placeholder_strings", placeholder_strings),
            ("set_as", f"{pp}.initializer_words", initializer_words),
            # Copy for convenience.
            ("set_from", f"{pp}.num_vectors_per_token", f"{dp}.num_vectors_per_token"),
            ("set_from", f"{pp}.subject_vectors", f"{dp}.subject_vectors"),
            ("set_from", f"{pp}.quality_vectors", f"{dp}.quality_vectors"),
            # And discard them.
            ("drop", f"{dp}.num_vectors_per_token"),
            ("drop", f"{dp}.subject_vectors"),
            ("drop", f"{dp}.quality_vectors"),
        )

        # Not compatible with managed data.
        per_image_tokens = OmegaConf.select(config, f"{pp}.per_image_tokens", default=False)
        assert not per_image_tokens, f"The `per_image_tokens` mode is not compatible with `{DATA_CONFIG.target}`."