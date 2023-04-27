from typing import Optional, Union, Sequence
from argparse import Namespace

import random
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig, OmegaConf

from ldm.util import default
from ldm.data.image_loader import ImageLoader, ImageData
from ldm.config import DatasetKey, ConfigInstantiable, BasicDataModule, operate_on_config
from ldm.util import compact_dict

# The interpolation placeholder `{0}` maps to the primary
# token for the subject being trained (usually "*") and
# `{1}` will map to the extra per-image token.

subject_templates = [
    'a photo of a {0}',
    'a rendering of a {0}',
    'a cropped photo of the {0}',
    'the photo of a {0}',
    'a photo of a clean {0}',
    'a photo of a dirty {0}',
    'a dark photo of the {0}',
    'a photo of my {0}',
    'a photo of the cool {0}',
    'a close-up photo of a {0}',
    'a bright photo of the {0}',
    'a cropped photo of a {0}',
    'a photo of the {0}',
    'a good photo of the {0}',
    'a photo of one {0}',
    'a close-up photo of the {0}',
    'a rendition of the {0}',
    'a photo of the clean {0}',
    'a rendition of a {0}',
    'a photo of a nice {0}',
    'a good photo of a {0}',
    'a photo of the nice {0}',
    'a photo of the small {0}',
    'a photo of the weird {0}',
    'a photo of the large {0}',
    'a photo of a cool {0}',
    'a photo of a small {0}',
    'an illustration of a {0}',
    'a rendering of a {0}',
    'a cropped photo of the {0}',
    'the photo of a {0}',
    'an illustration of a clean {0}',
    'an illustration of a dirty {0}',
    'a dark photo of the {0}',
    'an illustration of my {0}',
    'an illustration of the cool {0}',
    'a close-up photo of a {0}',
    'a bright photo of the {0}',
    'a cropped photo of a {0}',
    'an illustration of the {0}',
    'a good photo of the {0}',
    'an illustration of one {0}',
    'a close-up photo of the {0}',
    'a rendition of the {0}',
    'an illustration of the clean {0}',
    'a rendition of a {0}',
    'an illustration of a nice {0}',
    'a good photo of a {0}',
    'an illustration of the nice {0}',
    'an illustration of the small {0}',
    'an illustration of the weird {0}',
    'an illustration of the large {0}',
    'an illustration of a cool {0}',
    'an illustration of a small {0}',
    'a depiction of a {0}',
    'a rendering of a {0}',
    'a cropped photo of the {0}',
    'the photo of a {0}',
    'a depiction of a clean {0}',
    'a depiction of a dirty {0}',
    'a dark photo of the {0}',
    'a depiction of my {0}',
    'a depiction of the cool {0}',
    'a close-up photo of a {0}',
    'a bright photo of the {0}',
    'a cropped photo of a {0}',
    'a depiction of the {0}',
    'a good photo of the {0}',
    'a depiction of one {0}',
    'a close-up photo of the {0}',
    'a rendition of the {0}',
    'a depiction of the clean {0}',
    'a rendition of a {0}',
    'a depiction of a nice {0}',
    'a good photo of a {0}',
    'a depiction of the nice {0}',
    'a depiction of the small {0}',
    'a depiction of the weird {0}',
    'a depiction of the large {0}',
    'a depiction of a cool {0}',
    'a depiction of a small {0}',
]

subject_dual_templates = [
    'a photo of a {0} with {1}',
    'a rendering of a {0} with {1}',
    'a cropped photo of the {0} with {1}',
    'the photo of a {0} with {1}',
    'a photo of a clean {0} with {1}',
    'a photo of a dirty {0} with {1}',
    'a dark photo of the {0} with {1}',
    'a photo of my {0} with {1}',
    'a photo of the cool {0} with {1}',
    'a close-up photo of a {0} with {1}',
    'a bright photo of the {0} with {1}',
    'a cropped photo of a {0} with {1}',
    'a photo of the {0} with {1}',
    'a good photo of the {0} with {1}',
    'a photo of one {0} with {1}',
    'a close-up photo of the {0} with {1}',
    'a rendition of the {0} with {1}',
    'a photo of the clean {0} with {1}',
    'a rendition of a {0} with {1}',
    'a photo of a nice {0} with {1}',
    'a good photo of a {0} with {1}',
    'a photo of the nice {0} with {1}',
    'a photo of the small {0} with {1}',
    'a photo of the weird {0} with {1}',
    'a photo of the large {0} with {1}',
    'a photo of a cool {0} with {1}',
    'a photo of a small {0} with {1}',
]

style_templates = [
    'a painting in the style of {0}',
    'a rendering in the style of {0}',
    'a cropped painting in the style of {0}',
    'the painting in the style of {0}',
    'a clean painting in the style of {0}',
    'a dirty painting in the style of {0}',
    'a dark painting in the style of {0}',
    'a picture in the style of {0}',
    'a cool painting in the style of {0}',
    'a close-up painting in the style of {0}',
    'a bright painting in the style of {0}',
    'a cropped painting in the style of {0}',
    'a good painting in the style of {0}',
    'a close-up painting in the style of {0}',
    'a rendition in the style of {0}',
    'a nice painting in the style of {0}',
    'a small painting in the style of {0}',
    'a weird painting in the style of {0}',
    'a large painting in the style of {0}',
]

style_dual_templates = [
    'a painting in the style of {0} with {1}',
    'a rendering in the style of {0} with {1}',
    'a cropped painting in the style of {0} with {1}',
    'the painting in the style of {0} with {1}',
    'a clean painting in the style of {0} with {1}',
    'a dirty painting in the style of {0} with {1}',
    'a dark painting in the style of {0} with {1}',
    'a cool painting in the style of {0} with {1}',
    'a close-up painting in the style of {0} with {1}',
    'a bright painting in the style of {0} with {1}',
    'a cropped painting in the style of {0} with {1}',
    'a good painting in the style of {0} with {1}',
    'a painting of one {1} in the style of {0}',
    'a nice painting in the style of {0} with {1}',
    'a small painting in the style of {0} with {1}',
    'a weird painting in the style of {0} with {1}',
    'a large painting in the style of {0} with {1}',
]

extra_token_list = [
    '👹', '👻', '👽', '💄', '💃', '👯', '👠', '👟', '👑', '👅', '💀', '💍'
]


def as_extra_tokens(tokens: Union[list[str], bool, None]):
    if tokens is None: return extra_token_list
    if tokens is True: return extra_token_list
    if tokens is False: return []
    return tokens

def get_extra_tokens(tokens: Union[list[str], bool, None], count: int):
    tokens = as_extra_tokens(tokens)
    tokens_len = len(tokens)
    # extra tokens are not used when the length is 0
    if tokens_len == 0: return tokens
    # otherwise, make sure we have enough tokens for the count
    assert count <= tokens_len, f"{count} extra tokens were needed, but only {tokens_len} were available. Try adding more tokens to 'extra_token_list'."
    return tokens


class ClassicData(ImageData):
    """An image with caption."""
    caption: str


class ClassicStyle(Dataset, Sequence):
    def __init__(self,
                 images: ImageLoader,
                 multiplier: int=100,
                 placeholder_token: str="*",
                 templates: Optional[list[str]]=None,
                 dual_templates: Optional[list[str]]=None,
                 per_image_tokens: Union[list[str], bool, None]=False,
                 mixing_prob: float=0.25,
                 **kwargs
                 ):
        super().__init__()

        assert multiplier > 0, "The `multiplier` argument must be greater than zero."

        self.images = images
        self.multiplier = multiplier
        self.placeholder_token = placeholder_token
        self.templates = default(templates, style_templates)
        self.dual_templates = default(dual_templates, style_dual_templates)
        self.per_image_tokens = get_extra_tokens(per_image_tokens, len(self.images))
        self.mixing_prob = mixing_prob

    def __len__(self):
        return len(self.images) * self.multiplier

    def __getitem__(self, i: int):
        src_data = self.images[i % len(self.images)]

        using_dual = len(self.per_image_tokens) > 0 and np.random.uniform() < self.mixing_prob
        caption = self.caption_dual(i) if using_dual else self.caption_single()

        return ClassicData(caption=caption, **src_data)

    @property
    def placeholder_string(self):
        return self.placeholder_token

    def caption_single(self):
        assert len(self.templates) > 0, "Cannot provide a caption; `templates` is empty!"
        text = random.choice(self.templates)
        return text.format(self.placeholder_string, self.placeholder_token)

    def caption_dual(self, i: int):
        len_extra = len(self.per_image_tokens)
        # fail over to a single caption if we can't do a double
        if len_extra == 0 and len(self.templates) > 0: return self.caption_single()
        # in case we can't fail safely
        assert len(self.dual_templates) > 0, "Cannot provide a dual caption; `dual_templates` is empty!"
        extra_token = self.per_image_tokens[i % len(self.images) % len_extra]
        text = random.choice(self.dual_templates)
        return text.format(self.placeholder_string, extra_token)


class ClassicSubject(ClassicStyle):
    def __init__(self,
                 coarse_class_text: Optional[str]=None,
                 templates: Optional[list[str]]=None,
                 dual_templates: Optional[list[str]]=None,
                 **kwargs
                 ):
        # use the subject templates as default instead of the style templates
        templates = default(templates, subject_templates)
        dual_templates = default(dual_templates, subject_dual_templates)
        super().__init__(templates=templates, dual_templates=dual_templates, **kwargs)

        self.coarse_class_text = coarse_class_text

    @property
    def placeholder_string(self):
        if self.coarse_class_text:
            return f"{self.coarse_class_text} {self.placeholder_token}"
        return self.placeholder_token


# constants for config parameter locations
SUPPORTED_KLASSES = [
    "ldm.data.classic.ClassicSubject",
    "ldm.data.classic.ClassicStyle"
]

PERS_CONFIG = ConfigInstantiable(
    path="model.params.personalization_config",
    klasses=["ldm.modules.embedding_manager.EmbeddingManager"]
)
IMAGES_CONFIG = ConfigInstantiable(
    path="data.params.images",
    klasses=["ldm.data.image_loader.ImageLoader"]
)
DATA_CONFIGS: dict[DatasetKey, ConfigInstantiable] = {
    "train": ConfigInstantiable(
        path="data.params.train",
        klasses=SUPPORTED_KLASSES
    ),
    "validation": ConfigInstantiable(
        path="data.params.validation",
        klasses=SUPPORTED_KLASSES
    ),
    "test": ConfigInstantiable(
        path="data.params.test",
        klasses=SUPPORTED_KLASSES
    ),
    "predict": ConfigInstantiable(
        path="data.params.predict",
        klasses=SUPPORTED_KLASSES
    )
}


class ClassicDataModule(BasicDataModule):
    def __init__(self,
                 images: DictConfig,
                 templates: Optional[list[str]]=None,
                 dual_templates: Optional[list[str]]=None,
                 **kwargs
                ):
        super().__init__(**kwargs)

        self.extra_config_args = compact_dict({
            "images": IMAGES_CONFIG.direct().instantiate(images),
            "templates": templates,
            "dual_templates": dual_templates,
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
        operate_on_config(config,
            # apply custom cli for model
            ("set_as", f"{pp}.embedding_manager_ckpt", opt.embedding_manager_ckpt),
            ("set_as", f"{pp}.placeholder_strings", [opt.placeholder_string] if opt.placeholder_string else None),
            ("default_as", f"{pp}.initializer_words", [opt.init_word] if opt.init_word else None),
            ("set_as", f"{pp}.initializer_words[0]", opt.init_word),
        )

        # normalize the data configs; only work on expected data classes
        for dc in DATA_CONFIGS:
            dt = OmegaConf.select(config, dc.target, default="")
            if not dc.supports(dt): continue
            dp = dc.params
            operate_on_config(config,
                ("set_from", f"{dp}.per_image_tokens", f"{pp}.per_image_tokens")
            )