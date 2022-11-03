from typing import Optional
from dataclasses import dataclass

import os
import random
import itertools
import numpy as np
from omegaconf import OmegaConf

import ldm.data.image_loader as di
import ldm.data.classic as dc
from ldm.util import default, partition
from torch.utils.data import Dataset

subject_conditions = [
    'a rendition',
    'a depiction',
    'a photo',
    'the photo',
    'a rendering',
    'a cropped photo',
    'a close-up photo',
    'a bright photo',
    'a dark photo',
    'a cropped photo',
    'a good photo',
    'an illustration',
]

subject_templates = [
    '{condition} of a {subject}',
    '{condition} of the {subject}',
    '{condition} of my {subject}',
    '{condition} of one {subject}',
    '{condition} of a {subject} with {quality}',
    '{condition} of the {subject} with {quality}',
    '{condition} of my {subject} with {quality}',
    '{condition} of one {subject} with {quality}',
    '{condition} of a clean {subject}',
    '{condition} of the clean {subject}',
    '{condition} of a clean {subject} with {quality}',
    '{condition} of the clean {subject} with {quality}',
    '{condition} of a dirty {subject}',
    '{condition} of the dirty {subject}',
    '{condition} of a dirty {subject} with {quality}',
    '{condition} of the dirty {subject} with {quality}',
    '{condition} of a cool {subject}',
    '{condition} of the cool {subject}',
    '{condition} of a cool {subject} with {quality}',
    '{condition} of the cool {subject} with {quality}',
    '{condition} of a nice {subject}',
    '{condition} of the nice {subject}',
    '{condition} of a nice {subject} with {quality}',
    '{condition} of the nice {subject} with {quality}',
    '{condition} of a weird {subject}',
    '{condition} of the weird {subject}',
    '{condition} of a weird {subject} with {quality}',
    '{condition} of the weird {subject} with {quality}',
    '{condition} of a small {subject}',
    '{condition} of the small {subject}',
    '{condition} of a small {subject} with {quality}',
    '{condition} of the small {subject} with {quality}',
    '{condition} of a large {subject}',
    '{condition} of the large {subject}',
    '{condition} of a large {subject} with {quality}',
    '{condition} of the large {subject} with {quality}',
]

style_conditions = [
    'a painting',
    'a picture',
    'a rendering',
    'a cropped painting',
    'a close-up painting',
    'a bright painting',
    'a dark painting',
    'a cropped painting',
    'a good painting',
    'a nice painting',
    'a small painting',
    'a large painting',
    'a weird painting',
]

style_templates = [
    '{condition} in the style of {subject}',
    '{condition} in the style of {subject} with {quality}',
]


@dataclass
class MetadataConf:
    """Per-file metadata for each image."""
    conditions: list[str] = []
    qualities: list[str] = []
    templates: list[str] = []


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

def get_quality_tokens(qualities: list[str]):
    assert len(qualities) <= len(dc.extra_token_list), f"Loaded {len(qualities)} unique qualities, but only have {len(dc.extra_token_list)} extra tokens. Try adding more tokens to 'extra_token_list'."
    result: dict[str, str] = {}
    for i, v in enumerate(qualities):
        result[v] = dc.extra_token_list[i]
    return result

def is_dual_template(template: str):
    return "\{quality\}" in template

def pick_quality(qualities: list[str], mixing_prob: float, solo_count: int, dual_count: int):
    if dual_count == 0:
        assert solo_count > 0, "No templates were available to properly construct a conditioning caption."
        return None
    if len(qualities) == 0:
        assert solo_count > 0, "An image's metadata had an empty `qualities` array, but no template works without a quality."
        return None
    if np.random.uniform() >= mixing_prob:
        return None
    return random.choice(qualities)


class ManagedBase(Dataset):
    def __init__(self, images: di.ImageLoader, **kwargs):
        super().__init__(**kwargs)

        # Load the per-image metadata.
        self.images = list(map(load_image_metadata, images))

        # Generate a list of known qualities.
        # We have a resume issue I'm not sure how to resolve.  If the user adds
        # a new quality, it can change how the qualities are mapped to tokens.
        # Those tokens may already have a lot of training, which would be
        # upset by a sudden change in mappings.
        all_qualities = list(set(itertools.chain(*map(
            lambda img: img["metadata"].qualities,
            self.images
        )))).sort()

        # Assign them each a placeholder string.
        self.quality_to_placeholder = get_quality_tokens(all_qualities)

    def __getitem__(self, i: int):
        # Does not include the metadata.
        base_data = super().__getitem__(i)
        # So lets add it in.
        src_data = self.images[i % len(self.images)]
        return ImageDataWithMetadata(metadata=src_data["metadata"], **base_data)


class ManagedData(ImageDataWithMetadata):
    """An image with caption and used placeholders."""
    caption: str
    placeholders: list[str]


class ManagedStyle(ManagedBase):
    def __init__(self,
                 placeholder_token: str="*",
                 templates: Optional[list[str]]=None,
                 default_conditions: Optional[list[str]]=None,
                 extra_conditions: Optional[list[str]]=None,
                 mixing_prob=0.25,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.placeholder_token = placeholder_token
        self.default_conditions = default(default_conditions, style_conditions)
        self.extra_conditions = default(extra_conditions, [])
        self.base_templates = default(templates, style_templates)
        self.mixing_prob = mixing_prob

    def __getitem__(self, i: int):
        src_data = super().__getitem__(i)
        metadata = src_data["metadata"]

        base_conditions = metadata.conditions if len(metadata.conditions) > 0 else self.default_conditions
        all_conditions = set(base_conditions + self.extra_conditions)
        condition = random.choice(list(all_conditions))

        all_templates = set(self.base_templates + metadata.templates)
        parted_templates = partition(all_templates, is_dual_template)
        dual_templates = list(parted_templates[0])
        solo_templates = list(parted_templates[1])

        # This also handles some assertions.  If this is not `None`,
        # it is safe to use `dual_templates`.
        quality_key = pick_quality(metadata.qualities, self.mixing_prob, len(solo_templates), len(dual_templates))

        if quality_key is not None:
            extra_placeholder = self.quality_to_placeholder[quality_key]
            placeholders = [self.placeholder_string, extra_placeholder]
            text = random.choice(dual_templates)
            caption = text.format(
                condition=condition,
                subject=self.placeholder_string,
                quality=extra_placeholder
            )
        else:
            placeholders = [self.placeholder_string]
            text = random.choice(solo_templates)
            caption = text.format(
                condition=condition,
                subject=self.placeholder_string
            )

        return ManagedData(caption=caption, placeholders=placeholders, **src_data)

    @property
    def placeholder_string(self):
        return self.placeholder_token


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