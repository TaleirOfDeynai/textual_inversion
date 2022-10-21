from typing import Any, Optional, Union, TypedDict

import os
import random
import numpy as np

from ldm.util import default
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

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
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

INTERPOLATIONS = {
    "linear": Image.LINEAR,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


class ImageSrc(TypedDict):
    """An image setup for processing."""
    path: str
    image: np.ndarray[Any, np.dtype[np.uint8]]


class ImageData(TypedDict):
    """An image ready to use in the dataset."""
    path: str
    image: np.ndarray[Any, np.dtype[np.float32]]


def get_images_in(data_root: str,
                  center_crop: bool=False,
                  size: Optional[int]=None,
                  interpolation: Optional[str]="bicubic"):
    resample = INTERPOLATIONS[interpolation]
    images: list[ImageSrc] = []
    for file_path in os.listdir(data_root):
        img_path = os.path.join(data_root, file_path)
        try:
            image = Image.open(img_path)
            image = image.convert("RGB") if not image.mode == "RGB" else image
            img = np.array(image, dtype=np.uint8)

            if center_crop:
                crop = min(img.shape[0], img.shape[1])
                h, w, = img.shape[0], img.shape[1]
                img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

            if size is not None:
                image = Image.fromarray(img)
                image = image.resize((size, size), resample=resample)
                img = np.array(image, dtype=np.uint8)

            images.append(ImageSrc(path=img_path, image=img))
        except UnidentifiedImageError:
            continue
    return images

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


class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root: str,
                 size: Optional[int]=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 center_crop=False,
                 ):

        self.data_root = data_root
        self.images = get_images_in(data_root, center_crop, size, interpolation)

        self.num_images = len(self.images)
        self._length = self.num_images * repeats if set == "train" else self.num_images
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        src_data = self.images[i % self.num_images]

        image = Image.fromarray(src_data["image"].copy())
        image = self.flip(image)
        image = np.array(image, dtype=np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return ImageData(path=src_data["path"], image=image)


class PersonalizedData(ImageData):
    """An image with caption and used placeholders."""
    caption: str
    placeholders: list[str]


class PersonalizedStyle(PersonalizedBase):
    def __init__(self,
                 placeholder_token: str="*",
                 per_image_tokens: Union[list[str], bool, None]=False,
                 mixing_prob=0.25,
                 templates: Optional[list[str]]=None,
                 dual_templates: Optional[list[str]]=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.placeholder_token = placeholder_token
        self.templates = default(templates, style_templates)
        self.dual_templates = default(dual_templates, style_dual_templates)
        self.per_image_tokens = get_extra_tokens(per_image_tokens, self.num_images)
        self.mixing_prob = mixing_prob

    def __getitem__(self, i: int):
        src_data = super().__getitem__(i)

        using_dual = len(self.per_image_tokens) > 0 and np.random.uniform() < self.mixing_prob
        caption, placeholders = self.caption_dual(i) if using_dual else self.caption_single()

        return PersonalizedData(caption=caption, placeholders=placeholders, **src_data)

    @property
    def placeholder_string(self):
        return self.placeholder_token

    def caption_single(self) -> tuple[str, list[str]]:
        assert len(self.templates) > 0, "Cannot provide a caption; `templates` is empty!"
        text = random.choice(self.templates)
        caption = text.format(self.placeholder_string, self.placeholder_token)
        return (caption, [self.placeholder_string])

    def caption_dual(self, i: int) -> tuple[str, list[str]]:
        len_extra = len(self.per_image_tokens)
        # fail over to a single caption if we can't do a double
        if len_extra == 0 and len(self.templates) > 0: return self.caption_single()
        # in case we can't fail safely
        assert len(self.dual_templates) > 0, "Cannot provide a dual caption; `dual_templates` is empty!"
        extra_token = self.per_image_tokens[i % self.num_images % len_extra]
        text = random.choice(self.dual_templates)
        caption = text.format(self.placeholder_string, extra_token)
        return (caption, [self.placeholder_string, extra_token])


class PersonalizedSubject(PersonalizedStyle):
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