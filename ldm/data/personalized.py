import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# The interpolation placeholder `{0}` maps to the primary
# token for the subject being trained (usually "*") and
# `{1}` will map to the extra per-image token.

imagenet_templates_smallest = [
    'a photo of a {0}',
]

imagenet_templates_small = [
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

imagenet_dual_templates_small = [
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

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 templates=None,
                 dual_templates=None,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token
        self.templates = imagenet_templates_small if templates is None else templates
        self.dual_templates = imagenet_dual_templates_small if dual_templates is None else dual_templates

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        extra_token = per_img_token_list[i % self.num_images % len(per_img_token_list)]
        using_dual = self.per_image_tokens and np.random.uniform() < self.mixing_prob
        templates = self.templates if not using_dual else self.dual_templates
        text = random.choice(templates).format(self.placeholder_string, extra_token)
            
        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example

    @property
    def placeholder_string(self):
        if self.coarse_class_text:
            return f"{self.coarse_class_text} {self.placeholder_token}"
        return self.placeholder_token
