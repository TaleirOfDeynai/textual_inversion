import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ldm.data.personalized import per_img_token_list

imagenet_templates_small = [
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

imagenet_dual_templates_small = [
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
        text = random.choice(templates).format(self.placeholder_token, extra_token)
            
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