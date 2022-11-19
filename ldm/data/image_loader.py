from typing import Any, Optional, TypedDict, Sequence

import os
import numpy as np

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

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
        except IsADirectoryError:
            continue
    return images


class ImageLoader(Dataset, Sequence):
    def __init__(self,
                 data_root: str,
                 size: Optional[int]=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 center_crop=False,
                ):

        self.data_root = data_root
        self.images = get_images_in(data_root, center_crop, size, interpolation)
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        src_data = self.images[i]

        image = Image.fromarray(src_data["image"].copy())
        image = self.flip(image)
        image = np.array(image, dtype=np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return ImageData(path=src_data["path"], image=image)