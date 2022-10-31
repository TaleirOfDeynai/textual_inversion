from typing import Optional

import ldm.data.image_loader as di
import ldm.data.classic as dc

class PersonalizedBase(dc.ClassicSubject):
    """
    For backward compatibility purposes.

    Try to favor `ldm.data.classic.ClassicSubject` instead.
    """
    def __init__(self,
                 data_root: str,
                 size: Optional[int]=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 center_crop=False,
                 set: str="train",
                 repeats: int=100,
                 **kwargs
                 ):
        images = di.ImageLoader(data_root, size, interpolation, flip_p, center_crop)
        repeats = repeats if set == "train" else 1
        super().__init__(images=images, repeats=repeats, **kwargs)

    @property
    def num_images(self):
        return len(self.images)