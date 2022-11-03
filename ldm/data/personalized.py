from typing import Optional
from argparse import Namespace

from omegaconf import DictConfig, OmegaConf

import ldm.data.image_loader as di
import ldm.data.classic as dc
from ldm.config import DatasetKey, ConfigInstantiable, BasicDataModule, operate_on_config
from ldm.util import compact_dict

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

        # The word `repeats` is used as a bit of a misnomer.  It's treated
        # as a multiplier but reads like, "I want 1 repeat."  So, you want two
        # of each image, right?  Repeating each one once?  Well, for backward
        # compatibility, we're keeping that meaning here, but the new stuff
        # just uses the term `multiplier`.
        super().__init__(images=images, multiplier=repeats, **kwargs)

    @property
    def num_images(self):
        return len(self.images)


# constants for config parameter locations
SUPPORTED_KLASSES = [
    "ldm.data.personalized.PersonalizedBase",
    "ldm.data.personalized_style.PersonalizedBase"
]

PERS_CONFIG = ConfigInstantiable(
    path="model.params.personalization_config",
    klasses=["ldm.modules.embedding_manager.EmbeddingManager"]
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


class PersonalizedDataModule(BasicDataModule):
    def __init__(self,
                 templates: Optional[list[str]]=None,
                 dual_templates: Optional[list[str]]=None,
                 **kwargs
                ):
        super().__init__(**kwargs)

        self.extra_config_args = compact_dict({
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
        for dc in DATA_CONFIGS.values():
            dt = OmegaConf.select(config, dc.target, default="")
            if not dc.supports(dt): continue
            dp = dc.params
            operate_on_config(config,
                ("set_from", f"{dp}.per_image_tokens", f"{pp}.per_image_tokens"),
                ("set_as", f"{dp}.data_root", opt.data_root),
            )