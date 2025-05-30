from effdet.data.dataset_config import *


@dataclass
class FLIRConfig(CocoCfg):
    """
    Dataclass for FLIR dataset configuration.
    Path to dataset folder.
    """
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='FLIR_Aligned/meta/thermal/flir_train.json',
                   img_dir='FLIR_Aligned/images_thermal_train/data/', has_labels=True),
        val=dict(ann_filename='FLIR_Aligned/meta/thermal/flir_test.json',
                 img_dir='FLIR_Aligned/images_thermal_test/data/', has_labels=True),
        test=dict(ann_filename='FLIR_Aligned/meta/thermal/flir_test.json',
                  img_dir='FLIR_Aligned/images_thermal_test/data/', has_labels=True),
    ))


@dataclass
class FlirAlignedThermalCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='FLIR_Aligned/meta/thermal/flir_train.json',
                   img_dir='FLIR_Aligned/images_thermal_train/data/', has_labels=True),
        val=dict(ann_filename='FLIR_Aligned/meta/thermal/flir_test.json',
                 img_dir='FLIR_Aligned/images_thermal_test/data/', has_labels=True),
        test=dict(ann_filename='FLIR_Aligned/meta/thermal/flir_test.json',
                  img_dir='FLIR_Aligned/images_thermal_test/data/', has_labels=True),
    ))


@dataclass
class FlirAlignedRGBCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='FLIR_Aligned/meta/rgb/rgb-train-flir.json',
                   img_dir='FLIR_Aligned/images_rgb_train/data/', has_labels=True),
        val=dict(ann_filename='FLIR_Aligned/meta/rgb/rgb-test-flir.json', img_dir='FLIR_Aligned/images_rgb_test/data/',
                 has_labels=True),
        test=dict(ann_filename='FLIR_Aligned/meta/rgb/rgb-test-flir.json', img_dir='FLIR_Aligned/images_rgb_test/data/',
                  has_labels=True),
    ))
