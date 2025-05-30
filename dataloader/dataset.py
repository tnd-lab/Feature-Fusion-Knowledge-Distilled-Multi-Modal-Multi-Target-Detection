import torch.utils.data as data
from PIL import Image
from collections import OrderedDict
from pathlib import Path
from effdet.data.parsers import *
from effdet.data.parsers import create_parser
from .config import FLIRConfig, FlirAlignedRGBCfg, FlirAlignedThermalCfg


class FLIRDataset(data.Dataset):
    """ Fusion Dataset for Object Detection. Use with parsers for COCO
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, thermal_data_dir, rgb_data_dir, parser=None, parser_kwargs=None, transform=None):
        super(FLIRDataset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.thermal_data_dir = thermal_data_dir
        self.rgb_data_dir = rgb_data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (thermal_image, rgb_image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        thermal_img_path = self.thermal_data_dir / img_info['file_name']
        thermal_img = Image.open(thermal_img_path).convert('RGB')
        rgb_img_path = self.rgb_data_dir / img_info['file_name'].replace('PreviewData', 'RGB')
        rgb_img = Image.open(rgb_img_path).convert('RGB')
        if self.transform is not None:
            thermal_img, rgb_img, target = self.transform(thermal_img, rgb_img, target)

        return thermal_img, rgb_img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


class DetectionDataset(data.Dataset):
    """`Object Detection Dataset. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, data_dir, parser=None, parser_kwargs=None, transform=None):
        super(DetectionDataset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.data_dir = data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        # target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if 'scene' in img_info:
            target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']), img_scene=img_info['scene'])
        else:
            target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        img_path = self.data_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)

        # draw_img = img.transpose(1, 2, 0)
        # for box in target['bbox']:
        #     y1, x1, y2, x2 = box.astype(int)
        #     cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # cv2.imwrite('testimg.png', draw_img)
        return img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


def create_dataset(name, root, splits=('train', 'val')):
    """
    Create pytorch dataset and split to train and val set for Fusion dataset.
    :param root: root folder of dataset
    :param splits: split train and val set
    :return: dict of FLIRDataset
    """
    if isinstance(splits, str):
        splits = (splits,)
    name = name.lower()
    root = Path(root)

    # FLIR-Aligned Dataset
    if name == 'flir_aligned_full':
        dataset_cls = FLIRDataset
        datasets = OrderedDict()
        dataset_cfg = FLIRConfig()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('thermal', 'rgb')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )
    elif name == 'flir_aligned_thermal':
        dataset_cls = FLIRDataset
        datasets = OrderedDict()
        dataset_cfg = FlirAlignedThermalCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )

            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('thermal', 'rgb')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )
    elif name == 'flir_aligned_rgb':
        dataset_cls = FLIRDataset
        datasets = OrderedDict()
        dataset_cfg = FlirAlignedRGBCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )

            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('thermal', 'rgb')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )
    datasets = list(datasets.values())
    return datasets if len(datasets) > 1 else datasets[0]
