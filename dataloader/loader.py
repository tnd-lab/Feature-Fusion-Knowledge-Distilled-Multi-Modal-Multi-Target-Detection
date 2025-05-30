import torch
from timm.data.distributed_sampler import OrderedDistributedSampler
from .transform import *

MAX_NUM_INSTANCES = 100
""" Multi-Scale RandomErasing

Copyright 2020 Ross Wightman
"""
import random
import math
import torch


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is tweaked for multi-scale obj detection training.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1 / 4, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, thermal_img, rgb_img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    thermal_img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    rgb_img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def __call__(self, thermal_input, rgb_input, target):
        batch_size, chan, input_h, input_w = thermal_input.shape
        img_scales = target['img_scale']
        img_size = (target['img_size'] / img_scales.unsqueeze(1)).int()
        img_size[:, 0] = img_size[:, 0].clamp(max=input_w)
        img_size[:, 1] = img_size[:, 1].clamp(max=input_h)
        # skip first slice of batch if num_splits is set (for clean portion of samples)
        batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
        for i in range(batch_start, batch_size):
            self._erase(thermal_input[i], rgb_input[i], chan, img_size[i, 1], img_size[i, 0], thermal_input.dtype)
        return thermal_input, rgb_input


class DetectionFastCollate:
    """ A detection specific, optimized collate function w/ a bit of state.
    Optionally performs anchor labelling. Doing this here offloads some work from the
    GPU and the main training process thread and increases the load on the dataloader
    threads.

    """

    def __init__(
            self,
            instance_keys=None,
            instance_shapes=None,
            instance_fill=-1,
            max_instances=MAX_NUM_INSTANCES,
            anchor_labeler=None,
    ):
        instance_keys = instance_keys or {'bbox', 'bbox_ignore', 'cls', 'difficult', 'truncated', 'occluded'}
        instance_shapes = instance_shapes or dict(
            bbox=(max_instances, 4), bbox_ignore=(max_instances, 4), cls=(max_instances,), difficult=(max_instances,),
            truncated=(max_instances,), occluded=(max_instances,))
        self.instance_info = {k: dict(fill=instance_fill, shape=instance_shapes[k]) for k in instance_keys}
        self.max_instances = max_instances
        self.anchor_labeler = anchor_labeler

    def __call__(self, batch):
        batch_size = len(batch)
        target = dict()
        labeler_outputs = dict()
        thermal_img_tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        rgb_img_tensor = torch.zeros((batch_size, *batch[0][1].shape), dtype=torch.float32)

        for i in range(batch_size):
            thermal_img_tensor[i] += torch.from_numpy(batch[i][0])
            rgb_img_tensor[i] += torch.from_numpy(batch[i][1])
            labeler_inputs = {}
            for tk, tv in batch[i][2].items():
                instance_info = self.instance_info.get(tk, None)
                if instance_info is not None:
                    # target tensor is associated with a detection instance
                    tv = torch.from_numpy(tv).to(dtype=torch.float32)
                    if self.anchor_labeler is None:
                        if i == 0:
                            shape = (batch_size,) + instance_info['shape']
                            target_tensor = torch.full(shape, instance_info['fill'], dtype=torch.float32)
                            target[tk] = target_tensor
                        else:
                            target_tensor = target[tk]
                        num_elem = min(tv.shape[0], self.max_instances)
                        target_tensor[i, 0:num_elem] = tv[0:num_elem]
                    else:
                        # no need to pass gt tensors through when labeler in use
                        if tk in ('bbox', 'cls'):
                            labeler_inputs[tk] = tv
                else:
                    # target tensor is an image-level annotation / metadata
                    if i == 0:
                        # first batch elem, create destination tensors
                        if isinstance(tv, (tuple, list)):
                            # per batch elem sequence
                            shape = (batch_size, len(tv))
                            dtype = torch.float32 if isinstance(tv[0], (float, np.floating)) else torch.int32
                        else:
                            # per batch elem scalar
                            shape = batch_size,
                            dtype = torch.float32 if isinstance(tv, (float, np.floating)) else torch.int64
                        target_tensor = torch.zeros(shape, dtype=dtype)
                        target[tk] = target_tensor
                    else:
                        target_tensor = target[tk]
                    target_tensor[i] = torch.tensor(tv, dtype=target_tensor.dtype)

            if self.anchor_labeler is not None:
                cls_targets, box_targets, num_positives = self.anchor_labeler.label_anchors(
                    labeler_inputs['bbox'], labeler_inputs['cls'], filter_valid=False)
                if i == 0:
                    # first batch elem, create destination tensors, separate key per level
                    for j, (ct, bt) in enumerate(zip(cls_targets, box_targets)):
                        labeler_outputs[f'label_cls_{j}'] = torch.zeros(
                            (batch_size,) + ct.shape, dtype=torch.int64)
                        labeler_outputs[f'label_bbox_{j}'] = torch.zeros(
                            (batch_size,) + bt.shape, dtype=torch.float32)
                    labeler_outputs['label_num_positives'] = torch.zeros(batch_size)
                for j, (ct, bt) in enumerate(zip(cls_targets, box_targets)):
                    labeler_outputs[f'label_cls_{j}'][i] = ct
                    labeler_outputs[f'label_bbox_{j}'][i] = bt
                labeler_outputs['label_num_positives'][i] = num_positives
        if labeler_outputs:
            target.update(labeler_outputs)

        return thermal_img_tensor, rgb_img_tensor, target


class PrefetchLoader:

    def __init__(self,
                 loader,
                 rgb_mean=IMAGENET_DEFAULT_MEAN,
                 rgb_std=IMAGENET_DEFAULT_STD,
                 thermal_mean=IMAGENET_DEFAULT_MEAN,
                 thermal_std=IMAGENET_DEFAULT_STD,
                 re_prob=0.,
                 re_mode='pixel',
                 re_count=1,
                 ):
        self.loader = loader
        self.rgb_mean = torch.tensor([x * 255 for x in rgb_mean]).cuda().view(1, 3, 1, 1)
        self.rgb_std = torch.tensor([x * 255 for x in rgb_std]).cuda().view(1, 3, 1, 1)
        self.thermal_mean = torch.tensor([x * 255 for x in thermal_mean]).cuda().view(1, 3, 1, 1)
        self.thermal_std = torch.tensor([x * 255 for x in thermal_std]).cuda().view(1, 3, 1, 1)

        if re_prob > 0.:
            self.random_erasing = RandomErasing(probability=re_prob, mode=re_mode, max_count=re_count)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_thermal_input, next_rgb_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_thermal_input = next_thermal_input.cuda(non_blocking=True)
                next_thermal_input = next_thermal_input.float().sub_(self.thermal_mean).div_(self.thermal_std)
                next_rgb_input = next_rgb_input.cuda(non_blocking=True)
                next_rgb_input = next_rgb_input.float().sub_(self.rgb_mean).div_(self.rgb_std)
                next_target = {k: v.cuda(non_blocking=True) for k, v in next_target.items()}
                if self.random_erasing is not None:
                    next_thermal_input, next_rgb_input = self.random_erasing(next_thermal_input, next_rgb_input,
                                                                             next_target)

            if not first:
                yield thermal_input, rgb_input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            thermal_input = next_thermal_input
            rgb_input = next_rgb_input
            target = next_target

        yield thermal_input, rgb_input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        re_prob=0.,
        re_mode='pixel',
        re_count=1,
        interpolation='bilinear',
        fill_color='mean',
        rgb_mean=IMAGENET_DEFAULT_MEAN,
        rgb_std=IMAGENET_DEFAULT_STD,
        thermal_mean=IMAGENET_DEFAULT_MEAN,
        thermal_std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        pin_mem=False,
        anchor_labeler=None,
        collate_fn=None
):
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    if is_training:
        transform = transforms_train(
            img_size,
            interpolation=interpolation,
            fill_color=fill_color,
            rgb_mean=rgb_mean,
            thermal_mean=thermal_mean)
    else:
        transform = transforms_eval(
            img_size,
            interpolation=interpolation,
            fill_color=fill_color,
            rgb_mean=rgb_mean,
            thermal_mean=thermal_mean)

    dataset.transform = transform

    sampler = None
    if distributed:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = OrderedDistributedSampler(dataset)

    collate_fn = collate_fn or DetectionFastCollate(anchor_labeler=anchor_labeler)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
    )
    if use_prefetcher:
        if is_training:
            loader = PrefetchLoader(loader, rgb_mean=rgb_mean, rgb_std=rgb_std, thermal_mean=thermal_mean, thermal_std=thermal_std, re_prob=re_prob, re_mode=re_mode, re_count=re_count)
        else:
            loader = PrefetchLoader(loader, rgb_mean=rgb_mean, rgb_std=rgb_std, thermal_mean=thermal_mean, thermal_std=thermal_std)

    return loader
