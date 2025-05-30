import os
import time
import torch
from contextlib import suppress
from timm.models import load_checkpoint
from timm.utils import AverageMeter

from dataloader.loader import create_loader
from dataloader import resolve_input_config
from dataloader.dataset import create_dataset
from models.bench import DetBenchPredictImagePair
from models.model import AttentionFusionNet
from utils.evaluator import create_evaluator
from utils.utils import visualize_detections

import numpy as np
np.float_ = np.float64


def count_parameters(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)


def set_eval_mode(network, freeze_layer):
    for name, module in network.named_modules():
        if freeze_layer not in name:
            module.eval()


def freeze(network, freeze_layer):
    for name, param in network.named_parameters():
        if freeze_layer not in name:
            param.requires_grad = False


if __name__ == '__main__':

    class Args:
        branch = 'rgb'
        root = 'dataset'
        dataset = 'flir_aligned_rgb'
        split = ('test',)
        model = 'tf_efficientdet_d1'
        save = 'EXP'
        num_classes = 3
        workers = 8
        batch_size = 16
        att_type = 'cbam'
        img_size = None
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
        thermal_mean = [0.519, 0.519, 0.519]
        thermal_std = [0.225, 0.225, 0.225]
        interpolation = 'bilinear'
        fill_color = None
        log_freq = 5
        checkpoint = 'output/teacher/thermal/train_flir/EXP_FLIR_ALIGNED_THERMAL_CBAM/model_best.pth'
        prefetcher = True
        pin_mem = True
        gpu = 0
        results = 'result.json'
        wandb = False
        model_type = 'teacher'
        student_backbone = 'tf_mobilenetv3_large_100.in1k'


    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    net = AttentionFusionNet(args, model_type=args.model_type)

    bench = DetBenchPredictImagePair(net)
    bench.to(device)
    model_config = bench.config
    input_config = resolve_input_config(args, model_config)

    dataset = create_dataset(args.dataset, args.root, args.split)

    loader = create_loader(
        dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        rgb_mean=input_config['rgb_mean'],
        rgb_std=input_config['rgb_std'],
        thermal_mean=input_config['thermal_mean'],
        thermal_std=input_config['thermal_std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem,
        is_training=False
    )

    evaluator = create_evaluator(args.dataset + "_eval", dataset, distributed=False, pred_yxyx=False)
    # load checkpoint
    if args.branch == 'fusion':
      if args.checkpoint:
          load_checkpoint(net, args.checkpoint)
          print('Loaded checkpoint from ', args.checkpoint)
    else:
      ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
      net.rgb_backbone.load_state_dict(ckpt['rgb_backbone'])
      net.thermal_backbone.load_state_dict(ckpt['thermal_backbone'])
      net.thermal_fpn.load_state_dict(ckpt['thermal_fpn'])
      net.thermal_box_net.load_state_dict(ckpt['thermal_box_net'])
      net.thermal_class_net.load_state_dict(ckpt['thermal_class_net'])
      net.rgb_fpn.load_state_dict(ckpt['rgb_fpn'])
      net.rgb_box_net.load_state_dict(ckpt['rgb_box_net'])
      net.rgb_class_net.load_state_dict(ckpt['rgb_class_net'])
      print('Loading from {}'.format(args.checkpoint))
    if args.wandb:
        import wandb
        config = dict()
        config.update({arg: getattr(args, arg) for arg in vars(args)})
        wandb.init(
          project='deep-sensor-fusion-'+args.att_type,
          config=config
        )
    bench.eval()
    batch_time = AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1

    amp_autocast = suppress

    with torch.no_grad():
        for i, (thermal_input, rgb_input, target) in enumerate(loader):
            with amp_autocast():
                output = bench(thermal_input, rgb_input, img_info=target, branch=args.branch)
            evaluator.add_predictions(output, target)
            if args.wandb:
                visualize_detections(dataset, output, target, wandb, args, 'test')
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0 or i == last_idx:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.5f}/s)  '
                    .format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=1/(thermal_input.size(0) / batch_time.avg))
                )
        evaluator.evaluate()

    # mean_ap = 0
    # if dataset.parser.has_labels:
    #     mean_ap = evaluator.evaluate(output_result_file=args.results)
    # else:
    #     evaluator.save(args.results)

    # print("*" * 50)
    # print("Mean Average Precision Obtained is : " + str(mean_ap))
    # print("*" * 50)
