import tqdm
import numpy as np
import torch
from timm.models import load_checkpoint
from timm.utils import AverageMeter, get_outdir
from dataloader.loader import create_loader
from dataloader import resolve_input_config
from models.model import AttentionFusionNet
from models.bench import DetBenchTrainImagePair
from dataloader.dataset import create_dataset
from utils.evaluator import create_evaluator
from utils.utils import visualize_detections, visualize_target
import os
import matplotlib.pyplot as plt


def count_parameters(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)


def set_eval_mode(network, freeze_layer):
    for name, module in network.named_modules():
        if freeze_layer not in name:
            module.eval()


if __name__ == '__main__':
    class Args:
        branch = 'fusion'
        root = 'dataset'
        split = ('train', 'val')
        dataset = 'flir_aligned_full'
        image_size = None

        # Other configurable parameters
        rgb_mean = [0.485, 0.456, 0.406]
        thermal_mean = [0.229, 0.224, 0.225]
        rgb_std = [0.485, 0.456, 0.406]
        thermal_std = [0.225, 0.225, 0.225]
        interpolation = 'bilinear'
        fill_color = None
        num_classes = 3
        teacher_thermal_checkpoint_path = 'output/teacher/thermal/train_flir/EXP_FLIR_ALIGNED_THERMAL_CBAM/model_best.pth'
        student_thermal_checkpoint_path = None
        teacher_rgb_checkpoint_path = None
        student_rgb_checkpoint_path = None
        epochs = 50
        gpu = 0
        att_type = 'cbam'
        batch_size = 16
        workers = 8
        pin_mem = True
        checkpoint = None
        output = ''
        save = 'EXP'
        student_backbone = 'tf_mobilenetv3_large_100.in1k'
        model = 'tf_efficientdet_d1'
        model_type = 'student'
        model_size = 'large_100'
        prefetcher = True
        wandb = False

        def __init__(self):
            # You can also initialize default or optional arguments here if needed
            pass

        @classmethod
        def from_args(cls, args):
            # Method to update the class variables from parsed arguments
            for key, value in vars(args).items():
                if hasattr(cls, key):
                    setattr(cls, key, value)

        @classmethod
        def display(cls):
            # Optional: To display the current configuration for debugging or logs
            for key, value in cls.__dict__.items():
                if not key.startswith("__"):
                    print(f"{key}: {value}")


    args = Args()

    args.student_backbone = args.student_backbone.replace('large_100', args.model_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset, val_dataset = create_dataset(args.dataset, args.root)
    net = AttentionFusionNet(args, model_type=args.model_type)

    training_bench = DetBenchTrainImagePair(net, create_labeler=True)
    # training_bench.cuda()
    training_bench.to(device)
    optimizer = torch.optim.Adam(training_bench.parameters(), lr=1e-3, weight_decay=0.0001)
    ckpt = None
    if args.model_type == 'teacher':
        if args.teacher_rgb_checkpoint_path:
            ckpt = torch.load(args.teacher_rgb_checkpoint_path, map_location=device, weights_only=False)
            net.rgb_backbone.load_state_dict(ckpt['rgb_backbone'])
            net.rgb_fpn.load_state_dict(ckpt['rgb_fpn'])
            net.rgb_box_net.load_state_dict(ckpt['rgb_box_net'])
            net.rgb_class_net.load_state_dict(ckpt['rgb_class_net'])
            print('Loading Teacher RGB from {}'.format(args.teacher_rgb_checkpoint_path))
        else:
            print('RGB Teacher checkpoint path not provided.')
        if args.teacher_thermal_checkpoint_path:
            ckpt = torch.load(args.teacher_rgb_checkpoint_path, map_location=device, weights_only=False)
            net.thermal_backbone.load_state_dict(ckpt['thermal_backbone'])
            net.thermal_fpn.load_state_dict(ckpt['thermal_fpn'])
            net.thermal_box_net.load_state_dict(ckpt['thermal_box_net'])
            net.thermal_class_net.load_state_dict(ckpt['thermal_class_net'])
            print('Loading Teacher Thermal from {}'.format(args.teacher_thermal_checkpoint_path))
        else:
            print('Thermal Teacher checkpoint path not provided.')
    else:
        if args.student_rgb_checkpoint_path:
            ckpt = torch.load(args.student_rgb_checkpoint_path, map_location=device, weights_only=False)
            net.rgb_backbone.load_state_dict(ckpt['rgb_backbone'])
            net.rgb_fpn.load_state_dict(ckpt['rgb_fpn'])
            net.rgb_box_net.load_state_dict(ckpt['rgb_box_net'])
            net.rgb_class_net.load_state_dict(ckpt['rgb_class_net'])
            print('Loading Student RGB from {}'.format(args.student_rgb_checkpoint_path))
        else:
            print('RGB Student checkpoint path not provided.')
        if args.student_thermal_checkpoint_path:
            ckpt = torch.load(args.student_thermal_checkpoint_path, map_location=device, weights_only=False)
            net.thermal_backbone.load_state_dict(ckpt['thermal_backbone'])
            net.thermal_fpn.load_state_dict(ckpt['thermal_fpn'])
            net.thermal_box_net.load_state_dict(ckpt['thermal_box_net'])
            net.thermal_class_net.load_state_dict(ckpt['thermal_class_net'])
            print('Loading Student Thermal from {}'.format(args.student_thermal_checkpoint_path))
        else:
            print('Thermal Student checkpoint path not provided.')

    for param in training_bench.model.thermal_backbone.parameters():
        param.requires_grad = False
    for param in training_bench.model.rgb_backbone.parameters():
        param.requires_grad = False

    full_backbone_params = count_parameters(training_bench.model.thermal_backbone) + count_parameters(
        training_bench.model.rgb_backbone)
    head_net_params = count_parameters(training_bench.model.fusion_class_net) + count_parameters(
        training_bench.model.fusion_box_net)
    bifpn_params = count_parameters(training_bench.model.rgb_fpn) + count_parameters(
        training_bench.model.thermal_fpn)
    full_params = count_parameters(training_bench.model)

    print("*" * 50)
    print("Backbone Params : {}".format(full_backbone_params))
    print("Head Network Params : {}".format(head_net_params))
    print("BiFPN Params : {}".format(bifpn_params))
    print("Total Model Parameters : {}".format(full_params))
    total_trainable_params = sum(p.numel() for p in training_bench.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in training_bench.parameters())
    print('Total Parameters: {:,} \nTotal Trainable: {:,}\n'.format(total_params, total_trainable_params))
    print("*" * 50)

    model_config = training_bench.config
    input_config = resolve_input_config(args, model_config)
    train_dataloader = create_loader(
        train_dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        rgb_mean=input_config['rgb_mean'],
        thermal_mean=input_config['thermal_mean'],
        num_workers=args.workers,
        pin_mem=args.pin_mem,
        is_training=True
    )

    val_dataloader = create_loader(
        val_dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        rgb_mean=input_config['rgb_mean'],
        thermal_mean=input_config['thermal_mean'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)
    if ckpt is not None:
        evaluator = ckpt['evaluator']
        train_loss = ckpt['train_loss']
        val_loss = ckpt['val_loss']
        ckpt_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Loaded optimizer, ckpt epoch, loss checkpoint')
    else:
        evaluator = create_evaluator(args.dataset, val_dataset, distributed=False, pred_yxyx=False)
        train_loss = []
        val_loss = []
        ckpt_epoch = 0
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    # set up checkpoint saver
    if args.model_type == 'student':
        output_base = args.output if args.output else f'./output/{args.model_type}/{args.branch}/{args.model_size}'
    else:
        output_base = args.output if args.output else f'./output/{args.model_type}/{args.branch}'
    exp_name = args.save + "_" + args.dataset.upper() + "_" + args.att_type.upper()
    output_dir = get_outdir(output_base, 'train_flir', exp_name)

    if args.wandb:
        import wandb

        config = dict()
        config.update({arg: getattr(args, arg) for arg in vars(args)})
        wandb.init(
            project='deep-sensor-fusion-' + args.att_type,
            config=config
        )

    for epoch in range(1, args.epochs + 1):
        if epoch <= ckpt_epoch:
            continue
        train_losses_m = AverageMeter()
        val_losses_m = AverageMeter()

        training_bench.train()
        # set_eval_mode(training_bench, args.freeze_layer)
        try:
            pbar = tqdm.notebook.tqdm(train_dataloader, leave=False)
        except:
            pbar = tqdm.tqdm(train_dataloader, leave=False)
        batch_train_loss = []
        for batch in pbar:
            pbar.set_description('Epoch {}/{}'.format(epoch, args.epochs + 1))

            thermal_img_tensor, rgb_img_tensor, target = batch[0], batch[1], batch[2]

            output = training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=False, branch=args.branch)
            loss = output['loss']
            train_losses_m.update(loss.item(), thermal_img_tensor.size(0))
            batch_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.wandb:
                visualize_target(train_dataset, target, wandb, args, 'train')

        train_loss.append(sum(batch_train_loss) / len(batch_train_loss))

        training_bench.eval()
        with torch.no_grad():
            try:
                pbar = tqdm.notebook.tqdm(val_dataloader, leave=False)
            except:
                pbar = tqdm.tqdm(val_dataloader, leave=False)
            batch_val_loss = []
            for batch in pbar:
                pbar.set_description('Validating...')
                thermal_img_tensor, rgb_img_tensor, target = batch[0], batch[1], batch[2]

                output = training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=True, branch=args.branch)
                loss = output['loss']
                val_losses_m.update(loss.item(), thermal_img_tensor.size(0))
                batch_val_loss.append(loss.item())
                evaluator.add_predictions(output['detections'], target)

                if args.wandb and epoch == args.epochs:
                    visualize_detections(val_dataset, output['detections'], target, wandb, args, 'val')

            val_loss.append(sum(batch_val_loss) / len(batch_val_loss))
            if sum(batch_val_loss) / len(batch_val_loss) < min(val_loss):
                torch.save({
                    'thermal_backbone': net.thermal_backbone.state_dict(),
                    'rgb_backbone': net.rgb_backbone.state_dict(),
                    'thermal_fpn': net.thermal_fpn.state_dict(),
                    'rgb_fpn': net.rgb_fpn.state_dict(),
                    'rgb_box_net': net.rgb_box_net.state_dict(),
                    'thermal_box_net': net.thermal_box_net.state_dict(),
                    'thermal_class_net': net.thermal_class_net.state_dict(),
                    'rgb_class_net': net.rgb_class_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'evaluator': evaluator,
                }, f'{output_dir}/model_best.pth')
        if epoch % 5 == 0:
            torch.save({
                'thermal_backbone': net.thermal_backbone.state_dict(),
                'rgb_backbone': net.rgb_backbone.state_dict(),
                'thermal_fpn': net.thermal_fpn.state_dict(),
                'rgb_fpn': net.rgb_fpn.state_dict(),
                'rgb_box_net': net.rgb_box_net.state_dict(),
                'thermal_box_net': net.thermal_box_net.state_dict(),
                'thermal_class_net': net.thermal_class_net.state_dict(),
                'rgb_class_net': net.rgb_class_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'evaluator': evaluator,
            }, f'{output_dir}/model_{epoch}.pth')
        torch.save({
            'thermal_backbone': net.thermal_backbone.state_dict(),
            'rgb_backbone': net.rgb_backbone.state_dict(),
            'thermal_fpn': net.thermal_fpn.state_dict(),
            'rgb_fpn': net.rgb_fpn.state_dict(),
            'rgb_box_net': net.rgb_box_net.state_dict(),
            'thermal_box_net': net.thermal_box_net.state_dict(),
            'thermal_class_net': net.thermal_class_net.state_dict(),
            'rgb_class_net': net.rgb_class_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'evaluator': evaluator,
        }, f'{output_dir}/model_last.pth')
        evaluator.evaluate()
        current_val_loss = val_loss[-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'No improvement in validation loss for {epochs_no_improve} epoch(s).')
        if epochs_no_improve >= patience:
            print(f'\nEarly stopping triggered after {epoch} epochs.')
            break

    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    import pandas as pd

    df = pd.DataFrame({
        'epoch': list(range(1, len(train_loss) + 1)),
        'train_loss': train_loss,
        'val_loss': val_loss
    })
    df.to_csv(os.path.join(output_dir, 'loss.csv'), index=False)
