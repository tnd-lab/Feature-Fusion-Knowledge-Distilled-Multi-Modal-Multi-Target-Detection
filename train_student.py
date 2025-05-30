import tqdm
import numpy as np
import torch
from timm.models import load_checkpoint
from timm.utils import AverageMeter, CheckpointSaver, get_outdir
from dataloader.loader import create_loader
from dataloader import resolve_input_config
from models.loss import HybridDistillationLoss
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


def freeze(network, freeze_layer):
    for name, param in network.named_parameters():
        if freeze_layer not in name:
            param.requires_grad = False


if __name__ == '__main__':
    class Args:
        branch = 'fusion'
        root = 'dataset'
        dataset = 'flir_aligned_full'
        img_size = None
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
        thermal_mean = [0.519, 0.519, 0.519]
        thermal_std = [0.225, 0.225, 0.225]
        interpolation = 'bilinear'
        fill_color = None
        num_classes = 3
        student_thermal_checkpoint_path = 'output/student/thermal/large_100/train_flir/EXP_FLIR_ALIGNED_THERMAL_CBAM/model_last.pth'
        student_rgb_checkpoint_path = 'output/student/rgb/large_100/train_flir/EXP_FLIR_ALIGNED_RGB_CBAM/model_last.pth'
        freeze_layer = 'fusion_cbam'
        epochs = 100
        gpu = 0
        att_type = 'cbam'
        batch_size = 16
        workers = 8
        pin_mem = True
        teacher_checkpoint = 'output/teacher/fusion/train_flir/EXP_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar'
        student_checkpoint = None
        output = ''
        save = 'EXP2'
        model = 'tf_efficientdet_d1'
        student_backbone = 'tf_mobilenetv3_large_100.in1k'
        lambda_reg = 0.5
        temperature = 4.0
        lambda_cls = 0.5
        alpha = 0.2
        beta = 0.4
        gamma = 0.4
        prefetcher = True
        wandb = False



    args = Args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset, val_dataset = create_dataset(args.dataset, args.root)
    teacher_net = AttentionFusionNet(args, model_type='teacher')
    student_net = AttentionFusionNet(args, model_type='student')

    teacher_training_bench = DetBenchTrainImagePair(teacher_net, create_labeler=True)
    for param in teacher_training_bench.parameters():
        param.requires_grad = False

    student_training_bench = DetBenchTrainImagePair(student_net, create_labeler=True)

    student_training_bench.to(device)
    teacher_training_bench.to(device)
    # freeze(student_training_bench, args.freeze_layer)

    model_config = student_net.config
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

    optimizer = torch.optim.Adam(student_net.parameters(), lr=1e-3, weight_decay=0.0001)
    loss_fn = HybridDistillationLoss(args)
    evaluator = create_evaluator(args.dataset, val_dataset, distributed=False, pred_yxyx=False)

    if args.teacher_checkpoint:
        load_checkpoint(teacher_net, args.teacher_checkpoint)
        print('Loaded teacher checkpoint from ', args.teacher_checkpoint)

    if args.student_checkpoint:
        load_checkpoint(student_net, args.student_checkpoint)
        print('Loaded student checkpoint from ', args.student_checkpoint)
    else:
        if args.student_rgb_checkpoint_path:
            ckpt = torch.load(args.student_rgb_checkpoint_path, map_location=device, weights_only=False)
            student_net.rgb_backbone.load_state_dict(ckpt['rgb_backbone'])
            student_net.rgb_fpn.load_state_dict(ckpt['rgb_fpn'])
            student_net.rgb_box_net.load_state_dict(ckpt['rgb_box_net'])
            student_net.rgb_class_net.load_state_dict(ckpt['rgb_class_net'])
            print('Loading Student RGB from {}'.format(args.student_rgb_checkpoint_path))
        else:
            print('RGB Student checkpoint path not provided.')
        if args.student_thermal_checkpoint_path:
            ckpt = torch.load(args.student_thermal_checkpoint_path, map_location=device, weights_only=False)
            student_net.thermal_backbone.load_state_dict(ckpt['thermal_backbone'])
            student_net.thermal_fpn.load_state_dict(ckpt['thermal_fpn'])
            student_net.thermal_box_net.load_state_dict(ckpt['thermal_box_net'])
            student_net.thermal_class_net.load_state_dict(ckpt['thermal_class_net'])
            print('Loading Student Thermal from {}'.format(args.student_thermal_checkpoint_path))
        else:
            print('Thermal Student checkpoint path not provided.')

    for param in student_training_bench.model.thermal_backbone.parameters():
        param.requires_grad = False
    for param in student_training_bench.model.rgb_backbone.parameters():
        param.requires_grad = False

    full_backbone_params = count_parameters(student_training_bench.model.thermal_backbone) + count_parameters(
        student_training_bench.model.rgb_backbone)
    head_net_params = count_parameters(student_training_bench.model.fusion_class_net) + count_parameters(
        student_training_bench.model.fusion_box_net)
    bifpn_params = count_parameters(student_training_bench.model.rgb_fpn) + count_parameters(student_training_bench.model.thermal_fpn)
    full_params = count_parameters(student_training_bench.model)
    fusion_net_params = sum(
        [count_parameters(getattr(student_training_bench.model, "fusion_" + args.att_type + str(i))) for i in range(5)])

    print("*" * 50)
    print("Backbone Params : {}".format(full_backbone_params))
    print("Head Network Params : {}".format(head_net_params))
    print("BiFPN Params : {}".format(bifpn_params))
    print("Fusion Nets Params : {}".format(fusion_net_params))
    print("Total Model Parameters : {}".format(full_params))
    total_trainable_params = sum(p.numel() for p in student_training_bench.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_training_bench.parameters())
    print('Total Parameters: {:,} \nTotal Trainable: {:,}\n'.format(total_params, total_trainable_params))
    print("*" * 50)

    if args.wandb:
        import wandb
        config = dict()
        config.update({arg: getattr(args, arg) for arg in vars(args)})
        wandb.init(
          project='deep-sensor-fusion-'+args.att_type,
          config=config
        )
    # set up checkpoint saver
    output_base = args.output if args.output else f'./output/student/distilled'
    exp_name = args.save + "_" + args.dataset.upper() + "_" + args.att_type.upper()
    output_dir = get_outdir(output_base, 'train_flir', exp_name)
    saver = CheckpointSaver(
        student_net, optimizer, args=args, checkpoint_dir=output_dir)
    train_loss = []
    train_loss_det = []
    train_loss_feature = []
    train_loss_kd = []

    val_loss = []
    val_loss_det = []
    val_loss_feature = []
    val_loss_kd = []
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(0, args.epochs + 1):
        train_losses_m = AverageMeter()
        val_losses_m = AverageMeter()

        teacher_training_bench.eval()
        student_training_bench.train()
        set_eval_mode(student_training_bench, args.freeze_layer)
        try:
            pbar = tqdm.notebook.tqdm(train_dataloader)
        except:
            pbar = tqdm.tqdm(train_dataloader)
        batch_train_loss = []
        batch_train_loss_det = []
        batch_train_loss_feature = []
        batch_train_loss_kd = []
        for batch in pbar:
            pbar.set_description('Epoch {}/{}'.format(epoch, args.epochs + 1))

            thermal_img_tensor, rgb_img_tensor, target = batch[0], batch[1], batch[2]

            with torch.no_grad():
                teacher_output = teacher_training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=False)
            student_output = student_training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=False)

            loss, loss_det, loss_feature, loss_kd = loss_fn(student_output, teacher_output, val=True)
            train_losses_m.update(loss.item(), thermal_img_tensor.size(0))
            batch_train_loss.append(loss.item())
            batch_train_loss_det.append(loss_det.item())
            batch_train_loss_feature.append(loss_feature.item())
            batch_train_loss_kd.append(loss_kd.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.wandb:
               visualize_target(train_dataset, target, wandb, args, 'train')

        train_loss.append(sum(batch_train_loss) / len(batch_train_loss))
        train_loss_det.append(sum(batch_train_loss_det) / len(batch_train_loss_det))
        train_loss_feature.append(sum(batch_train_loss_feature) / len(batch_train_loss_feature))
        train_loss_kd.append(sum(batch_train_loss_kd) / len(batch_train_loss_kd))

        student_training_bench.eval()
        with torch.no_grad():
            try:
                pbar = tqdm.notebook.tqdm(val_dataloader)
            except:
                pbar = tqdm.tqdm(val_dataloader)
            batch_val_loss = []
            batch_val_loss_det = []
            batch_val_loss_feature = []
            batch_val_loss_kd = []

            for batch in pbar:
                pbar.set_description('Validating...')
                thermal_img_tensor, rgb_img_tensor, target = batch[0], batch[1], batch[2]
                with torch.no_grad():
                    teacher_output = teacher_training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=False)
                student_output = student_training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=True)
                loss, loss_det, loss_feature, loss_kd  = loss_fn(student_output, teacher_output, eval=True)
                val_losses_m.update(loss.item(), thermal_img_tensor.size(0))
                batch_val_loss.append(loss.item())
                batch_val_loss_det.append(loss_det.item())
                batch_val_loss_feature.append(loss_feature.item())
                batch_val_loss_kd.append(loss_kd.item())
                evaluator.add_predictions(student_output['detections'], target)
                if args.wandb and epoch == args.epochs:
                    visualize_detections(val_dataset, student_output['detections'], target, wandb, args, 'val')

            val_loss.append(sum(batch_val_loss)/len(batch_val_loss))
            val_loss_det.append(sum(batch_val_loss_det)/len(batch_val_loss_det))
            val_loss_feature.append(sum(batch_val_loss_feature)/len(batch_val_loss_feature))
            val_loss_kd.append(sum(batch_val_loss_kd)/len(batch_val_loss_kd))


        if saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch=epoch, metric=evaluator.evaluate())
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
    plt.plot(train_loss_det, label='Training loss det')
    plt.plot(train_loss_feature, label='Training loss feature')
    plt.plot(train_loss_kd, label='Training loss kd')
    plt.plot(val_loss, label='Validation loss')
    plt.plot(val_loss_det, label='Validation loss det')
    plt.plot(val_loss_feature, label='Validation loss feature')
    plt.plot(val_loss_kd, label='Validation loss kd')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    import pandas as pd
    df = pd.DataFrame({
        'epoch': list(range(1, len(train_loss) + 1)),
        'train_loss': train_loss,
        'train_loss_det': train_loss_det,
        'train_loss_feature': train_loss_feature,
        'train_loss_kd': train_loss_kd,
        'val_loss': val_loss,
        'val_loss_det': val_loss_det,
        'val_loss_feature': val_loss_feature,
        'val_loss_kd': val_loss_kd
    })
    df.to_csv(os.path.join(output_dir, 'loss.csv'), index=False)
