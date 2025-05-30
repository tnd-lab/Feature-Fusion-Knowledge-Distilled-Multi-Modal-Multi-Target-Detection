import argparse
import torch
import torch.nn as nn
import timm
from effdet.efficientdet import BiFpn, HeadNet
import effdet
from effdet import EfficientDet
from models.fusion import CBAMLayer


##################################### Attention Fusion Net ###############################################
class AttentionFusionNet(nn.Module):

    def __init__(self, args: argparse, model_type: str = 'teacher'):
        super(AttentionFusionNet, self).__init__()
        self.config = effdet.config.model_config.get_efficientdet_config(args.model)
        self.config.num_classes = args.num_classes
        if model_type == 'teacher':
            thermal_det = EfficientDet(self.config, pretrained_backbone=True)
            rgb_det = EfficientDet(self.config, pretrained_backbone=True)
        else:
            thermal_det = StudentDet(self.config, backbone=args.student_backbone)
            rgb_det = StudentDet(self.config, backbone=args.student_backbone)

        self.thermal_backbone = thermal_det.backbone
        self.thermal_fpn = thermal_det.fpn
        self.thermal_class_net = thermal_det.class_net
        self.thermal_box_net = thermal_det.box_net

        self.rgb_backbone = rgb_det.backbone
        self.rgb_fpn = rgb_det.fpn
        self.rgb_class_net = rgb_det.class_net
        self.rgb_box_net = rgb_det.box_net

        fusion_det = EfficientDet(self.config)

        self.fusion_class_net = fusion_det.class_net
        self.fusion_box_net = fusion_det.box_net

        if args.branch == 'fusion':
            self.attention_type = args.att_type
            print("{} using {} attention.".format(model_type, self.attention_type))
            in_chs = fusion_det.config.fpn_channels
            for level in range(self.config.num_levels):
                if self.attention_type == "cbam":
                    self.add_module("fusion_" + self.attention_type + str(level), CBAMLayer(2 * in_chs))
                else:
                    raise ValueError('Attention type not supported.')

    def forward(self, data_pair, branch='fusion'):
        thermal_x, rgb_x = data_pair[0], data_pair[1]

        class_net = getattr(self, f'{branch}_class_net')
        box_net = getattr(self, f'{branch}_box_net')

        x = None
        if branch == 'fusion':
            thermal_x = self.thermal_backbone(thermal_x)
            rgb_x = self.rgb_backbone(rgb_x)

            thermal_x = self.thermal_fpn(thermal_x)
            rgb_x = self.rgb_fpn(rgb_x)

            out = []
            for i, (tx, vx) in enumerate(zip(thermal_x, rgb_x)):
                x = torch.cat((tx, vx), dim=1)
                attention = getattr(self, "fusion_" + self.attention_type + str(i))
                att = attention(x)
                out.append(att)
        else:
            fpn = getattr(self, f'{branch}_fpn')
            backbone = getattr(self, f'{branch}_backbone')
            if branch == 'thermal':
                x = thermal_x
            elif branch == 'rgb':
                x = rgb_x
            feats = backbone(x)
            out = fpn(feats)
        x_class = class_net(out)
        x_box = box_net(out)

        return x_class, x_box, out


class StudentDet(nn.Module):

    def __init__(self, config, backbone: str, backbone_indices=(2, 3, 4)):
        super().__init__()
        self.backbone = timm.create_model(backbone, features_only=True
                                          , out_indices=backbone_indices
                                          , pretrained=True)
        self.fpn = BiFpn(config, self.backbone.feature_info.get_dicts())
        self.class_net = HeadNet(config, num_outputs=config.num_classes)
        self.box_net = HeadNet(config, num_outputs=4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box
