import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, lambda_reg=0.5, temperature=4.0, lambda_cls=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        self.temperature = temperature
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, student_out, teacher_out):
        student_class_out, student_box_out = student_out['class_out'], student_out['box_out']
        teacher_class_out, teacher_box_out = teacher_out['class_out'], teacher_out['box_out']

        student_cls = torch.cat([s.permute(0, 2, 3, 1).reshape(s.shape[0], -1, s.shape[1]) for s in student_class_out],
                                dim=1)
        teacher_cls = torch.cat([t.permute(0, 2, 3, 1).reshape(t.shape[0], -1, t.shape[1]) for t in teacher_class_out],
                                dim=1)
        kd_class_loss = F.kl_div(
            F.log_softmax(student_cls / self.temperature, dim=-1),
            F.softmax(teacher_cls / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        student_box = torch.cat([s.permute(0, 2, 3, 1).reshape(s.shape[0], -1, 4) for s in student_box_out], dim=1)
        teacher_box = torch.cat([t.permute(0, 2, 3, 1).reshape(t.shape[0], -1, 4) for t in teacher_box_out], dim=1)
        kd_box_loss = self.smooth_l1(student_box, teacher_box)
        kd_loss = self.lambda_cls * kd_class_loss + self.lambda_reg * kd_box_loss

        return kd_loss


class FeatureDistillationLoss(nn.Module):
    def __init__(self, feature_weights=None):
        """
        feature_weights: List weights of feature map (P3, P4, P5, P6, P7).
        """
        super(FeatureDistillationLoss, self).__init__()
        self.feature_weights = feature_weights if feature_weights else [1.0] * 5

    def forward(self, student_out, teacher_out):
        teacher_features, student_features = teacher_out['att_out'], student_out['att_out']
        loss = 0.0
        for i in range(len(teacher_features)):  # P3 -> P7
            t_feat, s_feat = teacher_features[i], student_features[i]
            level_loss = F.mse_loss(s_feat, t_feat.detach())

            loss += self.feature_weights[i] * level_loss

        return loss


class HybridDistillationLoss(nn.Module):
    def __init__(self, args: argparse):
        super(HybridDistillationLoss, self).__init__()
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.feature_loss = FeatureDistillationLoss()
        self.kd_loss = KnowledgeDistillationLoss(lambda_reg=args.lambda_reg, temperature=args.temperature,
                                                 lambda_cls=args.lambda_cls)

    def forward(self, student_out, teacher_out):
        loss_feature = self.feature_loss(student_out, teacher_out)
        loss_kd = self.kd_loss(student_out, teacher_out)
        loss_det = student_out['loss']

        total_loss = self.alpha * loss_det + self.beta * loss_feature + self.gamma * loss_kd
        return total_loss, loss_det, loss_feature, loss_kd
