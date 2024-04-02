# Copyright (c) OpenMMLab. All rights reserved.
# Author: hjc
# OverWritten: extract_feat
# Written: FeatKDRotatedRetinaNet LogitKDRotatedRetinaNet FRSRotatedRetinaNet DIDRotatedRetinaNet DeFeatRotatedRetinaNet
#          FGDRotatedRetinaNet PAKDRotatedRetinaNet ReviewKDRotatedRetinaNet SPKDRotatedRetinaNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, kaiming_init, build_conv_layer, build_norm_layer

from mmdet.core import multi_apply 

from mmrotate.core import obb2xyxy

from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class RotatedRetinaNet(RotatedSingleStageDetector):
    """Implementation of Rotated `RetinaNet.`__

    __ <https://arxiv.org/abs/1708.02002>
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedRetinaNet,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)
    # overwrite
    def extract_feat(self, img, return_middle_results=False):
        """Directly extract features from the backbone+neck."""
        middle_infos = dict()
        x = self.backbone(img)
        middle_infos["backbone_features"] = x
        if self.with_neck:
            x = self.neck(x)
            middle_infos["neck_features"] = x
        if return_middle_results:
            return x, middle_infos
        else:
            return x

# DID
@ROTATED_DETECTORS.register_module()
class DIDRotatedRetinaNet(RotatedSingleStageDetector):
    def __init__(self,
                 distillation, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None):
        # update bbox_head
        bbox_head["loss_balance"] = distillation["loss_balance"]
        bbox_head["temperature"] = distillation["temperature"]
        bbox_head["factor"] = distillation["factor"]
        # detector init
        super(DIDRotatedRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.distillation = distillation
        self.fusion_cfg = self.distillation["fusion_cfg"]
        self.fusion_init()
        # To align feature dim between student and teacher
        # self.tea_adapt = ConvModule(
        #     in_channels=distillation["in_channels"],
        #     out_channels=distillation["feat_channels"],
        #     kernel_size=1,
        #     conv_cfg=distillation["conv_cfg"],
        #     norm_cfg=distillation["norm_cfg"],
        #     act_cfg=distillation["act_cfg"]
        # )

    def fusion_init(self):
        conv_cfg = self.fusion_cfg["conv_cfg"]
        norm_cfg = self.fusion_cfg["norm_cfg"]
        in_channels = self.fusion_cfg["in_channels"]
        feat_channels = self.fusion_cfg["feat_channels"]

        self.CONV1 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                feat_channels,
                kernel_size=1,
                stride=1,
                bias=False,                
            ),
            build_norm_layer(
                norm_cfg,
                feat_channels,                
            )[1],
        )
        self.CONV2 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            build_norm_layer(
                norm_cfg,
                feat_channels,  
            )[1],
        )
        fusion_dim = 2
        self.Atten = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                feat_channels * 2,
                fusion_dim,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        nn.init.kaiming_uniform_(self.CONV1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.CONV2[0].weight, a=1)

    def fusion(self, x, y, att_conv):
        """
            Args:
                x: cur_feat, shape=[B, 256, Hi, Wi]
                y: pre_feat, shape=[B, 256, Hj, Wj]
                x's H, W large than y's H, W
        """
        B, D, H, W = x.shape
        x = self.CONV1(x)
        if att_conv:
            shape = x.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")
            atten_weights = self.Atten(torch.cat([x, y], dim=1))
            fusion = x * atten_weights[:, 0].view(B, 1, H, W) + y * atten_weights[:, 1].view(B, 1, H, W)
            x = fusion
            y = self.CONV2(fusion)
        else:
            y = self.CONV2(x)
        return y, x        

    def multi_layers_feature_fusion(self, stu_feats):
        """
            Args:
                stu_feats: list, len=num_levels|elements: tensor, shape=[B, C, Hi, Wi]
            Returns:
                stu_fusion_feats: list, len=Num_levels| elements: tensor, shape=[B, C, Hi, Wi]
        """
        num_layers = self.fusion_cfg["num_layers"]
        stu_fusion_res = []
        
        # multi-layers feature fusion
        stu_feats = stu_feats[::-1]  # sorted 
        out_feat, res_feat = self.fusion(stu_feats[0], stu_feats[0], att_conv=False)
        stu_fusion_res.append(out_feat)
        for i in range(1, num_layers):
            out_feat, res_feat = self.fusion(stu_feats[i], res_feat, att_conv=True)
            stu_fusion_res.insert(0, out_feat)
        
        return stu_fusion_res

    def train_step(self, data, optimizer, teacher):
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            teacher: model
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        with torch.no_grad():
            teacher = teacher.to(img.device)
            tea_feats = teacher.extract_feat(img)
            tea_cls_scores, tea_bbox_preds = teacher.bbox_head.forward(tea_feats)
        x = self.extract_feat(img)

        # multi-layers feature interaction
        stu_fusion_feats = self.multi_layers_feature_fusion(x)
        losses = self.bbox_head.forward_train(x, stu_fusion_feats, tea_feats, tea_cls_scores, tea_bbox_preds, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

        return losses

# Baseline
@ROTATED_DETECTORS.register_module()
class FeatKDRotatedRetinaNet(RotatedSingleStageDetector):
    """
        Implementation of FitNets
        <https://arxiv.org/abs/1412.6550>
    """
    def __init__(self,
                 distillation, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None):
        super(FeatKDRotatedRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.loss_balance = distillation["loss_balance"]

    def train_step(self, data, optimizer, teacher):
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    def feat_distillation(self, stu_feat, tea_feat):
        """
            Args:
                stu_feat: type=Tensor, shape=[B, C_S, H, W]
                tea_feat: type=Tensor, shape=[B, C_T, H, W]
        """
        # tea_feat = self.tea_adapt(tea_feat)
        loss_kd = F.mse_loss(tea_feat, stu_feat,reduction="mean")
        loss_kd = self.loss_balance * loss_kd
        
        return loss_kd, 

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            teacher: model
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        with torch.no_grad():
            teacher = teacher.to(img.device)
            tea_feats = teacher.extract_feat(img)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        # featKD
        losses_kd = multi_apply(
            self.feat_distillation,
            x,
            tea_feats,
        )[0]
        losses["loss_kd"] = losses_kd
        return losses

@ROTATED_DETECTORS.register_module()
class LogitKDRotatedRetinaNet(RotatedSingleStageDetector):
    """
        Implementation of LogitKD
        <https://arxiv.org/abs/1503.02531>
    """
    def __init__(self,
                 distillation,
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None):
        bbox_head["loss_balance"] = distillation["loss_balance"]
        bbox_head["temperature"] = distillation["temperature"]
        super(LogitKDRotatedRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)


    def train_step(self, data, optimizer, teacher):
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            teacher: model
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        with torch.no_grad():
            teacher = teacher.to(img.device)
            tea_feats = teacher.extract_feat(img)
            tea_cls_scores = teacher.bbox_head.forward(tea_feats)[0]
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, tea_cls_scores, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        return losses
    
@ROTATED_DETECTORS.register_module()
class FRSRotatedRetinaNet(RotatedSingleStageDetector):
    """
        Implementation of FRS
        <https://proceedings.neurips.cc/paper_files/paper/2021/file/29c0c0ee223856f336d7ea8052057753-Paper.pdf>
    """
    def __init__(self,
                 distillation, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None):
        super(FRSRotatedRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.loss_balance = distillation["loss_balance"]
    
    def train_step(self, data, optimizer, teacher):
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    def frs_distillation(self, stu_feat, stu_cls_score, tea_feat, tea_cls_score):
        """
            Args:
                stu_feat: type=Tensor, shape=[B, C_S, H, W]
                stu_cls_score: type=Tensor, shape=[B, num_anchors * num_classes, H, W]
                tea_feat: type=Tensor, shape=[B, C_T, H, W]
                tea_cls_score: type=Tensor, shape=[B, num_anchors * num_classes, H, W] 
        """
        with torch.no_grad():
            frs_mask = torch.max(tea_cls_score.sigmoid(), dim=1).values
        feat_loss = torch.pow((tea_feat - stu_feat), 2)
        cls_loss = F.binary_cross_entropy(stu_cls_score.sigmoid(), tea_cls_score.sigmoid(), reduction="none")
        loss_featkd = self.loss_balance[0] * (feat_loss * frs_mask[:, None, :, :]).sum() / frs_mask.sum()
        loss_clskd = self.loss_balance[1] * (cls_loss * frs_mask[:, None, :, :]).sum() / frs_mask.sum()
        
        return loss_featkd, loss_clskd


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            teacher: model
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        with torch.no_grad():
            teacher = teacher.to(img.device)
            tea_feats = teacher.extract_feat(img)
            tea_cls_scores = teacher.bbox_head.forward(tea_feats)[0]
        x = self.extract_feat(img)
        stu_cls_scores = self.bbox_head(x)[0]
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        # FRS
        losses_featkd, losses_clskd = multi_apply(
            self.frs_distillation,
            x,
            stu_cls_scores,
            tea_feats,
            tea_cls_scores
        )
        losses["loss_clskd"] = losses_clskd
        losses["loss_featkd"] = losses_featkd
        return losses

@ROTATED_DETECTORS.register_module()
class FGDRotatedRetinaNet(RotatedSingleStageDetector):
    """
        Implementation of FGD
        <https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.pdf>
    """
    def __init__(self,
                 distillation, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None):
        super(FGDRotatedRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        # init global layer
        self.distillation = distillation
        self.global_layer()
    
    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    def global_layer(self):
        feat_dim = self.distillation["feat_dim"]
        self.stu_conv_mask = nn.Conv2d(feat_dim, 1, kernel_size=1)
        self.tea_conv_mask = nn.Conv2d(feat_dim, 1, kernel_size=1)
        self.stu_channel_add_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim//2, kernel_size=1),
            nn.LayerNorm([feat_dim//2, 1, 1]),  
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(feat_dim//2, feat_dim, kernel_size=1))
        self.tea_channel_add_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim//2, kernel_size=1),
            nn.LayerNorm([feat_dim//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(feat_dim//2, feat_dim, kernel_size=1))
        # init
        kaiming_init(self.stu_conv_mask, mode='fan_in')
        kaiming_init(self.tea_conv_mask, mode='fan_in')
        self.stu_conv_mask.inited = True
        self.tea_conv_mask.inited = True

        self.last_zero_init(self.stu_channel_add_conv)
        self.last_zero_init(self.tea_channel_add_conv)

    def generate_attention(self, feats, factor):
        """
        Args:
            feats: tensor, shape=[B, C, H, W]
            factor: float
        Returns:
            spatial_attention: shape=[B, H, W]
            channel_attention: shape=[B, C]
        """
        B, C, H, W = feats.shape

        meaned_feats = torch.mean(torch.abs(feats), dim=1, keepdim=True)
        atten_s = (H * W * F.softmax((meaned_feats / factor).view(B, -1), dim=1)).view(B, H, W)

        channel_map = torch.abs(feats).mean(dim=2, keepdim=False).mean(dim=2, keepdim=False)
        atten_c = C * F.softmax(channel_map / factor, dim=1)
        
        return atten_s, atten_c

    def spatial_pool(self, feats, in_type):
        """
        Args:
            feats: shape=[B, C, H, W]
            in_type: teacher——1; student——0
        Returns:
            context: shape=[B, C, 1, 1]
        """
        B, C, H, W = feats.shape
        identity = feats
        identity = identity.view(B, C, H*W)
        identity = identity.unsqueeze(1)  # [B, 1, C, H*W]
        if in_type == 0:
            context_mask = self.stu_conv_mask(feats)
        else:
            context_mask = self.tea_conv_mask(feats)
        context_mask = context_mask.view(B, 1, H*W)
        context_mask = F.softmax(context_mask, dim=2)
        context_mask = context_mask.unsqueeze(-1)  # [B, 1, H*W, 1]
        context = torch.matmul(identity, context_mask)
        context = context.view(B, C, 1, 1)

        return context

    def generate_fg_and_bg_mask(self, size, gt_bboxes, img_metas, version):
        """
        Args:
            size: tuple (B, H, W)
            gt_bboxes: list, len=B|
                elements: tensor, shape=[num_gt_bboxes, 5] (xc, yc, w, h, theta)
            img_meats: list, len=B|
                elements: dict
                    img_shape: (H, W, 3)
        """
        B, H, W = size
        mask_fg = torch.zeros(size=size, device=gt_bboxes[0].device)
        mask_bg = torch.ones(size=size, device=gt_bboxes[0].device)
        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(B):
            num_bboxes = gt_bboxes[i].shape[0]
            gt_hbboxes = obb2xyxy(gt_bboxes[i], version)  # [num_gt_bboxes, 4] (x1, y1, x2, y2)
            new_hbboxes = torch.ones_like(gt_hbboxes)
            # gt bboxes map to featmap
            new_hbboxes[:, 0] = gt_hbboxes[:, 0] * W / img_metas[i]["img_shape"][1]
            new_hbboxes[:, 2] = gt_hbboxes[:, 2] * W / img_metas[i]["img_shape"][1]
            new_hbboxes[:, 1] = gt_hbboxes[:, 1] * H / img_metas[i]["img_shape"][0]
            new_hbboxes[:, 3] = gt_hbboxes[:, 3] * H / img_metas[i]["img_shape"][0]
            # gt region
            wmin.append(torch.floor(new_hbboxes[:, 0]).int())
            wmax.append(torch.ceil(new_hbboxes[:, 2]).int())
            hmin.append(torch.floor(new_hbboxes[:, 1]).int())
            hmax.append(torch.ceil(new_hbboxes[:, 3]).int())
            # gt area
            area = 1.0 / (hmax[i].view(1,-1)+1 - hmin[i].view(1,-1)) / (wmax[i].view(1,-1)+1 - wmin[i].view(1,-1))  # [1, num_gt_bboxes]
            # get weight for per pixel
            for j in range(num_bboxes):
                mask_fg[i, hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
                    torch.maximum(mask_fg[i, hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0, j])
            mask_bg[i] = torch.where(mask_fg[i] > 0, 0, 1)
            if torch.sum(mask_bg[i]):
                mask_bg[i] /= torch.sum(mask_bg[i])
            
        return mask_fg, mask_bg
    
    def get_localkd_loss(self, stu_feats, tea_feats, stu_atten_s, stu_atten_c, tea_atten_s, tea_atten_c, mask_fg, mask_bg):
        """
        Args:
            stu_level_feats: tensor, shape=[B, C, H, W]
            tea_level_feats: tensor, shape=[B, C, H, W]
            stu_atten_s: tensor, shape=[B, H, W]
            stu_atten_c: tensor, shape=[B, C]
            tea_atten_s: tensor, shape=[B, H, W]
            tea_atten_c: tensor, shape=[B, C]
            mask_fg: tensor, shape=[B, H, W]
            mask_bg: tensor, shape=[B, H, W]
        Return:
            loss: loss_featkd 
        """
        tea_atten_c = tea_atten_c.unsqueeze(dim=-1).unsqueeze(dim=-1)  # [B, C, 1, 1]
        tea_atten_s = tea_atten_s.unsqueeze(dim=1)  # [B, 1, H, W] 

        mask_fg = mask_fg.unsqueeze(dim=1)  # [B, 1, H, W]
        mask_bg = mask_bg.unsqueeze(dim=1)  # [B, 1, H, W]
        # stu
        masked_stu_feats = torch.mul(stu_feats, torch.sqrt(tea_atten_s))
        masked_stu_feats = torch.mul(masked_stu_feats, torch.sqrt(tea_atten_c))
        fg_masked_stu_feats = torch.mul(masked_stu_feats, torch.sqrt(mask_fg))
        bg_masked_stu_feats = torch.mul(masked_stu_feats, torch.sqrt(mask_bg))
        # tea
        masked_tea_feats = torch.mul(tea_feats, torch.sqrt(tea_atten_s))
        masked_tea_feats = torch.mul(masked_tea_feats, torch.sqrt(tea_atten_c))
        fg_masked_tea_feats = torch.mul(masked_tea_feats, torch.sqrt(mask_fg))
        bg_masked_tea_feats = torch.mul(masked_tea_feats, torch.sqrt(mask_bg))
        # fg loss
        loss_fg = F.mse_loss(fg_masked_stu_feats, fg_masked_tea_feats, reduction="sum") / mask_fg.shape[0]
        loss_bg = F.mse_loss(bg_masked_stu_feats, bg_masked_tea_feats, reduction="sum") / mask_bg.shape[0]
        
        return loss_fg, loss_bg

    def get_attention_loss(self, stu_attention_s, stu_attention_c, tea_attention_s, tea_attention_c):
        """
        Args:
            stu_attention_s: tensor, shape=[B, H, W]
            stu_attention_c: tensor, shape=[B, C]
            tea_attention_s: tensor, shape=[B, H, W]
            tea_attention_c: tensor, shape=[B, C]
        Returns:
            loss_attention
        """
        B = tea_attention_s.shape[0]
        loss_attention = torch.sum(torch.abs(stu_attention_s - tea_attention_s))/B + torch.sum(torch.abs(stu_attention_c - tea_attention_c))/B

        return loss_attention

    def get_globalkd_loss(self, stu_feats, tea_feats):
        """
        Args:
            stu_level_feats: tensor, shape=[B, C, H, W]
            tea_level_feats: tensor, shape=[B, C, H, W]
        Returns:
            loss_global
        """
        stu_context = self.spatial_pool(stu_feats, in_type=0)
        tea_context = self.spatial_pool(tea_feats, in_type=1)

        stu_out = stu_feats
        tea_out = tea_feats

        stu_out = stu_out + self.stu_channel_add_conv(stu_context)
        tea_out = tea_out + self.tea_channel_add_conv(tea_context)
        
        loss_global = F.mse_loss(stu_out, tea_out, reduction="sum") / stu_feats.shape[0]

        return loss_global   
     
    def fgd_distillation(self, stu_feat, tea_feat, gt_bboxes, img_metas):
        """
            Args:
                stu_feat: type=Tensor, shape=[B, C_S, H, W]
                tea_feat: type=Tensor, shape=[B, C_T, H, W]
                gt_bboxes: list, len=B| elements: tensor, shape=[num_gt_bboxes, 5] (xc, yc, w, h, theta)
                img_metas: list, len=B| elments: dict
        """
        T = self.distillation["temp"]
        alpha_fgd, beta_fgd, gamma_fgd, eta_fgd = self.distillation["alpha"], self.distillation["beta"], self.distillation["gamma"], self.distillation["eta"]
        version = self.bbox_head.bbox_coder.angle_range

        B, C, H, W = stu_feat.shape
        
        # generate spatial attention、channel attention
        stu_atten_s, stu_atten_c = self.generate_attention(stu_feat, T)
        tea_atten_s, tea_atten_c = self.generate_attention(tea_feat, T)
        # generate mask
        mask_fg, mask_bg = self.generate_fg_and_bg_mask(tea_atten_s.shape, gt_bboxes, img_metas, version)
        # localkd loss
        loss_fg, loss_bg = self.get_localkd_loss(stu_feat, tea_feat, stu_atten_s, stu_atten_c, tea_atten_s, tea_atten_c, mask_fg, mask_bg)
        # attenion loss
        loss_attention = self.get_attention_loss(stu_atten_s, stu_atten_c, tea_atten_s, tea_atten_c)
        # globalkd loss
        loss_global = self.get_globalkd_loss(stu_feat, tea_feat)

        loss_kd = alpha_fgd * loss_fg + beta_fgd * loss_bg + gamma_fgd * loss_attention + eta_fgd * loss_global

        return loss_kd

    def train_step(self, data, optimizer, teacher):
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            teacher: model
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        with torch.no_grad():
            teacher = teacher.to(img.device)
            tea_feats = teacher.extract_feat(img)
            tea_cls_scores = teacher.bbox_head.forward(tea_feats)[0]
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        # FGD
        for idx, (s_feat, t_feat) in enumerate(zip(x, tea_feats)):
            loss_name = "loss_fgd_fpn_{}".format(idx)
            losses[loss_name] = self.fgd_distillation(s_feat, t_feat, gt_bboxes, img_metas)

        return losses

@ROTATED_DETECTORS.register_module()
class PAKDRotatedRetinaNet(RotatedSingleStageDetector):
    """
        Implementation of PAKD
        <https://arxiv.org/pdf/1612.03928.pdf>
    """
    def __init__(self,
                 distillation, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None):
        bbox_head["loss_balance"] = distillation["loss_balance"][1]
        bbox_head["temperature"] = distillation["temperature"]
        super(PAKDRotatedRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.loss_balance = distillation["loss_balance"][0]
    
    def train_step(self, data, optimizer, teacher):
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    def payattention_distillation(self, stu_feat, tea_feat):
        """
            Args:
                stu_feat: type=Tensor, shape=[B, C_S, H, W]
                tea_feat: type=Tensor, shape=[B, C_T, H, W]
        """
        B = stu_feat.shape[0]
        stu_atten = F.normalize(stu_feat.pow(2).mean(1).view(B, -1))
        tea_atten = F.normalize(tea_feat.pow(2).mean(1).view(B, -1))

        loss = torch.pow(stu_atten - tea_atten, 2).sum() * self.loss_balance
        
        return loss,


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            teacher: model
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        with torch.no_grad():
            teacher = teacher.to(img.device)
            tea_feats = teacher.extract_feat(img)
            tea_cls_scores = teacher.bbox_head.forward(tea_feats)[0]
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, tea_cls_scores, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        # PAKD
        losses_pakd = multi_apply(
            self.payattention_distillation,
            x,
            tea_feats,
        )[0]
        losses["loss_pakd"] = losses_pakd
        return losses

@ROTATED_DETECTORS.register_module()
class ReviewKDRotatedRetinaNet(RotatedSingleStageDetector):
    """
        Implementation of ReviewKD
        <https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.pdf>
    """
    def __init__(self,
                 distillation, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None):
        super(ReviewKDRotatedRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.distillation = distillation
        self.loss_balance = self.distillation["loss_balance"]
        self.review_cfg = self.distillation["review_cfg"]
        self.review_feats_layers()

    def review_feats_layers(self):
        in_channels = self.review_cfg["in_channels"]
        feat_channels = self.review_cfg["feat_channels"]
        conv_cfg = self.review_cfg["conv_cfg"]
        norm_cfg = self.review_cfg["norm_cfg"]
        
        self.CONV1 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                feat_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            build_norm_layer(  # 返回norm_name, norm
                norm_cfg,
                feat_channels,
            )[1],
        )
        self.CONV2 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            build_norm_layer(
                norm_cfg,
                feat_channels,  
            )[1],
        )
        fusion_dim = 2
        self.Atten = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                feat_channels * 2,
                fusion_dim,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        nn.init.kaiming_uniform_(self.CONV1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.CONV2[0].weight, a=1)      

    def fusion(self, x, y, att_conv):
        """
            Args:
                x: cur_feat, shape=[B, 256, Hi, Wi]
                y: pre_feat, shape=[B, 256, Hj, Wj]
                x's H, W large than y's H, W
        """
        B, D, H, W = x.shape
        x = self.CONV1(x)
        if att_conv:
            shape = x.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")
            atten_weights = self.Atten(torch.cat([x, y], dim=1))
            fusion = x * atten_weights[:, 0].view(B, 1, H, W) + y * atten_weights[:, 1].view(B, 1, H, W)
            x = fusion
            y = self.CONV2(fusion)
        else:
            y = self.CONV2(x)
        return y, x        
    
    def featkd(self, stu_feats, tea_feats):
        loss_all = 0.0
        for stu_feat, tea_feat in zip(stu_feats, tea_feats):
            B, D, H, W = stu_feat.shape
            loss = F.mse_loss(stu_feat, tea_feat, reduction="mean")  # 引入mask
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:  # 这个超参是否有需要调?
                if H <= l:
                    continue
                tmp_stu_feat = F.adaptive_avg_pool2d(stu_feat, (l, l))
                tmp_tea_feat = F.adaptive_avg_pool2d(tea_feat, (l, l))
                cnt /= 2.0 
                loss += F.mse_loss(tmp_stu_feat, tmp_tea_feat, reduction="mean") * cnt
                tot += cnt
            loss /= tot
            loss_all = loss_all + loss  # 每level所计算的损失是直接求和的 每个level的损失权重是否需要动态？

        return loss_all
        
    def reviewfeat_distillation(self, stu_feats, tea_feats):
        """
            Args:
                stu_feats: list, len=5| elements: shape=[B, 256, Hi, Wi]
                    [0]: shape=[B, 256, 256, 256]
                    [1]: shape=[B, 256, 128, 128]
                    [2]: shape=[B, 256, 64, 64]
                    [3]: shape=[B, 256, 32, 32]
                    [4]: shape=[B, 256, 16, 16]
                tea_feats: list, len=5| elements: shape=[B, 256, Hi, Wi]
                    [0]: shape=[B, 256, 256, 256]
                    [1]: shape=[B, 256, 128, 128]
                    [2]: shape=[B, 256, 64, 64]
                    [3]: shape=[B, 256, 32, 32]
                    [4]: shape=[B, 256, 16, 16]
        """
        num_layers = self.review_cfg["num_layers"]
        stu_fusion_results = []

        # stu fusion
        stu_feats = stu_feats[::-1]  # 倒序
        out_feat, res_feat = self.fusion(stu_feats[0], stu_feats[0], att_conv=False)
        stu_fusion_results.append(out_feat)
        for i in range(1, num_layers):
            out_feat, res_feat = self.fusion(stu_feats[i], res_feat, att_conv=True)
            stu_fusion_results.insert(0, out_feat)
        """
            stu_fusion_results: list, len=5| elements: shape=[B, 256, Hi, Wi]
                [0]: shape=[B, 256, 256, 256]
                [1]: shape=[B, 256, 128, 128]
                [2]: shape=[B, 256, 64, 64]
                [3]: shape=[B, 256, 32, 32]
                [4]: shape=[B, 256, 16, 16]
        """

        # ReviewKD
        loss_kd = self.featkd(stu_fusion_results, tea_feats) * self.loss_balance
        
        return loss_kd
    
    def train_step(self, data, optimizer, teacher):
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            teacher: model
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        with torch.no_grad():
            teacher = teacher.to(img.device)
            tea_feats = teacher.extract_feat(img)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        # featKD
        losses_kd = self.reviewfeat_distillation(x, tea_feats)
        losses["loss_kd"] = losses_kd
        return losses

@ROTATED_DETECTORS.register_module()
class SPKDRotatedRetinaNet(RotatedSingleStageDetector):
    """
        Implementation of SPKD
        <https://openaccess.thecvf.com/content_ICCV_2019/papers/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.pdf>
    """
    def __init__(self,
                 distillation, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None):
        super(SPKDRotatedRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.loss_balance = distillation["loss_balance"]

    def batch_similarity(self, feats):
        """
            Args:
                feats: shape=[B, C, H, W]
            Returns:
                batch_sim: shape=[B, B]
        """
        feats = feats.view(feats.size(0), -1)  # [B, C*H*W]
        Q = torch.mm(feats, feats.transpose(0, 1))
        norm_Q = Q / torch.norm(Q, p=2, dim=1).unsqueeze(1).expand(Q.shape)
        
        return norm_Q    

    def spatial_similarity(self, feats):
        """
            Args:
                feats: shape=[B, C, H, W]
            Returns:
                spatial_sim: shape=[B, 1, H*W, H*W]
        """
        feats = feats.view(feats.size(0), feats.size(1), -1)  # [B, C, H*W]
        norm_feats = feats / (torch.sqrt(torch.sum(torch.pow(feats, 2), 1)).unsqueeze(1).expand(feats.shape) + 1e-7)
        s = norm_feats.transpose(1, 2).bmm(norm_feats)  # [B, H*W, C] @ [B, C, H*w] = [B, H*W, H*W]
        s = s.unsqueeze(1)
        return s

    def channel_similarity(self, feats):
        """
            Args:
                feats: shape=[B, C, H, W]
            Returns:
                spatial_sim: shape=[B, 1, C, C]
        """
        feats = feats.view(feats.size(0), feats.size(1), -1)  # [B, C, H*W]
        norm_fm = feats / (torch.sqrt(torch.sum(torch.pow(feats, 2), 2)).unsqueeze(2).expand(feats.shape) + 1e-7)
        s = norm_fm.bmm(norm_fm.transpose(1,2))  # [B, C, C]
        s = s.unsqueeze(1)  # [B, 1, C, C]
        return s 
        
    def sp_distillation(self, stu_feat, tea_feat):
        """
            Args:
                stu_feat: type=Tensor, shape=[B, C_S, H, W]
                tea_feat: type=Tensor, shape=[B, C_T, H, W]
            Returns:
                loss
        """
        # tea_feat = self.tea_adapt(tea_feat)
        # batch similarity
        L2 = nn.MSELoss(reduction="mean")
        stu_bs, tea_bs = self.batch_similarity(stu_feat), self.batch_similarity(tea_feat)
        stu_ss, tea_ss = self.spatial_similarity(stu_feat), self.spatial_similarity(tea_feat)
        tea_cs = self.channel_similarity(tea_feat)
        stu_cs = self.channel_similarity(stu_feat)
        loss_bs = self.loss_balance[0] * L2(stu_bs, tea_bs)
        loss_ss = self.loss_balance[1] * L2(stu_ss, tea_ss)
        loss_cs = self.loss_balance[2] * L2(stu_cs, tea_cs)
        
        loss = loss_bs + loss_ss + loss_cs

        return loss,   

    def train_step(self, data, optimizer, teacher):
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            teacher: model
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        with torch.no_grad():
            teacher = teacher.to(img.device)
            tea_feats = teacher.extract_feat(img)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        # SPKD
        losses_kd = multi_apply(
            self.sp_distillation,
            x,
            tea_feats,
        )[0]
        losses["loss_kd"] = losses_kd
        return losses

@ROTATED_DETECTORS.register_module()
class DeFeatRotatedRetinaNet(RotatedSingleStageDetector):
    """
        Implementation of DeFeat
        <https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Distilling_Object_Detectors_via_Decoupled_Features_CVPR_2021_paper.pdf>
    """
    def __init__(self,
                 distillation, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None, 
                 init_cfg=None):
        super(DeFeatRotatedRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.loss_balance = distillation["loss_balance"]

    def map_rois_levels(self, rois, num_levels):
        """
            ROI Map
            Args:
                rois: shape=[num_rois, 4] (x1, y1, x2, y2)
                num_levels: int

        """
        finest_scale = 56
        scale = torch.sqrt((rois[:, 3] - rois[:, 1] + 1) * (rois[:, 2] - rois[:, 0] + 1))
        target_levels = torch.floor(torch.log2(scale / finest_scale + 1e-6))
        target_levels = torch.clamp(target_levels, min=0, max=num_levels - 1).long()
        return target_levels

    def get_gt_mask(self, cls_scores, img_metas, gt_bboxes):
        """
            Args:
                cls_scores: list, len=num_levels| elements: shape=[B, num_anchors*num_classes, Hi, Wi]
                img_metas: list(dict)
                gt_bboxes: list, len=B| elements: shape=[num_gtbox, 5] (x, y, w, h, theta)
        """
        version = self.bbox_head.bbox_coder.angle_range
        featmap_strides = self.bbox_head.anchor_generator.strides  # type=list, len=num_levels| elements: tuple|0: Hi, 1:Wi

        featmap_size_list = [featmap.size()[-2:] for featmap in cls_scores]
        BATCH_SIZE, NUM_LEVELS = cls_scores[0].shape[0], len(cls_scores)
        imit_range = [0] * NUM_LEVELS  # imit_range.shape=[NUM_LEVELS]
        with torch.no_grad():
            gt_mask_batch_list = []
            for batch_idx in range(BATCH_SIZE):
                gt_hbboxes_per = obb2xyxy(gt_bboxes[batch_idx], version)  # gt_hbboxes: shape=[num_gtbboxes, 4] (x1, y1, x2, y2)
                """
                    obb2xyxy: (x, y, w, h, theta) -> (x1, y1, x2, y2)
                """
                target_levels = self.map_rois_levels(gt_hbboxes_per, NUM_LEVELS)
                """
                    map_rois_levels: 找到gtbboxes所对应的特征图
                        target_levels: shape = [num_gtbboxes] value = [0, 1, ..., NUM_LEVELS-1]
                """
                gt_mask_levels_list = []
                # gt_mask_levels_list: list, len=NUM_LEVELS| elements: gt_mask shape=[Hi, Wi]
                for featmap_idx in range(len(featmap_size_list)):
                    gt_hbboxes2featmap = gt_hbboxes_per[target_levels == featmap_idx]
                    H, W = featmap_size_list[featmap_idx]
                    gt_mask_per = cls_scores[0].new_zeros((H, W), requires_grad=False)
                    # gt_mask_per = torch.zeros([H, W], dtype=torch.double).cuda()
                    # 对该特征图对应的gtbbox区域生成mask
                    for gt_idx in range(gt_hbboxes2featmap.shape[0]):
                        gt_region = gt_hbboxes2featmap[gt_idx] / featmap_strides[featmap_idx][0]
                        lx = max(int(gt_region[0]) - imit_range[featmap_idx], 0)
                        rx = min(int(gt_region[2]) - imit_range[featmap_idx], W)
                        ly = max(int(gt_region[1]) - imit_range[featmap_idx], 0)
                        ry = min(int(gt_region[3]) - imit_range[featmap_idx], H)
                        if (lx == rx) or (ly == ry):
                            gt_mask_per[lx, ly] += 1
                        else:
                            gt_mask_per[lx:rx, ly:ry] += 1
                    gt_mask_per = (gt_mask_per > 0).double()
                    gt_mask_levels_list.append(gt_mask_per)
                gt_mask_batch_list.append(gt_mask_levels_list)
            """
                gt_mask_batch_list: list, len=B|
                    elements: list, len=num_levels|
                        [0]: shape=[H0, W0]
                        [1]: shape=[H1, W1]
                        [2]: shape=[H2, W2]
                        [3]: shape=[H3, W3]
                        [4]: shape=[H4, W4]
            """
            gt_mask_levels = []
            for l_idx in range(NUM_LEVELS):
                lst = []
                for b_idx in range(BATCH_SIZE):
                    lst.append(gt_mask_batch_list[b_idx][l_idx])
                gt_mask_levels.append(torch.stack(lst, dim=0))
            """
                gt_mask_levels: list, len=num_levels|
                    [0]: shape=[B, H0, W0]
                    [1]: shape=[B, H1, W1]
                    [2]: shape=[B, H2, W2]
                    [3]: shape=[B, H3, W3]
                    [4]: shape=[B, H4, W4]
            """
        return gt_mask_levels

    def defeat_distillation(self, stu_feat, tea_feat, gt_mask_level):
        """
            Args:
                stu_feat: type=Tensor, shape=[B, C_S, H, W]
                tea_feat: type=Tensor, shape=[B, C_T, H, W]
                gt_mask_level: type=Tensor, shape=[B, H, W]
            Returns:
                loss_kd
        """
        feat_dims = stu_feat.shape[1]
        gt_mask_f = gt_mask_level
        gt_mask_f = gt_mask_f.unsqueeze(1).repeat(1, feat_dims, 1, 1)
        gt_mask_f_norm = max(1.0, gt_mask_f.sum())
        gt_mask_b = 1 - gt_mask_level
        gt_mask_b = gt_mask_b.unsqueeze(1).repeat(1, feat_dims, 1, 1)
        gt_mask_b_norm = max(1.0, gt_mask_b.sum())

        loss_f = torch.sum(torch.abs(stu_feat - tea_feat) * gt_mask_f) / gt_mask_f_norm
        loss_b = torch.sum(torch.abs(stu_feat - tea_feat) * gt_mask_b) / gt_mask_b_norm

        loss_kd = loss_f * self.loss_balance[0] + loss_b * self.loss_balance[1]

        return loss_kd, 

    def train_step(self, data, optimizer, teacher):
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            teacher: model
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        with torch.no_grad():
            teacher = teacher.to(img.device)
            tea_feats = teacher.extract_feat(img)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        # DeFeat KD
        stu_cls_scores = self.bbox_head.forward(x)[0]
        gt_mask_levels = self.get_gt_mask(stu_cls_scores, img_metas, gt_bboxes)
        losses_kd = multi_apply(
            self.defeat_distillation,
            x,
            tea_feats,
            gt_mask_levels
        )[0]
        losses["loss_kd"] = losses_kd
        return losses