# Copyright (c) OpenMMLab. All rights reserved.
# Author: hjc
# Written: 
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmdet.core import images_to_levels, multi_apply

from ..builder import ROTATED_HEADS
from .rotated_anchor_head import RotatedAnchorHead


@ROTATED_HEADS.register_module()
class RotatedRetinaHead(RotatedAnchorHead):
    r"""An anchor-based head used in `RotatedRetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RotatedRetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 5.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def filter_bboxes(self, cls_scores, bbox_preds):
        """Filter predicted bounding boxes at each position of the feature
        maps. Only one bounding boxes with highest score will be left at each
        position. This filter will be used in R3Det prior to the first feature
        refinement stage.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level \
                of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]

            anchors = mlvl_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors,
                                          self.cls_out_channels)

            cls_score, _ = cls_score.max(dim=-1, keepdim=True)
            best_ind = cls_score.argmax(dim=-2, keepdim=True)
            best_ind = best_ind.expand(-1, -1, -1, 5)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 5)
            best_pred = bbox_pred.gather(
                dim=-2, index=best_ind).squeeze(dim=-2)

            anchors = anchors.reshape(-1, self.num_anchors, 5)

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]
                best_pred_i = best_pred[img_id]
                best_anchor_i = anchors.gather(
                    dim=-2, index=best_ind_i).squeeze(dim=-2)
                best_bbox_i = self.bbox_coder.decode(best_anchor_i,
                                                     best_pred_i)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds):
        """This function will be used in S2ANet, whose num_anchors=1.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)

        Returns:
            list[list[Tensor]]: refined rbboxes of each level of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 5)
            anchors = mlvl_anchors[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(anchors, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list

# DID
@ROTATED_HEADS.register_module()
class DIDRotatedRetinaHead(RotatedRetinaHead):
    def __init__(self, 
                 num_classes, 
                 in_channels, 
                 stacked_convs=4, 
                 conv_cfg=None, 
                 norm_cfg=None, 
                 anchor_generator=dict(
                     type='AnchorGenerator', 
                     octave_base_scale=4, 
                     scales_per_octave=3, 
                     ratios=[0.5, 1, 2], 
                     strides=[8, 16, 32, 64, 128]), 
                 init_cfg=dict(
                     type='Normal', 
                     layer='Conv2d', 
                     std=0.01, 
                     override=dict(
                         type='Normal', 
                         name='retina_cls', 
                         std=0.01, 
                         bias_prob=0.01)),
                 loss_balance=[1.0, 1.0],
                 temperature=1.0,
                 factor=1.0, 
                 **kwargs):
        super(DIDRotatedRetinaHead, self).__init__(num_classes, in_channels, stacked_convs, conv_cfg, norm_cfg, anchor_generator, init_cfg, **kwargs)
        self.loss_balance = loss_balance
        self.T = temperature
        self.factor = factor

    def forward_train(self,
                      x,
                      stu_fusion_feats,
                      tea_feats,
                      tea_cls_scores,
                      tea_bbox_preds,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            tea_cls_scores: list, len=num_layers|elements: tensor, shape=[B, num_anchors * num_classes, Hi, Wi]
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (stu_fusion_feats, tea_feats, tea_cls_scores, tea_bbox_preds, gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (stu_fusion_feats, tea_feats, tea_cls_scores, tea_bbox_preds, gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def get_knowledge_decouple_mask(self, tea_cls_score, tea_bbox_pred, anchor, gt_bbox):
        """
            tea_cls_score_list: list, len=num_levels|tensor, shape=[B, num_anchors * num_classes, Hi, Wi]
            tea_bbox_pred_list: list, len=num_levels|tensor, shape=[B, num_anchors * 5, Hi, Wi]
            anchor_list: list, len=num_levels|tensor, shape=[B, num_anchors * Hi * Wi, 5]
            gt_bbox_list: list, len=num_levels|list, len=B|tensor, shape=[num_gt_bboxes, 5]
            Args:
                tea_cls_score: tensor, shape=[B, num_anchors * num_classes, Hi, Wi]
                tea_bbox_pred: tensor, shape=[B, num_anchors * 5, Hi, Wi]
                anchors: tensor, shape=[B, Hi * Wi * num_anchors, 5] (x1, y1, x2, y2, theta)
                gt_bbox: list, len=B|shape=[num_gt_bbox_level, 5] (x1, y1, x2, y2, theta)
            Returns:
                ca_mask: tensor, shape=[B, H, W]
                lo_mask: tensor, shape=[B, H, W]
        """
        B, D, H, W = tea_cls_score.shape
        NUM_ANCHORS = int(D / self.cls_out_channels)
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(B, -1, self.cls_out_channels)  # B, H*W*num_anchor, 15
        tea_bbox_pred = tea_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)  # B*H*W*num_anchors, 5

        # category mask
        per_anchor = torch.max(tea_cls_score.sigmoid(), -1).values
        per_anchor = per_anchor.reshape(B, -1, NUM_ANCHORS)
        per_pixel = torch.max(per_anchor, dim=-1).values
        ca_mask = per_pixel.reshape(B, H, W)
        
        # localization mask
        anchor = anchor.reshape(-1, 5)
        tea_bbox = self.bbox_coder.decode(anchor, tea_bbox_pred)  # B*H*W*num_anchors, 5
        tea_bbox = tea_bbox.reshape(B, -1, 5)  # B, H*W*num_anchors, 5
        lo_mask = []
        for batch_id in range(B):
            iou = self.assigner.iou_calculator(tea_bbox[batch_id, :, :], gt_bbox[batch_id])  # iou: shape=[H*W*num_anchors, num_gt]
            max_iou = torch.max(iou, dim=-1).values
            max_iou_per_anchor = max_iou.reshape(-1, NUM_ANCHORS)
            max_iou_per_pixel = torch.max(max_iou_per_anchor, dim=-1).values
            mask_per_batch = max_iou_per_pixel.reshape(H, W)
            lo_mask.append(mask_per_batch)
        lo_mask = torch.stack(lo_mask, 0)

        return ca_mask, lo_mask

    def ca_logit_distillation(self, stu_cls_scores, tea_cls_scores, ca_mask, label_weight, avg_factor):
        """
            Args:
                stu_cls_scores: tensor, shape=[-1, num_classes]
                tea_cls_scores: tensor, shape=[-1, num_classes]
            Returns:
                Calogitkd loss: tensor
        """
        stu_cls_scores = (ca_mask[:, None, :, :] * stu_cls_scores).permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        tea_cls_scores = (ca_mask[:, None, :, :] * tea_cls_scores).permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        stu_cls_scores = F.log_softmax(stu_cls_scores / self.T, dim=1)
        tea_cls_scores = F.softmax(tea_cls_scores / self.T, dim=1)
        loss = F.kl_div(stu_cls_scores, tea_cls_scores, reduction="none") * (self.T ** 2)
        loss = (label_weight.reshape(-1, 1) * loss).sum() / avg_factor
        
        return loss
    
    def ca_lo_feat_distillation(self, stu_feats, tea_feats, ca_mask, lo_mask):
        """
            Args: 
                stu_feats: tensor, shape=[B, C, Hi, Wi]
                tea_feats: tensor, shape=[B, C, Hi, Wi]
                ca_mask: tensor, shape=[B, Hi, Wi]
                lo_mask: tensor, shape=[B, Hi, Wi]
            Returns:
                Cafeatloss + Lofeatloss          
        """
        # l1 loss
        original_kd = torch.abs(stu_feats - tea_feats)
        # semantic featkd 引入平滑因子
        ca_loss = (original_kd * ca_mask[:, None, :, :]).sum() / (ca_mask.sum() * (self.factor**2))
        # localizalization featkd
        lo_loss = (original_kd * lo_mask[:, None, :, :]).sum() / (lo_mask.sum() * (self.factor**2)) 

        return lo_loss + ca_loss   
        
    def loss_single(self, cls_score, bbox_pred, stu_feats, tea_feats, tea_cls_score, tea_bbox_pred, anchors, gt_bboxes, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            stu_feats: tensor, shape=[B, C, Hi, Wi]
            tea_feats: tensor, shape=[B, C, Hi, Wi]
            tea_cls_score: tensor, shape=[B, num_anchors * num_classes, H, W]
            tea_bbox_preds: tensor, shape=[B, num_anchors* 5, H, W]
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            gt_bboxes: list, len=B|elements: tensor, shape=[num_gt_bboxes, 5]
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
                - loss_logitkd: tensor
                - loss_featkd: tensor
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # Knowledege Decoupled
        with torch.no_grad():
            ca_mask, lo_mask = self.get_knowledge_decouple_mask(tea_cls_score, tea_bbox_pred, anchors, gt_bboxes)
        
        # SeLogitKD of Semantic Distillation 
        loss_logitkd = self.ca_logit_distillation(cls_score, tea_cls_score, ca_mask, label_weights, avg_factor=num_total_samples)
        loss_logitkd = self.loss_balance[0] * loss_logitkd
        
        # classification loss
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        
        # SeFeatKD of Semantic Distillation & LoFeatKD of Localization Distillation
        loss_featkd = self.ca_lo_feat_distillation(stu_feats, tea_feats, ca_mask, lo_mask)
        loss_featkd = self.loss_balance[1] * loss_featkd
        
        return loss_cls, loss_bbox, loss_logitkd, loss_featkd
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             stu_feats,
             tea_feats,
             tea_cls_scores,
             tea_bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            stu_feats: list, len=num_layers|elements: tensor, shape=[B, C, Hi, Wi]
            tea_feats: list, len=num_layers|elements: tensor, shape=[B, C, Hi, Wi]
            tea_cls_scores: list, len=num_layers|elements: tensor, shape=[B, num_anchors * num_classes, Hi, Wi]
            tea_bbox_preds: list, len=num_layers|elements: tensor, shape=[B, num_anchors * 5, Hi, Wi]
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        # copy batch size
        gt_bboxes_list = []
        gt_bboxes_list = [gt_bboxes] * len(stu_feats)

        losses_cls, losses_bbox, losses_logitkd, losses_featkd = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            stu_feats,
            tea_feats,
            tea_cls_scores,
            tea_bbox_preds,
            all_anchor_list,
            gt_bboxes_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_logitkd=losses_logitkd, loss_featkd=losses_featkd)    

# Baseline
@ROTATED_HEADS.register_module()
class LogitKDRotatedRetinaHead(RotatedRetinaHead):
    def __init__(self, 
                 num_classes, 
                 in_channels, 
                 stacked_convs=4, 
                 conv_cfg=None, 
                 norm_cfg=None, 
                 anchor_generator=dict(
                     type='AnchorGenerator', 
                     octave_base_scale=4, 
                     scales_per_octave=3, 
                     ratios=[0.5, 1, 2], 
                     strides=[8, 16, 32, 64, 128]), 
                 init_cfg=dict(
                     type='Normal', 
                     layer='Conv2d', 
                     std=0.01, 
                     override=dict(
                         type='Normal', 
                         name='retina_cls', 
                         std=0.01, 
                         bias_prob=0.01)),
                 loss_balance=1.0,
                 temperature=1.0, 
                 **kwargs):
        super(LogitKDRotatedRetinaHead, self).__init__(num_classes, in_channels, stacked_convs, conv_cfg, norm_cfg, anchor_generator, init_cfg, **kwargs)
        self.loss_balance = loss_balance
        self.T = temperature

    def forward_train(self,
                      x,
                      tea_cls_scores,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            tea_cls_scores: list, len=num_layers|elements: tensor, shape=[B, num_anchors * num_classes, Hi, Wi]
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (tea_cls_scores, gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (tea_cls_scores, gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def logit_distillation(self, stu_scores, tea_scores, label_weight, avg_factor):
        """
            Args:
                stu_scores: tensor, shape=[-1, num_classes]
                tea_scores: tensor, shape=[-1, num_classes]
        """
        stu_scores = F.log_softmax(stu_scores / self.T, dim=1)
        tea_scores = F.softmax(tea_scores / self.T, dim=1)
        loss = F.kl_div(stu_scores, tea_scores, reduction="none") * (self.T ** 2)
        loss = self.loss_balance * (label_weight.reshape(-1, 1) * loss).sum() / avg_factor
        
        return loss
      
    def loss_single(self, cls_score, bbox_pred, tea_cls_score, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            tea_cls_score: tensor, shape=[B, num_anchors * num_classes, H, W]
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
                - loss_kd: tensor
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # logitkd loss
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_kd = self.logit_distillation(cls_score, tea_cls_score, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox, loss_kd
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             tea_cls_scores,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            tea_cls_scores: list, len=num_layers|elements: tensor, shape=[B, num_anchors * num_classes, Hi, Wi]
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_kd = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            tea_cls_scores,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_kd=losses_kd)
