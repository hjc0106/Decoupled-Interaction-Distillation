# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, build_conv_layer

from mmdet.core import multi_apply

from ..builder import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector



@DETECTORS.register_module()
class MVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXFasterRCNN, self).__init__(**kwargs)

    # overwrite
    def extract_feat(self, points, img, img_metas, return_pts_middle=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        if return_pts_middle:
            pts_feats, pts_infos = self.extract_pts_feat(points, img_feats, img_metas, return_middle=return_pts_middle)
            return (img_feats, pts_feats, pts_infos)
        else:
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas, return_middle=return_pts_middle)
            return (img_feats, pts_feats)

    def extract_pts_feat(self, pts, img_feats, img_metas, return_middle=False):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        pts_middle_infos = dict()
        voxels, num_points, coors = self.voxelize(pts)
        pts_middle_infos["num_points"] = num_points
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)
        pts_middle_infos["voxel_features"] = voxel_features
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        pts_middle_infos["middle_encoder_features"] = x
        x = self.pts_backbone(x)
        pts_middle_infos["backbone_features"] = x
        if self.with_pts_neck:
            x = self.pts_neck(x)
            pts_middle_infos["neck_features"] = x
        if return_middle:
            return x, pts_middle_infos
        else:
            return x

    def forward_dummy(self, points):
        pts_feats = self.extract_pts_feat(points, None, None)
        outs = self.pts_bbox_head(pts_feats)

        return outs

@DETECTORS.register_module()
class DynamicMVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self, **kwargs):
        super(DynamicMVXFasterRCNN, self).__init__(**kwargs)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_pts_feat(self, points, img_feats, img_metas):
        """Extract point features."""
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxels, coors, points, img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

@DETECTORS.register_module()
class DIDMVXFasterRCNN(MVXTwoStageDetector):
    def __init__(self,
                 distillation=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        # update bbox_head
        pts_bbox_head["loss_balance"] = distillation["loss_balance"]
        pts_bbox_head["temperature"] = distillation["temperature"]
        pts_bbox_head["factor"] = distillation["factor"]
        # detector init                 
        super(DIDMVXFasterRCNN, self).__init__(**kwargs)
        self.distillation = distillation
        self.fusion_cfg = self.distillation["fusion_cfg"]
        self.fusion_init()
        
    def fusion_init(self):
        conv_cfg = self.fusion_cfg["conv_cfg"]
        norm_cfg = self.fusion_cfg["norm_cfg"]
        act_cfg = self.fusion_cfg["act_cfg"]
        in_channels = self.fusion_cfg["in_channels"]
        feat_channels = self.fusion_cfg["feat_channels"]

        self.CONV1 = ConvModule(
            in_channels,
            feat_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.CONV2 = ConvModule(
            in_channels,
            feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
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
            x's H/W larger than y's H/W
        Returns:
            
        """
        B, D, H, W = x.shape
        x = self.CONV1(x)
        if att_conv:
            shape = x.shape[-2:]  # H, W
            y = F.interpolate(y, shape, mode="nearest")  # y.shape=[B, 256, Hi, Wi]
            atten_weights = self.ATTEN(torch.cat([x, y], dim=1))  # atten_weights.shape[B, 2, Hi, Wi]
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

    def forward_pts_train(self, 
                          stu_pts_feats, 
                          stu_pts_fusion, 
                          tea_pts_feats, 
                          tea_cls_scores, 
                          tea_bbox_preds, 
                          gt_bboxes_3d, 
                          gt_labels_3d, 
                          img_metas, 
                          gt_bboxes_ignore=None):
        """
        Args:
            stu_pts_feats: list, len=3|
                elements: shape=[B, 256, Hi, Wi]
            tea_pts_feats: list, len=3|
                elements: shape=[B, 256, Hi, Wi]
            tea_cls_scores: list, len=3|
                elements: shape=[B, 80, Hi, Wi]
            tea_bbox_preds: list, len=3|
                elements: shape=[B, 72, Hi, Wi]
            gt_bboxes_3d: list, len=B|
                elements: object|
                    .tensor: shape=[num_gts, 9] (xc, yc, zc, w, h, d, ...)
            gt_labels_3d: list, len=B|
                elements: shape=[num_gts]
        """
        outs = self.pts_bbox_head(stu_pts_feats)  
        """
            outs: tuple, len=3|
                [0]: cls_score,list,len=3|
                    elements: shape=[B, 80, Hi, Wi] (80=8(num_anchors)*10(num_classes))
                [1]: bbox_pred,list,len=3|
                    elements: shape=[B, 72, Hi, Wi] (72=8(num_anchors)*9(delta))
                [2]: dir_cls_pred,list,len=3|
                    elements: shape=[B, 16, Hi, Wi] (16=8(num_anchors)*2())
        """

        loss_inputs = outs + (stu_pts_fusion, tea_pts_feats, tea_cls_scores, tea_bbox_preds, gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            tea_pts_feats, tea_pts_infos = teacher.extract_feat(points, img=img, img_metas=img_metas)
            tea_cls_scores, tea_bbox_preds = teacher.pts_bbox_head(tea_pts_feats)[:2]
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        
        # multi-layers feature interaction
        stu_pts_fusion_feats = self.multi_layers_feature_fusion(pts_feats)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, stu_pts_fusion_feats, tea_pts_feats, tea_cls_scores, tea_bbox_preds, 
                                                gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
                    
        return losses

    def train_step(self, data, optimizer, teacher):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
