import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from ..builder import HEADS
from .retina_head import RetinaHead
from mmdet.core import (images_to_levels, multi_apply)

@HEADS.register_module()
class RetinaEMDHead(RetinaHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_K=2,
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
                         bias_prob=0.01),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0]),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                     loss_bbox=dict(type='L1Loss', loss_weight=1.0))):
                    #  **kwargs):
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            stacked_convs,
            conv_cfg,
            norm_cfg,
            anchor_generator,
            init_cfg
            # **kwargs
        )
        self.num_K = num_K

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
            self.num_anchors * self.cls_out_channels * self.num_K,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4 * self.num_K, 3, padding=1)
            
    
    def get_emd_loss(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, num_total_samples):

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_k, self.cls_out_channels)
        
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)

        
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.num_k, 4)

        def _emd_loss(cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, num_total_samples):
            N = cls_score.shape[0]
            C = cls_score.shape[-1]
            BB = bbox_pred.shape[-1]

            pred_delta = cls_score.reshape(-1, C)
            pred_score = bbox_pred.reshape(-1, BB)

            loss_cls = self.loss_cls(
                pred_delta, labels, label_weights, avg_factor=num_total_samples)
            loss_bbox = self.loss_bbox(
                pred_score,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)
            
            emd_loss = loss_cls + loss_bbox
            emd_loss = emd_loss.reshape(-1, 2).sum(axis=1).reshape(-1,1)
            return emd_loss
        
        loss0 = _emd_loss(cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, num_total_samples)
        loss1 = _emd_loss(torch.flip(cls_score, dims=[1]), torch.flip(bbox_pred, dims=[1]), labels, label_weights, bbox_targets, bbox_weights, num_total_samples)

        loss = torch.cat([loss0, loss1], axis=1)
        _, min_indices = loss.min(axis=1)
        loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
        return loss_emd


    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """



        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        
        emd_loss = self.get_emd_loss(cls_score, bbox_pred, anchors, labels, label_weights, 
                                    bbox_targets, bbox_weights, num_total_samples)
        return emd_loss
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
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
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_emd = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_emd=losses_emd)
    