from multiprocessing import Pool

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from .bbox_overlaps import bbox_overlaps
from .mean_ap import tpfp_default, tpfp_imagenet, get_cls_results

def eval_PR(det_results,
             annotations,
             scale_ranges=None,
             iou_thr=0.5,
             dataset=None,
             logger=None,
             tpfp_fn=None,
             nproc=4):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: List[tuple(P, R, Score), …], descending order
    """

    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = Pool(nproc)
    eval_results = []

    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        if tpfp_fn is None:
            if dataset in ['det', 'vid']:
                tpfp_fn = tpfp_imagenet
            else:
                tpfp_fn = tpfp_default
        if not callable(tpfp_fn):
            raise ValueError(
                f'tpfp_fn has to be a function or None, but got {tpfp_fn}')
    # compute tp and fp for each image with multiple processes
    tpfp = pool.starmap(
        tpfp_fn,
        zip(cls_dets, cls_gts, cls_gts_ignore,
            [iou_thr for _ in range(num_imgs)],
            [area_ranges for _ in range(num_imgs)]))
    tp, fp = tuple(zip(*tpfp))
    # calculate gt number of each scale
    # ignored gts or gts beyond the specific scale are not counted
    num_gts = np.zeros(num_scales, dtype=int)
    for j, bbox in enumerate(cls_gts):
        if area_ranges is None:
            num_gts[0] += bbox.shape[0]
        else:
            gt_areas = (bbox[:, 2] - bbox[:, 0]) * (
                bbox[:, 3] - bbox[:, 1])
            for k, (min_area, max_area) in enumerate(area_ranges):
                num_gts[k] += np.sum((gt_areas >= min_area)
                                        & (gt_areas < max_area))
    # sort all det bboxes by score, also sort tp and fp
    cls_dets = np.vstack(cls_dets)
    num_dets = cls_dets.shape[0]
    sort_inds = np.argsort(-cls_dets[:, -1])
    tp = np.hstack(tp)[:, sort_inds]
    fp = np.hstack(fp)[:, sort_inds]
    # calculate recall and precision with tp and fp
    tp = np.cumsum(tp, axis=1)
    fp = np.cumsum(fp, axis=1)
    eps = np.finfo(np.float32).eps
    recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
    precisions = tp / np.maximum((tp + fp), eps)
    