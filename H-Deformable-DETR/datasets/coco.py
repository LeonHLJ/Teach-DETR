# ------------------------------------------------------------------------
# H-DETR
# Copyright (c) 2022 Peking University & Microsoft Research Asia. All Rights Reserved.
# Licensed under the MIT-style license found in the LICENSE file in the root directory
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import json
import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
from util import box_ops
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        cache_mode=False,
        local_rank=0,
        local_size=1,
    ):
        super(CocoDetection, self).__init__(
            img_folder,
            ann_file,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
        )

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ExtCocoDetection(TvCocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        ext_ann_files,
        ext_ann_weights,        
        transforms,
        return_masks,
        cache_mode=False,
        local_rank=0,
        local_size=1,
    ):
        super(ExtCocoDetection, self).__init__(
            img_folder,
            ann_file,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
        )

        self.ext_anns = {}
        for data_name, ext_ann_file in ext_ann_files.items():
            ext_anno = open(ext_ann_file)
            ext_ann = json.load(ext_anno)
            self.ext_anns[data_name] = ext_ann
            ext_anno.close()

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, ext_ann_weights)
     
    def __getitem__(self, idx):
        img, target = super(ExtCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]

        tmp_dict = {k:v[str(image_id)] for k, v in self.ext_anns.items() if str(image_id) in v}

        if tmp_dict:
            target = {"image_id": image_id, "annotations": target, "ext_annotations": tmp_dict}
        else:
            target = {"image_id": image_id, "annotations": target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, ext_ann_weights={}):
        self.return_masks = return_masks
        self.ext_ann_weights = ext_ann_weights

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        # extra annotation
        if "ext_annotations" in target.keys():

            ext_boxes = []
            ext_scores = []
            ext_classes = []
            ext_weighted_scores = []
            ext_box_count = 0
            for k, v in target["ext_annotations"].items():

                tmp_ext_boxes = [obj["bbox"] for obj in v]
                # guard against no boxes via resizing
                tmp_ext_boxes = torch.as_tensor(tmp_ext_boxes, dtype=torch.float32).reshape(-1, 4)
                tmp_ext_boxes[:, 2:] += tmp_ext_boxes[:, :2]
                tmp_ext_boxes[:, 0::2].clamp_(min=0, max=w)
                tmp_ext_boxes[:, 1::2].clamp_(min=0, max=h)

                tmp_ext_classes = [obj["category_id"] for obj in v]
                tmp_ext_classes = torch.tensor(tmp_ext_classes, dtype=torch.int64)

                tmp_ext_scores = [obj["score"] for obj in v]
                tmp_ext_scores = torch.tensor(tmp_ext_scores, dtype=torch.float32)

                # extra annotation process
                ext_keep = (tmp_ext_boxes[:, 3] > tmp_ext_boxes[:, 1]) & (tmp_ext_boxes[:, 2] > tmp_ext_boxes[:, 0])
                tmp_ext_boxes = tmp_ext_boxes[ext_keep]
                tmp_ext_scores = tmp_ext_scores[ext_keep]
                tmp_ext_classes = tmp_ext_classes[ext_keep]

                ext_boxes.append(tmp_ext_boxes)
                ext_scores.append(tmp_ext_scores)
                ext_classes.append(tmp_ext_classes)
                ext_weighted_scores.append(tmp_ext_scores + self.ext_ann_weights[k])
                ext_box_count += len(tmp_ext_boxes)

            avg_boxes = int(ext_box_count // len(ext_boxes))

            ext_boxes = torch.cat(ext_boxes, dim=0)
            ext_scores = torch.cat(ext_scores, dim=0)
            ext_classes = torch.cat(ext_classes, dim=0)
            ext_weighted_scores = torch.cat(ext_weighted_scores, dim=0)
            tmp_keep = box_ops.batched_nms(ext_boxes, ext_weighted_scores, ext_classes, 0.2)

            tmp_keep = tmp_keep[:avg_boxes]
            ext_boxes = ext_boxes[tmp_keep]
            ext_scores = ext_scores[tmp_keep]
            ext_classes = ext_classes[tmp_keep]

            # iou, _ = box_ops.box_iou(boxes, ext_boxes)
            # max_iou_scores, max_idx = iou.max(0)
            # nearest_classes = classes[max_idx]
            # ext_classes = torch.where(max_iou_scores>0.8, nearest_classes, ext_classes)
        else:
            ext_boxes = torch.as_tensor([0, 0, w, h], dtype=torch.float32).reshape(-1, 4)
            ext_classes = torch.tensor([0], dtype=torch.int64)
            ext_scores = torch.tensor([0], dtype=torch.float32)

        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["ext_boxes"] = ext_boxes
        target["ext_labels"] = ext_classes
        target["ext_scores"] = ext_scores

        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose([T.RandomResize([800], max_size=1333), normalize,])

    raise ValueError(f"unknown {image_set}")


def build(image_set, args, eval_in_training_set):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    if eval_in_training_set:
        image_set = "val"
        print("use validation dataset transforms")

    if image_set == 'train':
        dataset = ExtCocoDetection(
            img_folder,
            ann_file,
            args.ext_ann_files,
            args.ext_ann_weights,            
            transforms=make_coco_transforms(image_set),
            return_masks=args.masks,
            cache_mode=args.cache_mode,
            local_rank=get_local_rank(),
            local_size=get_local_size(),
        )
    else:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(image_set),
            return_masks=args.masks,
            cache_mode=args.cache_mode,
            local_rank=get_local_rank(),
            local_size=get_local_size(),
        )
    return dataset
