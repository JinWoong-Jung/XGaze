#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import glob
import os
from typing import Dict, Union

import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from XGaze.transforms import Compose, Normalize, Resize, ToTensor
from XGaze.utils.common import expand_bbox, generate_gaze_heatmap, generate_mask, get_img_size, pair, square_bbox


IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]

# ============================================================================= #
#                          VIDEOATTENTIONTARGET DATASET                         #
# ============================================================================= #
class VideoAttentionTargetDataset(Dataset):
    """
    Loads the VideoAttentionTarget dataset (Chong et al., CVPR 2020). Each row is a single
    (frame, tracked-person) annotation, so unlike GazeFollow, there is exactly one gaze target
    per sample (no multi-annotator ground truth). Only a single person (the tracked one) is used
    per sample, matching the original paper's protocol.
    """

    def __init__(
        self,
        root,
        split: str = "test",
        transform: Union[Compose, None] = None,
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        return_head_mask: bool = False,
    ):
        super().__init__()

        assert split in ("train", "test"), f"Expected `split` to be one of [`train`, `test`] but received `{split}` instead."

        self.root = root
        self.split = split
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.return_head_mask = return_head_mask
        self.annotations = self.load_annotations()

    def load_annotations(self) -> pd.DataFrame:
        # Each *.txt file is a per-person track within a clip: [img_name, head_xmin, head_ymin, head_xmax, head_ymax, gaze_x, gaze_y]
        columns = ["img_name", "head_xmin", "head_ymin", "head_xmax", "head_ymax", "gaze_x", "gaze_y"]
        ann_files = sorted(glob.glob(os.path.join(self.root, "annotations", self.split, "*", "*", "*.txt")))

        dfs = []
        for ann_file in ann_files:
            df = pd.read_csv(ann_file, sep=",", names=columns, index_col=False, encoding="utf-8-sig")
            clip_dir = os.path.dirname(ann_file)
            clip = os.path.basename(clip_dir)
            show = os.path.basename(os.path.dirname(clip_dir))
            df["path"] = df["img_name"].apply(lambda name: os.path.join(show, clip, name))
            dfs.append(df)

        annotations = pd.concat(dfs, ignore_index=True)
        # <-1, -1> gaze target means the target is outside the frame (cf. dataset README)
        annotations["inout"] = (annotations["gaze_x"] != -1).astype(int)
        self.length = len(annotations)
        return annotations

    def __getitem__(self, index: int) -> Dict:
        item = self.annotations.iloc[index]
        path = item["path"]
        image_path = os.path.join(self.root, "images", path)

        # Load image
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        # Load target head bbox (pixel coordinates)
        target_head_bbox = torch.tensor(
            [item["head_xmin"], item["head_ymin"], item["head_xmax"], item["head_ymax"]], dtype=torch.float
        ).unsqueeze(0)
        target_head_bbox = expand_bbox(target_head_bbox, img_w, img_h, k=0.1)  # annotated boxes are a bit tight

        # Square head bbox (can have negative values)
        head_bboxes = square_bbox(target_head_bbox, img_w, img_h)

        # Extract head crop
        heads = [image.crop(head_bboxes[0].int().tolist())]  # type: ignore

        # Normalize head bbox and clip to [0, 1]
        head_bboxes = head_bboxes / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)
        head_bboxes = torch.clamp(head_bboxes, min=0.0, max=1.0)

        inout = torch.tensor(item["inout"], dtype=torch.float)
        if inout == 1.0:
            gaze_pt = torch.tensor([item["gaze_x"] / img_w, item["gaze_y"] / img_h], dtype=torch.float)
        else:
            gaze_pt = torch.tensor([-1.0, -1.0], dtype=torch.float)

        # Build Sample
        sample = {
            "image": image,
            "heads": heads,
            "head_bboxes": head_bboxes,
            "gaze_pt": gaze_pt,
            "inout": inout,
            "id": index,
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": path,
        }

        # Transform
        if self.transform:
            sample = self.transform(sample)

        # Compute head centers
        sample["head_centers"] = torch.hstack(
            [
                (sample["head_bboxes"][:, [0]] + sample["head_bboxes"][:, [2]]) / 2,
                (sample["head_bboxes"][:, [1]] + sample["head_bboxes"][:, [3]]) / 2,
            ]
        )

        # Generate gaze heatmap
        if sample["inout"] == 1.0:
            sample["gaze_heatmap"] = generate_gaze_heatmap(sample["gaze_pt"], sigma=self.heatmap_sigma, size=self.heatmap_size)
        else:
            sample["gaze_heatmap"] = torch.zeros((self.heatmap_size, self.heatmap_size), dtype=torch.float)

        # Compute gaze vector
        new_img_w, new_img_h = get_img_size(sample["image"])
        gaze_vec = sample["gaze_pt"] - sample["head_centers"][-1]
        gaze_vec = gaze_vec * torch.tensor([new_img_w, new_img_h])
        sample["gaze_vec"] = F.normalize(gaze_vec, p=2, dim=-1)

        # Generate head mask
        if self.return_head_mask:
            sample["head_masks"] = generate_mask(sample["head_bboxes"], new_img_w, new_img_h)

        return sample

    def __len__(self):
        return self.length


# ============================================================================= #
#                        VIDEOATTENTIONTARGET DATAMODULE                        #
# ============================================================================= #
class VideoAttentionTargetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: Union[int, dict] = 32,
        image_size: Union[int, tuple] = (224, 224),
        heatmap_sigma: int = 3,
        heatmap_size: Union[int, tuple] = 64,
        return_head_mask: bool = False,
        num_workers: int = 4,
    ):
        super().__init__()
        self.root = root
        self.image_size = pair(image_size)
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.batch_size = {stage: batch_size for stage in ["train", "val", "test"]} if isinstance(batch_size, int) else batch_size
        self.return_head_mask = return_head_mask
        self.num_workers = num_workers

    def _dataloader_kwargs(self):
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 1
        return kwargs

    def setup(self, stage: str):
        if stage == "test":
            test_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.test_dataset = VideoAttentionTargetDataset(
                self.root,
                "test",
                test_transform,
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                return_head_mask=self.return_head_mask,
            )

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            **self._dataloader_kwargs(),
        )
        return dataloader
