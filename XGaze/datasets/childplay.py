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

from XGaze.transforms import ColorJitter, Compose, Normalize, RandomHeadBboxJitter, RandomHorizontalFlip, Resize, ToTensor
from XGaze.utils.common import expand_bbox, generate_gaze_heatmap, generate_mask, get_img_size, pair, square_bbox


IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]

# ============================================================================= #
#                                CHILDPLAY DATASET                              #
# ============================================================================= #
class ChildPlayDataset(Dataset):
    """
    Loads the ChildPlay Gaze dataset (Tafasca et al., ICCV 2023) as multi-person, per-frame
    samples: every row is a (clip, frame, person) annotation, so we group all people annotated in
    the same frame together and predict a heatmap/gaze_vec/inout for every one of them (up to
    `max_people`).

    ChildPlay ships official `train`/`val`/`test` annotation folders, used as-is. Only the
    `inside_visible` and `outside_frame` gaze classes correspond to a definite in/out label (cf.
    the dataset README); every other class (`gaze_shift`, `inside_occluded`, `inside_uncertain`,
    `eyes_closed`, etc.) is unknown and mapped to -1, excluded from every loss/metric.
    """

    # 2 of the 95 source YouTube videos are no longer available; their 4 clips have no extracted images.
    EXCLUDED_CLIPS = {
        "js4wxP9HxG0_3496-3539",
        "js4wxP9HxG0_3589-3760",
        "w6oRoyTBmU4_3693-3871",
        "w6oRoyTBmU4_1653-1939",
    }

    def __init__(
        self,
        root,
        split: str = "test",
        transform: Union[Compose, None] = None,
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        max_people: int = 4,
        return_head_mask: bool = False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), f"Expected `split` to be one of [`train`, `val`, `test`] but received `{split}` instead."
        assert max_people > 0, f"Expected `max_people` to be strictly positive. Received {max_people} instead."

        self.root = root
        self.split = split
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.max_people = max_people
        self.return_head_mask = return_head_mask
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=(-0.1, 0.1))

        self.frames = self.load_annotations()
        self.paths = sorted(self.frames.groups.keys())
        self.length = len(self.paths)

    def load_annotations(self):
        clips_meta = pd.read_csv(os.path.join(self.root, "clips.csv"), usecols=["clip", "video_id"])
        video_id_by_clip = dict(zip(clips_meta["clip"], clips_meta["video_id"]))

        ann_files = sorted(glob.glob(os.path.join(self.root, "annotations", self.split, "*.csv")))

        dfs = []
        for ann_file in ann_files:
            clip = os.path.splitext(os.path.basename(ann_file))[0]
            if clip in self.EXCLUDED_CLIPS:
                continue

            video_id = video_id_by_clip[clip]
            # clip name = "{video_id}_{start_frame}-{end_frame}[-downsampled]"; `frame` in the csv
            # is relative to the clip, and image files are named after the absolute frame number.
            start_frame = int(clip[len(video_id) + 1 :].split("-")[0])

            df = pd.read_csv(ann_file)
            df["path"] = df["frame"].apply(
                lambda frame, clip=clip, video_id=video_id, start_frame=start_frame: os.path.join(
                    clip, f"{video_id}_{start_frame + frame - 1}.jpg"
                )
            )
            dfs.append(df)

        annotations = pd.concat(dfs, ignore_index=True)
        annotations["head_xmin"] = annotations["bbox_x"]
        annotations["head_ymin"] = annotations["bbox_y"]
        annotations["head_xmax"] = annotations["bbox_x"] + annotations["bbox_width"]
        annotations["head_ymax"] = annotations["bbox_y"] + annotations["bbox_height"]
        # Only `inside_visible`/`outside_frame` map to a definite in/out label (cf. README); the
        # remaining gaze classes (occluded, uncertain, gaze_shift, eyes_closed, ...) are unknown.
        annotations["inout"] = -1
        annotations.loc[annotations["gaze_class"] == "inside_visible", "inout"] = 1
        annotations.loc[annotations["gaze_class"] == "outside_frame", "inout"] = 0

        return annotations.groupby("path")

    def __getitem__(self, index: int) -> Dict:
        path = self.paths[index]
        people = self.frames.get_group(path)
        if self.split == "train":
            people = people.sample(frac=1.0)  # shuffle person order for augmentation
        people = people.iloc[: self.max_people].reset_index(drop=True)

        image_path = os.path.join(self.root, "images", path)

        # Load image
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        # Load head bboxes (pixel coordinates) for every person in the frame
        head_bboxes = torch.tensor(
            people[["head_xmin", "head_ymin", "head_xmax", "head_ymax"]].values, dtype=torch.float
        )
        head_bboxes = expand_bbox(head_bboxes, img_w, img_h, k=0.1)  # annotated boxes are a bit tight
        if self.split == "train":
            head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h)

        # Square head bboxes (can have negative values)
        head_bboxes = square_bbox(head_bboxes, img_w, img_h)

        # Extract head crops
        heads = [image.crop(head_bbox.int().tolist()) for head_bbox in head_bboxes]  # type: ignore

        # Normalize head bboxes and clip to [0, 1]
        head_bboxes = head_bboxes / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)
        head_bboxes = torch.clamp(head_bboxes, min=0.0, max=1.0)

        inout = torch.tensor(people["inout"].values, dtype=torch.float)
        gaze_pt = torch.tensor(people[["gaze_x", "gaze_y"]].values, dtype=torch.float)
        inside = inout == 1.0
        gaze_pt[inside, 0] = gaze_pt[inside, 0] / img_w
        gaze_pt[inside, 1] = gaze_pt[inside, 1] / img_h
        gaze_pt[~inside] = -1.0

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

        # Generate per-person gaze heatmaps
        heatmaps = [
            generate_gaze_heatmap(gp, sigma=self.heatmap_sigma, size=self.heatmap_size)
            if io == 1.0
            else torch.zeros((self.heatmap_size, self.heatmap_size), dtype=torch.float)
            for gp, io in zip(sample["gaze_pt"], sample["inout"])
        ]
        sample["gaze_heatmap"] = torch.stack(heatmaps)

        # Compute per-person gaze vectors
        new_img_w, new_img_h = get_img_size(sample["image"])
        gaze_vec = sample["gaze_pt"] - sample["head_centers"]
        gaze_vec = gaze_vec * torch.tensor([new_img_w, new_img_h])
        sample["gaze_vec"] = F.normalize(gaze_vec, p=2, dim=-1)

        # Generate head masks
        if self.return_head_mask:
            sample["head_masks"] = generate_mask(sample["head_bboxes"], new_img_w, new_img_h)

        # Pad missing people (padding goes at the end; inout=-1 marks pad slots, excluded from every loss/metric)
        num_people = len(sample["head_bboxes"])
        num_missing = self.max_people - num_people
        if num_missing > 0:
            sample["head_bboxes"] = F.pad(sample["head_bboxes"], (0, 0, 0, num_missing), value=0.0)
            sample["heads"] = F.pad(sample["heads"], (0, 0, 0, 0, 0, 0, 0, num_missing), value=0.0)
            sample["head_centers"] = F.pad(sample["head_centers"], (0, 0, 0, num_missing), value=0.0)
            sample["gaze_pt"] = F.pad(sample["gaze_pt"], (0, 0, 0, num_missing), value=-1.0)
            sample["inout"] = F.pad(sample["inout"], (0, num_missing), value=-1.0)
            sample["gaze_heatmap"] = F.pad(sample["gaze_heatmap"], (0, 0, 0, 0, 0, num_missing), value=0.0)
            sample["gaze_vec"] = F.pad(sample["gaze_vec"], (0, 0, 0, num_missing), value=0.0)
            if self.return_head_mask:
                sample["head_masks"] = F.pad(sample["head_masks"], (0, 0, 0, 0, 0, 0, 0, num_missing), value=0.0)

        return sample

    def __len__(self):
        return self.length


# ============================================================================= #
#                              CHILDPLAY DATAMODULE                             #
# ============================================================================= #
class ChildPlayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: Union[int, dict] = 32,
        image_size: Union[int, tuple] = (224, 224),
        heatmap_sigma: int = 3,
        heatmap_size: Union[int, tuple] = 64,
        max_people: int = 4,
        return_head_mask: bool = False,
        num_workers: int = 4,
    ):
        super().__init__()
        self.root = root
        self.image_size = pair(image_size)
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.max_people = max_people
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
            # "spawn" avoids forking DataLoader workers after CUDA is already initialized in the
            # main process (PyTorch Lightning moves the model to GPU before the first batch is
            # fetched) -- forking post-CUDA-init crashes workers with "CUDA error: initialization error".
            kwargs["multiprocessing_context"] = "spawn"
        return kwargs

    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomHorizontalFlip(p=0.5),
                    ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.0, 1.5), hue=None, p=0.8),
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.train_dataset = ChildPlayDataset(
                self.root,
                "train",
                train_transform,
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                max_people=self.max_people,
                return_head_mask=self.return_head_mask,
            )

            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = ChildPlayDataset(
                self.root,
                "val",
                val_transform,
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                max_people=self.max_people,
                return_head_mask=self.return_head_mask,
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = ChildPlayDataset(
                self.root,
                "val",
                val_transform,
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                max_people=self.max_people,
                return_head_mask=self.return_head_mask,
            )

        elif stage == "test":
            test_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.test_dataset = ChildPlayDataset(
                self.root,
                "test",
                test_transform,
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                max_people=self.max_people,
                return_head_mask=self.return_head_mask,
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            **self._dataloader_kwargs(),
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            **self._dataloader_kwargs(),
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            **self._dataloader_kwargs(),
        )
        return dataloader
