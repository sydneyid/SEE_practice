import json
import os
import random
import re
import sys
from datetime import timedelta
from os import listdir
from os.path import exists, isdir, isfile, join, splitext
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from absl.logging import debug, error, flags, info, warn
from pudb import set_trace
from scipy import special
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELBC
from see.datasets.basic_batch import get_ev_low_light_batch
from see.utils.event_representation_builder import EventRepresentationBuilder

# NNN-indoor / NNN-outdoor (no suffix variants like 072-indoor-Checkerboard).
NORMALIZED_GROUP_RE = re.compile(r"^\d{3}-(indoor|outdoor)$")


class _EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


def _concat_or_empty(datasets):
    if len(datasets) == 0:
        return _EmptyDataset()
    return ConcatDataset(datasets)

"""
video_name-0:
video_name-1:
    frame_events:
        1721524679594711_0_0_1721524679594711_1721524679595711.png
        1721524679594711_1721524679634711.npy
        1721524679594711_1721524679634711_vis.png
        1721524679634711_0_0_1721524679634711_1721524679635711.png
        1721524679634711_1721524679674711.npy
        1721524679634711_1721524679674711_vis.png
        1721524679674711_0_0_1721524679674711_1721524679675711.png
        1721524679674711_1721524679714711.npy
        1721524679674711_1721524679714711_vis.png
"""


class SeeEverythingEveryTimePairedVideoDataset(Dataset):
    def __init__(
        self,
        group_folder,
        input_video,
        normal_video,
        inputs_frame_events,
        outputs_frame_events,
        in_frames,
        crop_h,
        crop_w,
        ev_rep_cfg,
        is_training,
        input_exposure_states,
        sample_step,
        single_output=True,
    ):
        """
        group_folder: Folder containing the group
        input_video: Input video folder name
        normal_video: Normal video folder name
        inputs_frame_events: List of input frame events. [[frame_list], [event_list]]
        outputs_frame_events: List of outputs frame events. [[frame_list], [event_list]]
        in_frames: Number of frames to use for each sample.
        crop_h: Height of the crop.
        crop_w: Width of the crop.
        ev_rep_cfg: Event representation configuration.
        is_training: Whether the dataset is for training or not.
        input_exposure_states: List of input exposure states. "low-light" or "normal-light" or "high-light"
        sample_step: Number of frames to skip when sampling.
        """
        super().__init__()
        self.group_folder = group_folder
        self.input_video = input_video
        self.normal_video = normal_video
        group_name = group_folder.split("/")[-1]
        self.dataset_video_name = f"{group_name}-{input_exposure_states}-{input_video}-{normal_video}"

        self.in_frames_count = in_frames
        self.inputs_frame = inputs_frame_events[0]
        self.inputs_event = inputs_frame_events[1]
        self.outputs_frame = outputs_frame_events[0]
        self.outputs_event = outputs_frame_events[1]
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.is_training = is_training
        self.input_exposure_states = input_exposure_states
        self.erpcfg = ev_rep_cfg
        self.sample_step = sample_step
        self.single_output = single_output
        # DVS 346 camera height and width
        self.H = 260
        self.W = 346
        # event representation builder
        self.using_event = self.erpcfg.type != "empty"
        self.erpcfg.H = self.H
        self.erpcfg.W = self.W
        self.erbuilder = EventRepresentationBuilder(self.erpcfg)
        #
        self.items = self._generate_items()
        # #
        # info(f"Video Group: {self.group_folder}")
        # info(f"  - input video : {self.input_video}, with {self.input_exposure_states} exposure")
        # info(f"  - normal video: {self.normal_video}")
        # info(f"  - length      : {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        event_files, in_frame_files, ou_frame_files = self.items[index]
        if self.using_event:
            event_stream = []
            for event_file in event_files:
                event = np.load(event_file)
                if event.ndim != 2 or event.shape[1] != 4:
                    warn(f"ERROR: Video: {self.video}, Frame: {index}, Event: {event_file}")
                    continue
                event_stream.append(event)
            event_stream = np.concatenate(event_stream, axis=0)
            events = self.erbuilder(event_stream)
        else:
            events = np.zeros(shape=(self.erpcfg.channel, self.H, self.W))
        # 1.2 load frames
        lfs, lbs, lis = [], [], []
        nfs, nbs, nis = [], [], []
        for lowlgt_frame_path, normal_frame_path in zip(in_frame_files, ou_frame_files):
            lf, lb, li = self._load_frame_and_blur_and_illmap(lowlgt_frame_path)
            nf, nb, ni = self._load_frame_and_blur_and_illmap(normal_frame_path)
            # load in to list
            for x, y in zip([lfs, lbs, lis, nfs, nbs, nis], [lf, lb, li, nf, nb, ni]):
                x.append(y)
        [lfs, lbs, lis, nfs, nbs, nis] = [np.concatenate(x, axis=0) for x in [lfs, lbs, lis, nfs, nbs, nis]]
        # 2. data augmentation
        (
            events,
            lq_frames,
            lq_frame_blurs,
            lq_frame_illmaps,
            normal_frames,
            normal_frame_blurs,
            normal_frame_illmaps,
        ) = self._totensor_crop_flip(events, lfs, lbs, lis, nfs, nbs, nis)
        # 3. construct batch
        batch = get_ev_low_light_batch()
        batch[ELBC.E] = events
        batch[ELBC.LQET] = self.input_exposure_states
        batch[ELBC.LL] = lq_frames
        # More information of inputs.
        batch[ELBC.LLB] = lq_frame_blurs
        batch[ELBC.ILL] = lq_frame_illmaps
        if self.single_output:
            CN, H, W = normal_frames.shape
            N = CN // 3
            normal_frames = normal_frames.reshape(N, 3, H, W)
            batch[ELBC.NL] = normal_frames[N // 2]
        else:
            batch[ELBC.NL] = normal_frames
        batch[ELBC.NLB] = normal_frame_blurs
        batch[ELBC.INL] = normal_frame_illmaps
        # 3.1 add filename and video name
        batch[ELBC.FRAME_NAME] = in_frame_files[len(in_frame_files) // 2].split("/")[-1].split(".")[0]
        batch[ELBC.VIDEO_NAME] = self.dataset_video_name
        return batch

    def _generate_items(self):
        # Align the inputs and outputs
        length = min(len(self.inputs_event), len(self.inputs_frame), len(self.outputs_event), len(self.outputs_frame))
        self.inputs_event = self.inputs_event[:length]
        self.inputs_frame = self.inputs_frame[:length]
        self.outputs_event = self.outputs_event[:length]
        self.outputs_frame = self.outputs_frame[:length]
        #
        items = []
        bias = self.in_frames_count // 2
        for i in range(bias + 1, length - bias - 1, self.sample_step):
            idxs = list(range(i - bias, i + bias + 1))
            # read more events
            in_events = [self.inputs_event[i - bias - 1]] + [self.inputs_event[idx] for idx in idxs]
            in_frames = [self.inputs_frame[idx] for idx in idxs]
            ou_frames = [self.outputs_frame[idx] for idx in idxs]
            # join the goup folder and video to full path.
            in_events = [join(self.group_folder, self.input_video, "frame_event", f) for f in in_events]
            in_frames = [join(self.group_folder, self.input_video, "frame_event", f) for f in in_frames]
            ou_frames = [join(self.group_folder, self.normal_video, "frame_event", f) for f in ou_frames]
            items.append([in_events, in_frames, ou_frames])
        return items

    def _totensor_crop_flip(self, *chw_ndarrays):
        # To Torch Tensor
        chws = [torch.from_numpy(x) for x in chw_ndarrays]
        # Crop
        crop_h, crop_w = self.crop_h, self.crop_w
        if self.is_training:
            top = random.randint(0, self.H - crop_h) // 4 * 4
            left = random.randint(0, self.W - crop_w) // 4 * 4
        else:
            top, left = 0, 0
        chw_ndarrays = [x[..., top : top + crop_h, left : left + crop_w] for x in chws]
        # Flip for horizontal
        if self.is_training and random.random() < 0.5:
            chws = [x.flip(-1) for x in chws]
        # Flip for vertical
        if self.is_training and random.random() < 0.5:
            chws = [x.flip(-2) for x in chws]
        return chw_ndarrays

    def _load_frame_and_blur_and_illmap(self, image_path):
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_blur = cv2.blur(frame, (5, 5))
        frame = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
        frame_blur = frame_blur.astype(np.float32).transpose(2, 0, 1) / 255.0
        # illumiantion map is the max value of RGB channels.
        frame_illmap = np.max(frame, axis=0, keepdims=True)
        return frame, frame_blur, frame_illmap


def _cross_exposure_pairs(inputs, outputs, tag):
    if not inputs or not outputs:
        return []
    return [(inp, out, tag) for inp in inputs for out in outputs]


def _same_exposure_pairs(videos, tag, allow_self_pair=False):
    if not videos:
        return []
    pairs = []
    for i, v_in in enumerate(videos):
        for j, v_out in enumerate(videos):
            if allow_self_pair or i != j:
                pairs.append((v_in, v_out, tag))
    return pairs


def _collect_mapping_pairs(normal_list, low_list, high_list, mapping_type, allow_self_pair=False):
    pairs = []
    for mapping in mapping_type:
        if mapping == "low-normal":
            pairs.extend(_cross_exposure_pairs(low_list, normal_list, "low-normal"))
        elif mapping == "high-normal":
            pairs.extend(_cross_exposure_pairs(high_list, normal_list, "high-normal"))
        elif mapping == "low-high":
            pairs.extend(_cross_exposure_pairs(low_list, high_list, "low-high"))
        elif mapping == "high-low":
            pairs.extend(_cross_exposure_pairs(high_list, low_list, "high-low"))
        elif mapping == "normal-normal":
            pairs.extend(_same_exposure_pairs(normal_list, "normal-normal", allow_self_pair))
        elif mapping == "low-low":
            pairs.extend(_same_exposure_pairs(low_list, "low-low", allow_self_pair))
        elif mapping == "high-high":
            pairs.extend(_same_exposure_pairs(high_list, "high-high", allow_self_pair))
    return pairs


def _fallback_mapping_types(normal_list, low_list, high_list):
    """Mappings that can work with whichever exposure buckets are non-empty."""
    fallbacks = []
    if low_list and normal_list:
        fallbacks.append("low-normal")
    if high_list and normal_list:
        fallbacks.append("high-normal")
    if low_list and high_list:
        fallbacks.extend(["low-high", "high-low"])
    if len(normal_list) >= 1:
        fallbacks.append("normal-normal")
    if len(low_list) >= 1:
        fallbacks.append("low-low")
    if len(high_list) >= 1:
        fallbacks.append("high-high")
    # preserve order, dedupe
    seen = set()
    ordered = []
    for m in fallbacks:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def get_see_everything_everytime_with_event_dataset_for_each_group(
    group_folder, in_frames, crop_h, crop_w, ev_rep_cfg, is_training, mapping_type, sample_step
):
    def _get_dataset_by_in_out_video_name(input_video, normal_video, input_exposure_states):
        if input_video not in video_to_frame_events or normal_video not in video_to_frame_events:
            return None
        sample = SeeEverythingEveryTimePairedVideoDataset(
            group_folder,
            input_video,
            normal_video,
            video_to_frame_events[input_video],
            video_to_frame_events[normal_video],
            in_frames,
            crop_h,
            crop_w,
            ev_rep_cfg,
            is_training,
            input_exposure_states=input_exposure_states,
            sample_step=sample_step,
        )
        if len(sample) == 0:
            return None
        return sample

    def _append_pairs(pairs):
        seen = set()
        for input_video, output_video, tag in pairs:
            key = (input_video, output_video, tag)
            if key in seen:
                continue
            seen.add(key)
            sample = _get_dataset_by_in_out_video_name(input_video, output_video, tag)
            if sample is not None:
                dataset.append(sample)

    with open(join(group_folder, "registrate_result.json"), "r") as f:
        registrate_result = json.load(f)
    video_to_frame_events = _build_video_to_frame_events(group_folder, registrate_result)

    with open(join(group_folder, "exposure_state.json"), "r") as f:
        exposure_state = json.load(f)
    normal_video_list, lowlight_video_list, highlight_video_list = _classify_videos_by_exposure(
        exposure_state, video_to_frame_events
    )

    any_videos = normal_video_list or lowlight_video_list or highlight_video_list
    if not any_videos:
        return [], "no videos with aligned frame_event data"

    dataset = []
    used_fallback = False

    # 1) Requested mappings — skip cross-types when either side is empty; allow self-pairs if only one video in bucket.
    primary_pairs = _collect_mapping_pairs(
        normal_video_list,
        lowlight_video_list,
        highlight_video_list,
        mapping_type,
        allow_self_pair=True,
    )
    _append_pairs(primary_pairs)

    # 2) If nothing matched (e.g. normal=0 or low=0 blocked all yaml mappings), use any feasible mapping.
    if len(dataset) == 0:
        used_fallback = True
        fallback_types = _fallback_mapping_types(
            normal_video_list, lowlight_video_list, highlight_video_list
        )
        fallback_pairs = _collect_mapping_pairs(
            normal_video_list,
            lowlight_video_list,
            highlight_video_list,
            fallback_types,
            allow_self_pair=True,
        )
        _append_pairs(fallback_pairs)

    if len(dataset) == 0:
        reason = (
            f"exposure counts (with data): low={len(lowlight_video_list)}, "
            f"high={len(highlight_video_list)}, normal={len(normal_video_list)}; "
            f"mapping={mapping_type}"
        )
        return [], reason

    if used_fallback:
        info(
            f"Group {group_folder.split('/')[-1]}: used fallback mappings "
            f"(low={len(lowlight_video_list)}, high={len(highlight_video_list)}, "
            f"normal={len(normal_video_list)}) -> {len(dataset)} pair(s)"
        )
    return dataset, None


def _normalize_exposure_label(label: str) -> str:
    s = label.strip().lower().replace("_", "-")
    aliases = {
        "normallight": "normal-light",
        "lowlight": "low-light",
        "highlight": "high-light",
        "highlights": "high-light",
    }
    return aliases.get(s.replace("-", ""), s)


def _build_video_to_frame_events(group_folder, registrate_result):
    """Build aligned frame/event file lists; skip videos with missing or empty frame_event."""
    video_to_frame_events = {}
    for key, value in registrate_result.items():
        start_timestamp = value["start_timestamp"]
        end_timestamp = value["end_timestamp"]
        frame_event_folder = join(group_folder, key, "frame_event")
        if not isdir(frame_event_folder):
            debug(f"Skip video (no frame_event): {join(group_folder, key)}")
            continue
        files = [f for f in listdir(frame_event_folder) if f.endswith(".png")]
        files = sorted(files)
        frame_event_files = [[], []]
        for f in files:
            timestamp = float(f.split("_")[0])
            if start_timestamp <= timestamp <= end_timestamp:
                if "_vis" in f:
                    event_file_name = f.replace("_vis.png", ".npy")
                    npy_path = join(frame_event_folder, event_file_name)
                    if isfile(npy_path):
                        frame_event_files[1].append(event_file_name)
                else:
                    frame_event_files[0].append(f)
        length = min(len(frame_event_files[0]), len(frame_event_files[1]))
        if length == 0:
            debug(f"Skip video (no aligned frames/events): {join(group_folder, key)}")
            continue
        frame_event_files[0] = frame_event_files[0][:length]
        frame_event_files[1] = frame_event_files[1][:length]
        video_to_frame_events[key] = frame_event_files
    return video_to_frame_events


def _classify_videos_by_exposure(exposure_state, video_to_frame_events):
    """Only classify videos that exist in registrate_result with non-empty frame_event."""
    normal_video_list = []
    lowlight_video_list = []
    highlight_video_list = []
    for video_name, values in exposure_state.items():
        if video_name not in video_to_frame_events:
            continue
        video_exposure_state = _normalize_exposure_label(values["exposure_state"])
        if video_exposure_state == "normal-light":
            normal_video_list.append(video_name)
        elif video_exposure_state == "low-light":
            lowlight_video_list.append(video_name)
        elif video_exposure_state == "high-light":
            highlight_video_list.append(video_name)
    return normal_video_list, lowlight_video_list, highlight_video_list


def is_normalized_group(group_name: str) -> bool:
    return NORMALIZED_GROUP_RE.match(group_name) is not None


def _group_name_matches_filter(group_name: str, group_name_filter: str | None) -> bool:
    if group_name_filter is None or group_name_filter in ("all", ""):
        return True
    if group_name_filter == "normalized":
        return is_normalized_group(group_name)
    raise ValueError(f"Unknown group_name_filter: {group_name_filter!r} (use 'all' or 'normalized')")


def group_numeric_id(group_name: str) -> int:
    return int(group_name.split("-", 1)[0])


def is_indoor_group(group_name: str) -> bool:
    """RoboticArm indoor scenes use IDs 000-099 (e.g. 003-indoor, 072-indoor-AAAI)."""
    return group_numeric_id(group_name) < 100


def is_outdoor_group(group_name: str) -> bool:
    return group_numeric_id(group_name) >= 100


def _scenario_matches_filter(group_name: str, scenario_filter: str | None) -> bool:
    if scenario_filter is None or scenario_filter in ("all", ""):
        return True
    if scenario_filter == "indoor":
        return is_indoor_group(group_name)
    if scenario_filter == "outdoor":
        return is_outdoor_group(group_name)
    raise ValueError(f"Unknown scenario_filter: {scenario_filter!r} (use 'all', 'indoor', or 'outdoor')")


def _group_has_required_metadata(group_folder: str) -> bool:
    return isfile(join(group_folder, "registrate_result.json")) and isfile(
        join(group_folder, "exposure_state.json")
    )


def get_see_everything_everytime_with_event_dataset_all(
    root,
    in_frames,
    crop_h,
    crop_w,
    ev_rep_cfg,
    testing_mapping_type,
    training_mapping_type,
    sample_step,
    train_scenario_filter=None,
    val_scenario_filter=None,
    train_group_name_filter=None,
    val_group_name_filter=None,
    test_only=False,
):
    all_train_dataset, all_test_dataset = [], []
    video_all_folder = os.path.abspath(root)
    if not isdir(video_all_folder):
        raise ValueError(
            f"DATASET.root does not exist: {video_all_folder}\n"
            "On Colab, use root: ./SEE-600K/RoboticArm/ and download SEE-600K into the repo."
        )
    if train_scenario_filter or val_scenario_filter:
        info(
            f"Scenario filters: train={train_scenario_filter or 'all'}, val={val_scenario_filter or 'all'}"
        )
    if train_group_name_filter or val_group_name_filter:
        info(
            f"Group name filters: train={train_group_name_filter or 'all'}, "
            f"val={val_group_name_filter or 'all'}"
        )
    if test_only:
        info("TEST_ONLY: skipping training group scan; loading TESTING_GROUPS only.")

    for group in sorted(listdir(video_all_folder)):
        group_folder = join(video_all_folder, group)
        if not isdir(group_folder):
            continue
        if group in TESTING_GROUPS:
            if not _scenario_matches_filter(group, val_scenario_filter):
                continue
            if not _group_name_matches_filter(group, val_group_name_filter):
                continue
            if not _group_has_required_metadata(group_folder):
                warn(f"Skip Group (Testing, incomplete): {group_folder}")
                continue
            dataset_in_one_group, empty_reason = get_see_everything_everytime_with_event_dataset_for_each_group(
                group_folder,
                in_frames,
                crop_h,
                crop_w,
                ev_rep_cfg,
                is_training=False,
                mapping_type=testing_mapping_type,
                sample_step=sample_step,
            )
            if len(dataset_in_one_group) == 0:
                warn(f"Skip Group (Testing, no samples): {group_folder} — {empty_reason}")
                continue
            all_test_dataset.extend(dataset_in_one_group)
        else:
            if test_only:
                continue
            if not _scenario_matches_filter(group, train_scenario_filter):
                continue
            if not _group_name_matches_filter(group, train_group_name_filter):
                continue
            if not _group_has_required_metadata(group_folder):
                warn(f"Skip Group (Training, incomplete): {group_folder}")
                continue
            dataset_in_one_group, empty_reason = get_see_everything_everytime_with_event_dataset_for_each_group(
                group_folder,
                in_frames,
                crop_h,
                crop_w,
                ev_rep_cfg,
                is_training=True,
                mapping_type=training_mapping_type,
                sample_step=sample_step,
            )
            if len(dataset_in_one_group) == 0:
                warn(f"Skip Group (Training, no samples): {group_folder} — {empty_reason}")
                continue
            all_train_dataset.extend(dataset_in_one_group)
    info(f"all_test_dataset: {len(all_test_dataset)}")
    info(f"all_train_dataset: {len(all_train_dataset)}")
    if len(all_test_dataset) == 0:
        has_archives = any(
            isfile(join(video_all_folder, name)) and name.endswith(".tar.gz")
            for name in listdir(video_all_folder)
        )
        if has_archives:
            warn(
                "No test/val samples: test groups may still be .tar.gz only. "
                "Training can continue; use TEST_ONLY after extracting TESTING_GROUPS."
            )
        else:
            warn(
                f"No test/val samples under {video_all_folder}. "
                "Training can continue; inference (TEST_ONLY) needs test group folders."
            )
    if len(all_train_dataset) == 0 and not test_only:
        raise ValueError(
            f"No training samples under {video_all_folder}. "
            "Extract non-test RoboticArm group folders (all groups not in TESTING_GROUPS)."
        )
    if test_only and len(all_test_dataset) == 0:
        raise ValueError(
            f"TEST_ONLY: no validation/test samples under {video_all_folder}. "
            "Extract official TESTING_GROUPS (see see_dataset.py)."
        )
    return _concat_or_empty(all_train_dataset), _concat_or_empty(all_test_dataset)


"""
CONFIG of the dataset
"""

# Official indoor test split (must not be used for training).
INDOOR_TESTING_GROUPS = [
    "000-indoor_ceiling_table_light",
    "001-indoor_wall_displayboard_wood_luggage",
    "002-indoor_trophy_shelf_wall",
    "006-indoor_shot",
    "012-indoor",
    "018-indoor",
    "030-indoor",
    "042-indoor",
    "048-indoor",
    "054-indoor",
    "060-indoor",
    "065-indoor",
    "070-indoor",
    "074-indoor-ResolutionBoard",
    "075-indoor-ICLR",
]

TESTING_GROUPS = [
    *INDOOR_TESTING_GROUPS,
    "001-indoor_wall_displayboard_wood_luggage",
    "002-indoor_trophy_shelf_wall",
    "006-indoor_shot",
    "012-indoor",
    "018-indoor",
    "030-indoor",
    "042-indoor",
    "048-indoor",
    "054-indoor",
    "060-indoor",
    "065-indoor",
    "070-indoor",
    "074-indoor-ResolutionBoard",
    "075-indoor-ICLR",
    "100-outdoor",
    "106-outdoor",
    "112-outdoor",
    "118-outdoor",
    "124-outdoor",
    "130-outdoor",
    "136-outdoor",
    "142-outdoor",
    "148-outdoor",
    "154-outdoor",
    "160-outdoor",
    "166-outdoor",
    "173-outdoor",
    "184-outdoor",
    "189-outdoor",
    "194-outdoor",
    "200-outdoor",
    "206-outdoor",
    "212-outdoor",
    "217-outdoor",
    "222-outdoor",
    "225-outdoor",
]
