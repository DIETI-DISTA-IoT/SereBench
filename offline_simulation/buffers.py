"""Telemetry and weight buffers — faithful ports of the platform's buffers.

* :class:`Buffer` is the consumer's ``preprocessing.Buffer`` (a size-capped
  reservoir of (features, label) pairs, sampled per training batch).
* :class:`GenericBuffer` is the FL manager's ``preprocessing.GenericBuffer``
  (a small FIFO of the most recent per-vehicle ``state_dict``s).

Both are reproduced exactly, including the consumer Buffer's independent
sampling of features and labels — kept as-is so offline behaviour matches the
Dockerised runs rather than silently "fixing" it.
"""

import random
from threading import Lock

import numpy as np
import torch


def dict_to_tensor(data_dict):
    """Port of preprocessing.dict_to_tensor: NaN/non-numeric -> 0.0."""
    values = [
        (value if isinstance(value, (int, float)) and not np.isnan(value) else 0.0)
        for value in data_dict.values()
    ]
    return torch.tensor(values, dtype=torch.float32)


class Buffer:
    """Consumer-side reservoir buffer (preprocessing.Buffer)."""

    def __init__(self, size):
        self.size = size
        self.feats = []
        self.main_labels = []
        self.lock = Lock()

    def add(self, feat_tensor, main_label_tensor):
        with self.lock:
            self.feats.append(feat_tensor)
            self.main_labels.append(main_label_tensor)
            if len(self.feats) > self.size:
                self.feats.pop(0)
                self.main_labels.pop(0)

    def format(self, item):
        """Extract the label (``event_type``) and tensorise the rest, in order."""
        main_label = torch.tensor(item['event_type'], dtype=torch.long)
        del item['event_type']
        feat_tensor = dict_to_tensor(item)
        return feat_tensor, main_label

    def sample(self, n):
        feats = []
        main_labels = []
        with self.lock:
            if len(self.feats) < n:
                feats = self.feats
                main_labels = self.main_labels
            else:
                feats = random.sample(self.feats, n)
                main_labels = random.sample(self.main_labels, n)

        if len(feats) > 0:
            feats = torch.stack(feats)
            main_labels = torch.stack(main_labels).unsqueeze(-1).long()

        return feats, main_labels


class GenericBuffer:
    """FL-side FIFO buffer of the most recent per-vehicle state_dicts."""

    def __init__(self, size, label=None):
        self.size = size
        self.buffer = []
        self.label = label

    def add(self, item):
        self.buffer.insert(0, item)
        if len(self.buffer) > self.size:
            self.buffer.pop()

    def get(self):
        if len(self.buffer) > 0:
            return self.buffer[-1]
        return None

    def pop(self):
        if len(self.buffer) > 0:
            self.buffer.pop()

    def __len__(self):
        return len(self.buffer)
