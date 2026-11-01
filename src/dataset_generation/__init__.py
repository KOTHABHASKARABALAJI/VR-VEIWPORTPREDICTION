"""
Dataset generation module for 360° viewport trajectory prediction:
- create_sliding_windows: Generates 10-step history and 5-step horizon tensors with boundary protection.
- get_subject_split_mapping: Provides the official subject-independent train/val/test mapping.
"""

from .sequence_generator import (
    create_sliding_windows,
    get_subject_split_mapping,
)

__all__ = [
    "create_sliding_windows",
    "get_subject_split_mapping",
]
