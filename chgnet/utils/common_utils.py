from __future__ import annotations

import json
import os

import torch
from torch import Tensor


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        """Initialize the meter."""
        self.reset()

    def reset(self) -> None:
        """Reset the meter value, average, sum and count to 0."""
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter value, average, sum and count.

        Args:
            val (float): New value to be added to the running average.
            n (int, optional): Number of times the value is added. Default = 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count


def mae(prediction: Tensor, target: Tensor) -> Tensor:
    """Computes the mean absolute error between prediction and target.

    Args:
        prediction: Tensor (N, 1)
        target: Tensor (N, 1).

    Returns:
        tensor
    """
    return torch.mean(torch.abs(target - prediction))


def read_json(fjson: str):
    """Read the json file.

    Args:
        fjson (str): file name of json to read.

    Returns:
        dictionary stored in fjson
    """
    with open(fjson) as file:
        return json.load(file)


def write_json(d: dict, fjson: str):
    """Write the json file.

    Args:
        d (dict): dictionary to write
        fjson (str): file name of json to write.

    Returns:
        written dictionary
    """
    with open(fjson, "w") as file:
        json.dump(d, file)


def mkdir(path: str):
    """Make directory.

    Args:
        path (str): directory name

    Returns:
        path
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")
    return path
