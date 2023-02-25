import torch
import os
import json


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count


class MeanNormalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class BaseNormalizer(object):
    """
    Base normalizer to normalize target scalar
    """

    def __init__(self):
        self.mean = 1
        self.std = 1

    def norm(self, tensor):
        return tensor

    def denorm(self, normed_tensor):
        return normed_tensor

    def state_dict(self):
        return {"mean": 1, "std": 1}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    Parameters
    ----------
    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")
    return path


def read_json(fjson):
    """
    Args:
        fjson (str) - file name of json to read

    Returns:
        dictionary stored in fjson
    """
    with open(fjson) as f:
        return json.load(f)


def write_json(d, fjson):
    """
    Args:
        d (dict) - dictionary to write
        fjson (str) - file name of json to write

    Returns:
        written dictionary
    """
    with open(fjson, "w") as f:
        json.dump(d, f)


def solve_charge_by_mag(structure,
                        default_ox = {'Li': 1, 'O': -2},
                        ox_ranges = {'Mn': {(0.5, 1.5): 2, (1.5, 2.5): 3, (2.5, 3.5): 4, (3.5, 4.2): 3, (4.2, 5): 2}} ):
    """
    Args:
        structure: input pymatgen structure
        ox_ranges: user defined range to convert magmoms into formal valence
    """
    ox_list = []
    solved_ox = True

    if "final_magmom" in structure.site_properties.keys():
        mag_key = "final_magmom"
    else:
        mag_key = "magmom"
        
    mag = structure.site_properties[mag_key]

    for site_i, site in enumerate(structure.sites):
        assigned = False
        if site.species_string in ox_ranges.keys():
            for (minmag, maxmag), magox in ox_ranges[site.species_string].items():
                if mag[site_i] >= minmag and mag[site_i] < maxmag:
                    ox_list.append(magox)
                    # print(magox, mag[site_i])
                    assigned = True
                    break
        elif site.species_string in default_ox.keys():
            ox_list.append(default_ox[site.species_string])
            assigned = True
        if not assigned:
            solved_ox = False

    if solved_ox:
        print(ox_list)
        structure.add_oxidation_state_by_site(ox_list)
        return structure

    else:
        return
