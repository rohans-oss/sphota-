import numpy as np
import torch
from PIL import Image

from truesight.data import CropDataset, FullImageDataset
from truesight.data_prepare import _dedup_within, _near_dup_mask, check_no_leakage


def _mk(tmp_path, n=3, size=(300, 200)):
    files = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        Image.fromarray(np.random.randint(0, 255, (size[1], size[0], 3), np.uint8)).save(p)
        files.append(str(p))
    lst = tmp_path / "l.txt"
    lst.write_text("\n".join(files))
    return lst


def test_crop_shape_and_range(tmp_path):
    x = CropDataset(_mk(tmp_path), 256)[0]
    assert x.shape == (3, 256, 256) and 0 <= x.min() and x.max() <= 1


def test_full_image_padded_to_64(tmp_path):
    img, (h, w), _ = FullImageDataset(_mk(tmp_path))[0]
    assert (h, w) == (200, 300) and img.shape[1] % 64 == 0 and img.shape[2] % 64 == 0


def test_hamming_dedup():
    big = 0xFFFFFFFF << 32
    h = np.array([0, 0b11, big, big + 1], dtype=np.uint64)  # pairs at Hamming distance 2 and 1
    assert _dedup_within(h).tolist() == [True, False, True, False]
    assert _near_dup_mask(np.array([1], dtype=np.uint64), h[:1]).all()


def test_leakage_check():
    assert check_no_leakage({"train": ["a"], "test_kodak": ["a"]}) == {"train&test_kodak": 1}
    assert check_no_leakage({"train": ["a"], "val": ["b"]}) == {"train&val": 0}
