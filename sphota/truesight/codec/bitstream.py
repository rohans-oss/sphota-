"""The .sphota file format (Step 8): header, z / y streams and the protected-region list, packed with ``struct``.

All integers are big-endian.

    offset  size  field
    0       4     magic            b"SPH1"
    4       1     version          1
    5       2     height           original image height (pixels)
    7       2     width            original image width
    9       1     mode             index of the lambda / model (0 = smallest)
    10      4     seed             diffusion seed, so decoding is reproducible
    14      1     guidance         CFG scale x 10 (15 = 1.5)
    15      4     len(Bz)          byte length of the hyper-latent stream
    19      4     len(By)          byte length of the latent stream
    23      ...   Bz, By           entropy-coded streams
    ...     1     n_boxes          number of protected regions (0-255)
    per box 2+2+2+2+4  x, y, w, h (multiples of 16), len(Broi); then Broi (high-quality stream of that crop)

The fixed header is 23 bytes + 1 (n_boxes) + 12 bytes per protected box. Every bpp we report is
8 * file_size / (H * W), i.e. it includes all of this.

encode_image / decode_image are the user-level functions: pad to a multiple of 64, compress, pack - and the reverse.
"""
import os
import struct
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

MAGIC = b"SPH1"
VERSION = 1
HEADER = struct.Struct(">4sBHHBIBII")      # magic, version, h, w, mode, seed, guidance, len(Bz), len(By)
BOX = struct.Struct(">HHHHI")              # x, y, w, h, len
BOX_ALIGN = 16
MAX_U32 = 2 ** 32 - 1


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    data: bytes = b""


@dataclass
class Sphota:
    """Decoded contents of a .sphota file."""
    height: int
    width: int
    mode: int
    seed: int
    guidance: int                          # CFG scale x 10
    z: bytes
    y: bytes
    boxes: list = field(default_factory=list)

    @property
    def guidance_scale(self):
        return self.guidance / 10.0


def header_bytes(n_boxes=0):
    """Overhead of everything except the z / y / box payloads."""
    return HEADER.size + 1 + BOX.size * n_boxes


def pack(height, width, mode, seed, guidance, z, y, boxes=()):
    """Serialise to bytes. Raises ValueError for fields that do not fit the format."""
    for name, v, hi in (("height", height, 65535), ("width", width, 65535), ("mode", mode, 255),
                        ("seed", seed, MAX_U32), ("guidance", guidance, 255)):
        if not 0 <= int(v) <= hi:
            raise ValueError(f"{name}={v} does not fit the header (0..{hi})")
    if len(boxes) > 255:
        raise ValueError(f"at most 255 protected boxes, got {len(boxes)}")
    if len(z) > MAX_U32 or len(y) > MAX_U32:
        raise ValueError("stream longer than 4 GiB")
    parts = [HEADER.pack(MAGIC, VERSION, height, width, mode, seed, guidance, len(z), len(y)), z, y,
             struct.pack(">B", len(boxes))]
    for b in boxes:
        b = b if isinstance(b, Box) else Box(*b)
        if any(v % BOX_ALIGN for v in (b.x, b.y, b.w, b.h)):
            raise ValueError(f"box {(b.x, b.y, b.w, b.h)} is not aligned to {BOX_ALIGN} px")
        if b.w <= 0 or b.h <= 0 or b.x + b.w > width or b.y + b.h > height:
            raise ValueError(f"box {(b.x, b.y, b.w, b.h)} lies outside the {width}x{height} image")
        parts += [BOX.pack(b.x, b.y, b.w, b.h, len(b.data)), b.data]
    return b"".join(parts)


def unpack(data):
    """Parse bytes into a Sphota. Raises ValueError on a bad magic / version, truncation or trailing bytes."""
    data = bytes(data)
    if len(data) < HEADER.size + 1:
        raise ValueError(f"file too short ({len(data)} bytes)")
    magic, version, h, w, mode, seed, guid, lz, ly = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version} (this reader supports {VERSION})")
    pos = HEADER.size

    def take(n):
        nonlocal pos
        if pos + n > len(data):
            raise ValueError("truncated file")
        chunk = data[pos:pos + n]
        pos += n
        return chunk

    z, y = take(lz), take(ly)
    n_boxes = take(1)[0]
    boxes = []
    for _ in range(n_boxes):
        x0, y0, bw, bh, ln = BOX.unpack(take(BOX.size))
        boxes.append(Box(x0, y0, bw, bh, take(ln)))
    if pos != len(data):
        raise ValueError(f"{len(data) - pos} unexpected trailing bytes")
    return Sphota(h, w, mode, seed, guid, z, y, boxes)


def write_file(path, *args, **kw):
    """pack(...) and write to ``path``; returns the file size in bytes."""
    blob = pack(*args, **kw)
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


def read_file(path):
    with open(path, "rb") as f:
        return unpack(f.read())


# ------------------------------------------------------------------------------------------ image level
def latent_shapes(height, width, M, down=64):
    """(y_shape, z_shape) of the latents of an image of this size: it is padded up to a multiple of ``down``."""
    hp, wp = -(-height // down) * down, -(-width // down) * down
    return (1, M, hp // 16, wp // 16), (hp // 64, wp // 64)


def pad_to_multiple(x, m=64):
    """Edge-replicate pad (B, C, H, W) up to multiples of m (same convention as FullImageDataset)."""
    ph, pw = (-x.shape[-2]) % m, (-x.shape[-1]) % m
    return F.pad(x, (0, pw, 0, ph), mode="replicate")


@torch.no_grad()
def encode_image(net, x, mode=0, seed=0, guidance=15, boxes=()):
    """x: (1, 3, H, W) in [0, 1] (any H, W). Returns (file bytes, compress() dict). ``net`` must be in eval mode."""
    h, w = x.shape[-2:]
    d = net.compress(pad_to_multiple(x.to(next(net.parameters()).device), net.DOWN))
    return pack(h, w, mode, seed, guidance, d["z"], d["y"], boxes), d


@torch.no_grad()
def decode_image(net, blob):
    """File bytes (or a path) -> (x_hat (1, 3, H, W) cropped to the original size, decoded dict, parsed Sphota)."""
    s = read_file(blob) if isinstance(blob, (str, os.PathLike)) else unpack(blob)
    y_shape, z_shape = latent_shapes(s.height, s.width, net.M, net.DOWN)
    out = net.decompress({"z": s.z, "y": s.y, "z_shape": z_shape, "y_shape": y_shape})
    return out["x_hat"][..., :s.height, :s.width], out, s
