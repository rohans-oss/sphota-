"""Drop-in for the two torchac functions the project uses (encode_float_cdf / decode_float_cdf), in pure Python."""
import numpy as np
import torch

from _purerc import Decoder, Encoder, quantize_cdf


def encode_float_cdf(cdf_float, sym, needs_normalization=True, check_input_bounds=False):
    cdf = quantize_cdf(cdf_float.reshape(-1, cdf_float.shape[-1]).numpy())
    s = sym.reshape(-1).numpy().astype(np.int64)
    if check_input_bounds and (s.min() < 0 or s.max() >= cdf.shape[1] - 1):
        raise ValueError("symbol out of the CDF range")
    idx = np.arange(len(s))
    lo, hi = cdf[idx, s], cdf[idx, s + 1]
    enc = Encoder()
    enc.encode_many(lo.tolist(), (hi - lo).tolist())
    return enc.finish()


def decode_float_cdf(cdf_float, byte_stream):
    cdf = quantize_cdf(cdf_float.reshape(-1, cdf_float.shape[-1]).numpy())
    dec = Decoder(byte_stream)
    out = np.empty(len(cdf), dtype=np.int16)
    for i in range(len(cdf)):
        row = cdf[i]
        v = dec.target()
        s = int(np.searchsorted(row, v, side="right")) - 1
        out[i] = s
        dec.consume(int(row[s]), int(row[s + 1] - row[s]))
    return torch.from_numpy(out).reshape(cdf_float.shape[:-1])
