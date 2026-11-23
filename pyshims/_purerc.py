"""Pure-Python range coder (LZMA-style carry handling, 16-bit probabilities). Replaces the compiled coders of torchac and
compressai on machines without a C++ compiler. The byte format is its own (encoder and decoder both come from this file)."""
import numpy as np

PREC = 16
TOP = 1 << 24


class Encoder:
    def __init__(self):
        self.low, self.rng, self.cache, self.cache_size, self.out = 0, 0xFFFFFFFF, 0, 1, bytearray()

    def _shift(self):
        low = self.low
        if low < 0xFF000000 or low >= 0x100000000:
            carry = low >> 32
            self.out.append((self.cache + carry) & 0xFF)
            for _ in range(self.cache_size - 1):
                self.out.append((0xFF + carry) & 0xFF)
            self.cache_size = 0
            self.cache = (low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (low & 0x00FFFFFF) << 8

    def encode(self, start, size):
        r = self.rng >> PREC
        self.low += r * start
        self.rng = r * size
        while self.rng < TOP:
            self.rng <<= 8
            self._shift()

    def encode_many(self, starts, sizes):
        for a, s in zip(starts, sizes):
            self.encode(a, s)

    def finish(self):
        for _ in range(5):
            self._shift()
        return bytes(self.out[1:])          # the first byte is always the initial cache (0)


class Decoder:
    def __init__(self, data):
        self.data = bytes(data) + b"\0" * 8
        self.pos = 4
        self.code = int.from_bytes(self.data[0:4], "big")
        self.rng = 0xFFFFFFFF

    def _norm(self):
        while self.rng < TOP:
            self.code = ((self.code << 8) | self.data[self.pos]) & 0xFFFFFFFF
            self.pos += 1
            self.rng <<= 8

    def target(self):
        self._r = self.rng >> PREC
        return min(self.code // self._r, (1 << PREC) - 1)

    def consume(self, start, size):
        self.code -= self._r * start
        self.rng = self._r * size
        self._norm()


def quantize_cdf(cdf):
    """float CDF (N, Lp) from 0 to 1 -> int CDF (N, Lp), strictly increasing, 0 .. 2^16 (every symbol keeps width >= 1)."""
    cdf = np.asarray(cdf, dtype=np.float64)
    Lp = cdf.shape[-1]
    q = np.rint(cdf * ((1 << PREC) - (Lp - 1))).astype(np.int64) + np.arange(Lp)
    q[..., 0], q[..., -1] = 0, 1 << PREC
    return q
