"""SQ quadraphonic passive decoder.

Reads an SQ-encoded stereo FLAC and writes a 4-channel WAV (FL, FR, BL, BR).
"""
import numpy as np
import soundfile as sf
from scipy.signal import hilbert

INPUT_PATH = "input.flac"
OUTPUT_PATH = "decoded_quad.wav"

BLOCK = 1 << 20          # 2^20 samples per block
OVERLAP = 8192           # crossfade length between blocks
SQRT_HALF = 1.0 / np.sqrt(2.0)   # 0.7071067811865475


def sq_decode_block(Lt, Rt):
    """Apply passive SQ decoding to a pair of stereo blocks.

    LF = Lt
    RF = Rt
    LB = 0.707*Rt - 0.707*j*Lt
    RB = 0.707*Lt + 0.707*j*Rt
    where j*x = imag(hilbert(x)) (+90 deg phase shift of x).
    """
    jLt = np.imag(hilbert(Lt))
    jRt = np.imag(hilbert(Rt))
    LF = Lt
    RF = Rt
    LB = SQRT_HALF * Rt - SQRT_HALF * jLt
    RB = SQRT_HALF * Lt + SQRT_HALF * jRt
    return np.stack([LF, RF, LB, RB], axis=1)


def decode_file(stereo):
    """Decode full stereo signal using overlapping blocks with linear crossfade."""
    n = stereo.shape[0]
    out = np.zeros((n, 4), dtype=np.float64)
    step = BLOCK - OVERLAP
    ramp_up = np.linspace(0.0, 1.0, OVERLAP, endpoint=False)[:, None]
    ramp_down = 1.0 - ramp_up

    start = 0
    first = True
    while start < n:
        end = min(start + BLOCK, n)
        Lt = stereo[start:end, 0]
        Rt = stereo[start:end, 1]
        block = sq_decode_block(Lt, Rt)
        blen = block.shape[0]

        if first:
            out[start:end] = block
            first = False
        else:
            ov = min(OVERLAP, blen)
            out[start:start + ov] = (
                out[start:start + ov] * ramp_down[:ov] + block[:ov] * ramp_up[:ov]
            )
            if blen > ov:
                out[start + ov:end] = block[ov:]

        if end == n:
            break
        start += step

    return out


def channel_stats(x, sr):
    names = ["FL", "FR", "BL", "BR"]
    eps = 1e-20
    lines = []
    for i, name in enumerate(names):
        ch = x[:, i]
        peak = np.max(np.abs(ch)) + eps
        rms = np.sqrt(np.mean(ch * ch)) + eps
        lines.append(
            f"  {name}: peak {20 * np.log10(peak):+7.2f} dBFS   "
            f"rms {20 * np.log10(rms):+7.2f} dBFS"
        )
    duration = x.shape[0] / sr
    print(f"Duration:    {duration:.3f} s")
    print(f"Sample rate: {sr} Hz")
    print(f"Samples:     {x.shape[0]}")
    print(f"Channels:    {x.shape[1]} (FL, FR, BL, BR)")
    print("Per-channel levels:")
    for line in lines:
        print(line)


def main():
    print(f"Reading {INPUT_PATH} ...")
    stereo, sr = sf.read(INPUT_PATH, dtype="float64", always_2d=True)
    if stereo.shape[1] != 2:
        raise ValueError(
            f"expected 2-channel stereo input, got {stereo.shape[1]} channels"
        )
    print(f"  {stereo.shape[0]} samples @ {sr} Hz")

    print("Decoding SQ -> quad (blocked Hilbert with crossfade) ...")
    quad = decode_file(stereo)

    peak = float(np.max(np.abs(quad)))
    print(f"Raw peak before normalization: {20 * np.log10(peak + 1e-20):+.2f} dBFS")
    if peak > 1.0:
        quad = (quad / peak) * 0.966
        print("  peak exceeded 0 dBFS; normalized to -0.3 dBFS")
    else:
        print("  peak within 0 dBFS; preserving levels as-is")

    print(f"Writing {OUTPUT_PATH} (24-bit PCM, 4 channels) ...")
    sf.write(OUTPUT_PATH, quad, sr, subtype="PCM_24")

    print()
    channel_stats(quad, sr)


if __name__ == "__main__":
    main()
