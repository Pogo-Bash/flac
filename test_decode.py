"""Self-check for the SQ decoder.

For each of LF/RF/LB/RB independently, generate a sine tone (200/400/
800/1600 Hz), encode it through the forward SQ matrix, decode with the
script's logic, and verify the designated output channel recovers that
frequency as the dominant FFT component.

Driving one source at a time is the correct way to check decode paths:
passive SQ has unity-gain crosstalk between LB and RB (they share an
axis), so a simultaneous 4-tone mix cannot resolve all four output
channels to their own frequencies -- that is a property of the format,
not the decoder.
"""
import numpy as np
from scipy.signal import hilbert

from decode_sq import sq_decode_block, SQRT_HALF

SR = 48000
SECONDS = 4.0
AMP = 0.25

SOURCES = [
    ("LF", 0, 200.0),   # -> FL (output column 0)
    ("RF", 1, 400.0),   # -> FR (output column 1)
    ("LB", 2, 800.0),   # -> BL (output column 2)
    ("RB", 3, 1600.0),  # -> BR (output column 3)
]


def sq_encode(LF, RF, LB, RB):
    """Forward SQ encoder consistent with the passive decoder.

    Lt = LF + 0.707*j*LB + 0.707*RB
    Rt = RF + 0.707*LB - 0.707*j*RB
    """
    jLB = np.imag(hilbert(LB))
    jRB = np.imag(hilbert(RB))
    Lt = LF + SQRT_HALF * jLB + SQRT_HALF * RB
    Rt = RF + SQRT_HALF * LB - SQRT_HALF * jRB
    return Lt, Rt


def dominant_freq(x, sr):
    n = len(x)
    win = np.hanning(n)
    spec = np.fft.rfft(x * win)
    peak_bin = int(np.argmax(np.abs(spec)))
    return peak_bin * sr / n


def main():
    n = int(SR * SECONDS)
    t = np.arange(n) / SR
    zero = np.zeros(n)

    print(f"{'source':<8}{'out ch':<8}{'freq':>10}{'dominant':>14}{'peak':>12}{'result':>10}")
    all_ok = True
    for name, ch, freq in SOURCES:
        tone = AMP * np.sin(2 * np.pi * freq * t)
        sources = {"LF": zero, "RF": zero, "LB": zero, "RB": zero}
        sources[name] = tone

        Lt, Rt = sq_encode(sources["LF"], sources["RF"],
                           sources["LB"], sources["RB"])
        decoded = sq_decode_block(Lt, Rt)

        trim = SR // 10  # drop Hilbert edge transients
        col = decoded[trim:-trim, ch]

        found = dominant_freq(col, SR)
        peak = float(np.max(np.abs(col)))
        ok = abs(found - freq) < 2.0
        all_ok &= ok
        out_name = ["FL", "FR", "BL", "BR"][ch]
        print(
            f"{name:<8}{out_name:<8}{freq:>8.1f} Hz"
            f"{found:>12.1f} Hz{20*np.log10(peak + 1e-20):>10.2f} dB   "
            f"{'PASS' if ok else 'FAIL'}"
        )

    print()
    if all_ok:
        print("OK - every channel recovered its source frequency.")
    else:
        raise SystemExit("FAIL - at least one channel did not recover its source.")


if __name__ == "__main__":
    main()
