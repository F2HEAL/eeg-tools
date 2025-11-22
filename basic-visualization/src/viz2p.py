#!/usr/bin/env python3
"""Read CSV produced by brainflow (with FreeEEG32 device) and create timeseries
and PSD using MNE

"""


import os
import argparse
import yaml
import pandas as pd
import mne
import numpy as np
from scipy.stats import kurtosis, skew

import matplotlib.pyplot as plt


def read_yaml_config(args):
    """Read YAML and return as config object"""
    if args.verbose:
        print(f"* Reading config from {args.config}")

    with open(args.config, encoding="utf8") as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    """Parse cmdline, read config and return as args,config objects"""
    parser = argparse.ArgumentParser(description="My script description.")

    parser.add_argument("-f", "--file", type=str, required=True, help="Input file path")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Configuration file path"
    )
    parser.add_argument(
        "-r",
        "--crop",
        type=int,
        required=False,
        help="Leading/trailing seconds to drop",
    )
    parser.add_argument(
        "-a",
        "--average",
        action="store_true",
        help="Use average signal as reference for all channels",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    config = read_yaml_config(args)

    return args, config


def mne_from_brainflow(args, config):
    """Read Brainflow CSV and return as MNE Raw object"""
    if args.verbose:
        print(f"* Reading Brainflow CSV from {args.file}")

    csv_file = args.file
    data_in = pd.read_csv(csv_file, sep="\t", header=None).values.T

    # Convert from brainflow (V) to MNE (uV)
    data = data_in[1 : len(config["channels"]) + 1] * 1e-6

    ch_types = ["eeg"] * len(config["channels"])  # Assuming all are EEG channels
    sfreq = 512
    info = mne.create_info(
        ch_names=config["channels"],
        sfreq=sfreq,
        ch_types=ch_types,
        verbose=args.verbose,
    )

    raw = mne.io.RawArray(data, info, verbose=args.verbose)

    return raw


def compose_and_filter_raw(args, config):
    """Create MNE raw object and apply pre-processing"""
    raw = mne_from_brainflow(args, config)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=args.verbose)

    raw.plot(
        scalings={"eeg": 50e-5},
        bad_color="red",
        title="Before filtering " + os.path.basename(args.file),
        verbose=args.verbose,
    )

    # Apply bandpass filter as specified in config file
    raw.filter(l_freq=config["fmin"], h_freq=config["fmax"], verbose=args.verbose)

    # Apply notch filter for mains as specified in config file
    raw.notch_filter(freqs=config["freqs_main"], verbose=args.verbose)

    # crop if requested from cmd-line
    if args.crop and args.crop > 0:
        tmin = args.crop
        tmax = raw.times[-1] - args.crop
        raw = raw.crop(tmin=tmin, tmax=tmax)

    raw.drop_channels([
        ch for ch in config["channels"]
        if ch.startswith("U") and ch not in ("U26", "U30")
    ])

    if args.average:
        raw.set_eeg_reference(ref_channels="average", verbose=args.verbose)

    return raw

def spectral_entropy(psd):
    """
    Compute spectral entropy of a 1D PSD array.
    Safe for MNE-generated PSDs.

    Parameters
    ----------
    psd : array-like
        Power spectral density values (1D).

    Returns
    -------
    float
        Spectral entropy value.
    """
    psd = np.asarray(psd)

    # Safety checks
    if psd.size == 0:
        return 0.0
    total_power = np.sum(psd)
    if total_power <= 0 or not np.isfinite(total_power):
        return 0.0  # no signal or invalid PSD

    # Normalize to probability distribution
    p = psd / total_power

    # Compute Shannon entropy
    H = -np.sum(p * np.log2(p + 1e-20))

    return float(H)

def line_noise_ratio(freqs, psd, fline=50.0, fband=1.0):
    """
    Compute line noise ratio (LNR) around a specified line frequency.
    Safe for PSD arrays from MNE or SciPy.

    Parameters
    ----------
    freqs : array-like
        Frequency vector.
    psd : array-like
        Power spectral density (1D for one channel).
    fline : float
        Line frequency (typically 50 or 60 Hz).
    fband : float
        Half-bandwidth around line frequency (Hz).

    Returns
    -------
    float
        Line-noise ratio in dB.
    """
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)

    # Select frequency bins around the line
    idx_line = np.where((freqs >= fline - fband) &
                        (freqs <= fline + fband))[0]

    # Baseline = two side bands, skipping the center
    idx_baseline = np.where(
        ((freqs >= fline - 3*fband) & (freqs < fline - fband)) |
        ((freqs > fline + fband)  & (freqs <= fline + 3*fband))
    )[0]

    # Safety checks
    if len(idx_line) == 0 or len(idx_baseline) == 0:
        return 0.0   # cannot compute

    line_p = np.mean(psd[idx_line])
    base_p = np.mean(psd[idx_baseline])

    if base_p <= 0 or not np.isfinite(base_p):
        return 0.0

    return float(10 * np.log10((line_p + 1e-20) / (base_p + 1e-20)))

# Band power helper (Python-native)
# def band_power(freqs, psd, f1, f2):
#     """Integrate PSD between f1 and f2"""
#     indices = [i for i, f in enumerate(freqs) if f1 <= f <= f2]
#     if not indices:
#         return 0.0
#     return sum(psd[i] for i in indices)

def band_power(freqs, psd, fmin, fmax):
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(idx) == 0:
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))




def main():
    args, config = parse_args()

    # Compose and filter raw EEG
    raw = compose_and_filter_raw(args, config)

    raw.plot(
        scalings={"eeg": 50e-5},
        bad_color="red",
        title=os.path.basename(args.file),
        verbose=args.verbose,
        block=True
    )
    raw.compute_psd(fmin=0, fmax=200, method="welch", verbose=args.verbose).plot()
    plt.show(block=True)

    # ---------------------------
    # DEVICE SIGNAL QUALITY CHECK
    # ---------------------------
    print("\n=== DEVICE SIGNAL QUALITY CHECK ===")
    print("Computing PSD…")
    psd_obj = raw.compute_psd(fmin=1, fmax=200, method="welch", verbose=args.verbose)
    print("PSD done.")

    freqs = psd_obj.freqs
    psd_all = psd_obj.get_data()
    data_all = raw.get_data()
    ch_names = raw.ch_names

    for i, ch in enumerate(ch_names):
        sig = data_all[i]

        ptp = max(sig) - min(sig)
        rms = (sum(v**2 for v in sig)/len(sig))**0.5
        dc  = sum(sig)/len(sig)
        diffs = [abs(sig[j+1]-sig[j]) for j in range(len(sig)-1)]
        flat = sum(1 for d in diffs if d < 3e-6) / len(diffs)

        psd_ch = psd_all[i]
        entropy = spectral_entropy(psd_ch)
        ln      = line_noise_ratio(freqs, psd_ch)

        alpha = band_power(freqs, psd_ch, 8, 13)
        beta  = band_power(freqs, psd_ch, 13, 30)
        gamma = band_power(freqs, psd_ch, 30, 60)
        muscle_ratio = gamma / (alpha + beta + 1e-20)

        status = "GOOD"
        if ptp < 10e-6:
            status = "SHORTED"
        elif flat > 0.95:
            status = "FLATLINE"
        elif ptp > 800e-6:
            status = "NOISY / POSSIBLE SATURATION"
        elif ln > 20:
            status = "HIGH LINE-NOISE"
        elif muscle_ratio > 1.0:
            status = "EMG / MUSCLE CONTAMINATION"
        elif entropy > 5.0:
            status = "RANDOM NOISE"

        status2 = "UNKNOWN"
        if 2.5 < entropy < 4.8:
            status2 = "HUMAN EEG"
        elif entropy > 4.8:
            status2 = "RANDOM NOISE"

        print(f"{ch:10s} | PTP={ptp*1e6:6.1f} µV | RMS={rms*1e6:6.1f} µV | "
              f"ET={entropy:4.2f} | LNR={ln:5.2f} dB | MR={muscle_ratio:4.2f} | "
              f"DC={dc:.1f} | FT={flat:.2f} | {status} | {status2}")

    print("=== END DEVICE CHECK ===\n")




if __name__ == "__main__":
    main()
