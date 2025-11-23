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
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


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

    # ---------------------------
    # Annotations (if column 34 present)
    # ---------------------------
    annotations = None
    if data_in.shape[1] > 34:
        events = data_in[34]
        mask = events != 0
        onsets = np.arange(len(events)) / sfreq
        onsets_masked = onsets[mask]
        descriptions = [str(int(e)) for e in events[mask]]
        annotations = mne.Annotations(
            onset=onsets_masked,
            duration=[1.0/sfreq] * len(onsets_masked),
            description=descriptions
        )
        raw.set_annotations(annotations)
        print(f"Annotations loaded: {len(onsets_masked)} events")

    return raw


def compose_and_filter_raw(args, config):
    """Create MNE raw object and apply pre-processing"""
    raw = mne_from_brainflow(args, config)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=args.verbose)

    raw.drop_channels([
        ch for ch in config["channels"]
        if ch.startswith("U") and ch not in ("U26", "U30")
    ])

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



    # Print remaining channels
    print("Remaining channels:", raw.ch_names)

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

def plot_spectrograms(raw, filename, channels=None, f_low=0, f_high=100):
    """
    Plot spectrograms per channel with event markers (from annotations) using raw MNE object.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object (can be unfiltered).
    channels : list or None
        List of channel names to plot. If None, plot all EEG channels.
    f_low : float
        Lower frequency limit to display.
    f_high : float
        Upper frequency limit to display.
    """

    # Select channels
    if channels is None:
        channels = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, exclude='bads')
        channels = [raw.ch_names[i] for i in channels]

    sfreq = raw.info['sfreq']

    # Extract annotation info if available
    ann_onsets = []
    ann_descr = []
    if hasattr(raw, 'annotations') and raw.annotations is not None:
        ann_onsets = raw.annotations.onset
        ann_descr  = raw.annotations.description

    for ch in channels:
        raw_ch = raw.copy().pick([ch])
        sig = raw_ch.get_data()[0]

        # Spectrogram parameters
        nperseg_spec = 4096
        if nperseg_spec > sig.size:
            nperseg_spec = max(256, 2 ** int(np.floor(np.log2(sig.size / 8))))
        noverlap_spec = int(nperseg_spec * 0.75)

        f_spec, t_spec, Sxx = spectrogram(
            sig, fs=sfreq, nperseg=nperseg_spec, noverlap=noverlap_spec,
            scaling="density", mode="psd"
        )

        # Frequency mask
        mask_spec = (f_spec >= f_low) & (f_spec <= f_high)
        f_show = f_spec[mask_spec]
        Sxx_db = 10.0 * np.log10(Sxx[mask_spec, :] + 1e-20)

        # Relative dB normalization
        Sxx_db_rel = Sxx_db - np.median(Sxx_db)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        pcm = ax.pcolormesh(t_spec, f_show, Sxx_db_rel, shading='gouraud', cmap='inferno')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(f_low, f_high)
        ax.set_title(f"Spectrogram: {os.path.basename(filename)} \nChannel: {ch}")
        fig.colorbar(pcm, ax=ax, label="Relative dB")

        # Overlay event markers if any
        for onset, desc in zip(ann_onsets, ann_descr):
            if t_spec.min() <= onset <= t_spec.max():
                ax.axvline(onset, linestyle='--', linewidth=1.0, color='white')
                y_text = f_low + (f_high - f_low) * 0.02
                ax.text(onset, y_text, str(desc), rotation=90, verticalalignment='bottom', fontsize=8, color='white')

        plt.tight_layout()
        plt.show(block=True)



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

    # Compute PSD with MNE (raw in Volts)
    psd_obj = raw.compute_psd(fmin=1, fmax=200, method="welch", verbose=args.verbose)
    print("PSD done.")

    freqs = psd_obj.freqs
    psd_all = psd_obj.get_data()  # V²/Hz by default if raw in V
    data_all = raw.get_data()     # signal in V
    ch_names = raw.ch_names

    # Convert PSD from V²/Hz to µV²/Hz for realistic EEG units
    psd_all *= 1e12  # 1 V² = 1e12 µV²

    avg_bands = [0.0] * 7
    valid_channel_count = 0

    # --- Helper: integrate PSD over frequency band ---
    def band_power(freqs, psd, fmin, fmax):
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not any(mask):
            return 0.0
        return float(np.trapz(psd[mask], freqs[mask]))  # µV²

    for i, ch in enumerate(ch_names):
        sig = data_all[i]

        # --- Time-domain stats ---
        ptp = np.ptp(sig)
        rms = np.sqrt(np.mean(sig**2))
        dc  = np.mean(sig)
        diffs = np.diff(sig)
        flat = np.sum(np.abs(diffs) < 3e-6) / len(diffs)  # flatline heuristic

        k = kurtosis(sig)
        s = skew(sig)

        is_bad = ptp < 10e-6 or flat > 0.95

        psd_ch = psd_all[i]
        entropy = spectral_entropy(psd_ch)
        ln      = line_noise_ratio(freqs, psd_ch)

        if is_bad:
            print(f"{ch:10s} | Channel skipped from band power summary (shorted or flat).")
            delta = theta = alpha = beta = gamma = h_gamma = 0.0
        else:
            delta   = band_power(freqs, psd_ch, 1.0, 4.0)
            theta   = band_power(freqs, psd_ch, 4.0, 8.0)
            alpha   = band_power(freqs, psd_ch, 8.0, 13.0)
            beta = band_power(freqs, psd_ch, 13.0, 20.0)
            h_beta = band_power(freqs, psd_ch, 20.0, 30.0)
            gamma   = band_power(freqs, psd_ch, 30.0, 60.0)
            h_gamma = band_power(freqs, psd_ch, 60.0, 98.0)

            avg_bands = [sum(x) for x in zip(avg_bands, [delta, theta, alpha, beta, h_beta, gamma, h_gamma])]
            valid_channel_count += 1

        muscle_ratio = gamma / (alpha + beta + h_beta + 1e-20) if (alpha + beta + h_beta) > 0 else 0.0

        # --- Classification ---
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

        # --- Print per channel ---
        print(f"{ch:10s} | PTP={ptp*1e6:6.1f} µV | RMS={rms*1e6:6.1f} µV | "
            f"ET={entropy:4.2f} | LNR={ln:5.2f} dB | MR={muscle_ratio:4.2f} | "
            f"DC={dc*1e6:.1f} µV | FT={flat:.2f} | KT={k:.2f} | SK={s:.2f} | {status} | {status2}")

        print(f"δ: {delta:.2f} µV² ({10*np.log10(delta + 1e-20):.1f} dB) | "
            f"θ: {theta:.2f} µV² ({10*np.log10(theta + 1e-20):.1f} dB) | "
            f"α: {alpha:.2f} µV² ({10*np.log10(alpha + 1e-20):.1f} dB) | "
            f"β: {beta:.2f} µV² ({10*np.log10(beta + 1e-20):.1f} dB) | "
            f"h-β: {h_beta:.2f} µV² ({10*np.log10(h_beta + 1e-20):.1f} dB) | "
            f"γ: {gamma:.2f} µV² ({10*np.log10(gamma + 1e-20):.1f} dB) | "
            f"h-γ: {h_gamma:.2f} µV² ({10*np.log10(h_gamma + 1e-20):.1f} dB)")

    # --- Average over valid channels ---
    if valid_channel_count > 0:
        avg_bands = [b/valid_channel_count for b in avg_bands]
        print("\nAverage band powers over valid channels (µV² | dB):")
        print(f"δ: {avg_bands[0]:.2f} µV² ({10*np.log10(avg_bands[0] + 1e-20):.1f} dB) | "
            f"θ: {avg_bands[1]:.2f} µV² ({10*np.log10(avg_bands[1] + 1e-20):.1f} dB) | "
            f"α: {avg_bands[2]:.2f} µV² ({10*np.log10(avg_bands[2] + 1e-20):.1f} dB) | "
            f"β: {avg_bands[3]:.2f} µV² ({10*np.log10(avg_bands[3] + 1e-20):.1f} dB) | "
            f"h-β: {avg_bands[4]:.2f} µV² ({10*np.log10(avg_bands[4] + 1e-20):.1f} dB) | "
            f"γ: {avg_bands[5]:.2f} µV² ({10*np.log10(avg_bands[4] + 1e-20):.1f} dB) | "
            f"h-γ: {avg_bands[6]:.2f} µV² ({10*np.log10(avg_bands[5] + 1e-20):.1f} dB)")
    else:
        print("\nNo valid channels for band power computation.")

    print("=== END DEVICE CHECK ===\n")



 
    plot_spectrograms(raw, filename=args.file, channels=raw.ch_names, f_low=0, f_high=60)



if __name__ == "__main__":
    main()
