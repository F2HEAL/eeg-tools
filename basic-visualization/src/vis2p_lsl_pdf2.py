#!/usr/bin/env python3
"""Read CSV produced by brainflow (with FreeEEG32 device) and create timeseries,
PSDs, spectrograms and a device-check summary, saving everything into a single
multipage PDF report. This variant includes measures to reduce PDF size:
- stronger PDF compression,
- rasterize large pcolormesh objects,
- decimate long time-series and spectrograms before plotting,
- save PDF pages at reduced DPI.
"""

import os
import argparse
import yaml
import pandas as pd
import mne
import numpy as np
from scipy.stats import kurtosis, skew
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import spectrogram
from mne.time_frequency import tfr_morlet
from scipy.signal import welch, find_peaks

# Reduce PDF size by enabling higher compression and using TrueType fonts
mpl.rcParams['pdf.compression'] = 9       # 0..9, higher -> smaller
mpl.rcParams['pdf.fonttype'] = 42         # embed fonts as Truetype
mpl.rcParams['savefig.bbox'] = 'tight'

def read_yaml_config(args):
    """Read YAML and return as config object"""
    if args.verbose:
        print(f"* Reading config from {args.config}")

    with open(args.config, encoding="utf8") as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    """Parse cmdline, read config and return as args,config objects"""
    parser = argparse.ArgumentParser(description="Create EEG PDF report.")

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
    parser.add_argument(
        "-o", "--out", type=str, required=False,
        help="Output PDF path. Default: <input_basename>_report.pdf next to the CSV"
    )

    args = parser.parse_args()

    config = read_yaml_config(args)

    return args, config

def mne_from_brainflow(args, config):
    """Read Brainflow CSV and return as MNE Raw object"""
    if args.verbose:
        print(f"* Reading Brainflow CSV from {args.file}")

    csv_file = args.file
    data_in = pd.read_csv(csv_file, sep=",", header=None).values.T

    # Convert from brainflow (V) to MNE (uV)
    data = data_in[1 : len(config["channels"]) + 1] * 1e-6

    ch_types = ["eeg"] * len(config["channels"])  # Assuming all are EEG channels
    sfreq = 1000
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

    # Ensure we have at least 34 columns (timestamp + 33 EEG + marker)
    if data_in.shape[1] > 33:
        events = data_in[33]

        # Mask out NaNs and zeros (no event)
        mask = (~np.isnan(events)) & (events != 0)

        # Onsets in seconds
        onsets = np.arange(len(events)) / sfreq
        onsets_masked = onsets[mask]

        # Convert only valid event codes to strings
        descriptions = [str(int(e)) for e in events[mask]]

        # Build MNE annotations
        annotations = mne.Annotations(
            onset=onsets_masked,
            duration=[1.0 / sfreq] * len(onsets_masked),
            description=descriptions
        )
        raw.set_annotations(annotations)
        print(f"Annotations loaded: {len(onsets_masked)} events")

    return raw

def decimate_for_plot(times, data, max_points_per_channel=2000):
    """
    Decimate time & data to limit number of plotted points per channel.
    times: 1D array
    data: 2D array shape (n_channels, n_times)
    """
    n_pts = data.shape[1]
    step = max(1, n_pts // max_points_per_channel)
    if step > 1:
        return times[::step], data[:, ::step]
    return times, data

# def save_raw_timeseries(raw, pdf, title=None, max_points_per_channel=2000, dpi=90):
#     """Create a static stacked time-series plot and save it to the PDF (decimated)."""
#     data = raw.get_data() * 1e6  # convert to µV
#     times = raw.times
#     ch_names = raw.ch_names
#     nchan = data.shape[0]

#     # Decimate to limit points per channel
#     times_dec, data_dec = decimate_for_plot(times, data, max_points_per_channel=max_points_per_channel)

#     # Determine vertical spacing
#     ptp = np.ptp(data_dec, axis=1)
#     global_ptp = max(1e-6, np.median(ptp))  # µV
#     offset = global_ptp * 3.0 if global_ptp > 0 else 200.0

#     fig_h = max(4, 0.25 * nchan)  # height scales with number of channels
#     fig, ax = plt.subplots(figsize=(12, fig_h))
#     offsets = np.arange(nchan)[::-1] * offset
#     for i in range(nchan):
#         ax.plot(times_dec, data_dec[i] + offsets[i], linewidth=0.5, color='black', rasterized=True)
#     ax.set_yticks(offsets)
#     ax.set_yticklabels(ch_names[::-1])
#     ax.set_xlabel("Time (s)")
#     ax.set_title(title or "Raw EEG timeseries")
#     ax.grid(False)
#     plt.tight_layout()
#     pdf.savefig(fig, dpi=dpi)
#     plt.close(fig)

def save_raw_timeseries(raw, pdf, title=None, max_points_per_channel=2000, dpi=90):
    """Create a static stacked time-series plot and save it to the PDF (decimated)."""
    data = raw.get_data() * 1e6  # convert to µV
    times = raw.times
    ch_names = raw.ch_names
    nchan = data.shape[0]

    # Decimate to limit points per channel
    times_dec, data_dec = decimate_for_plot(times, data, max_points_per_channel=max_points_per_channel)

    # BETTER OFFSET CALCULATION - use actual data range
    # Calculate robust data range (use 95th percentile to avoid outliers)
    data_ranges = np.percentile(data_dec, 95, axis=1) - np.percentile(data_dec, 5, axis=1)
    robust_range = np.median(data_ranges[data_ranges > 0])  # Avoid zero or negative ranges
    
    if robust_range <= 0:
        robust_range = 100.0  # fallback in µV
    
    # Increase offset for better separation - use 4-5 times the robust range
    offset = robust_range * 4.5

    fig_h = max(6, 0.4 * nchan)  # Increase height scaling
    fig, ax = plt.subplots(figsize=(12, fig_h))
    offsets = np.arange(nchan)[::-1] * offset
    
    # Plot with thinner lines for better visibility
    for i in range(nchan):
        ax.plot(times_dec, data_dec[i] + offsets[i], linewidth=0.3, color='black', rasterized=True)
    
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names[::-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    ax.set_title(title or "Raw EEG timeseries")
    ax.grid(False)
    
    # Add some margin to y-axis
    y_margin = offset * 0.5
    ax.set_ylim(offsets[-1] - y_margin, offsets[0] + y_margin)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)

# import numpy as np


def band_power_with_peaks(
        data,
        fs,
        fmin=1, 
        fmax=200,
        peak_height=1e-6,      # Minimum PSD amplitude to accept a peak
        neighbor_width=2.0     # Hz left/right window for noise estimate
    ):
    """
    Returns ALL PSD peaks above a threshold and computes SNR vs neighbors.

    Parameters
    ----------
    data : 1D numpy array
        EEG signal
    fs : float
        Sampling frequency
    fmin, fmax : float
        Frequency range of interest
    peak_height : float
        Minimum PSD peak amplitude
    neighbor_width : float
        Width (Hz) to left + right of peak used for noise estimation

    Returns
    -------
    results : list of dict
        Each entry:
        {
            "freq": float,
            "peak_power": float,
            "noise_power": float,
            "snr_db": float
        }

    freqs : array
        All frequencies from PSD
    psd : array
        PSD values
    """

    # ---- 1) Welch PSD ----
    freqs, psd = welch(data, fs=fs, nperseg=fs*2, noverlap=fs)

    # keep only fmin–fmax
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    freqs = freqs[idx]
    psd = psd[idx]

    # ---- 2) Find peaks above threshold ----
    peaks, _ = find_peaks(psd, height=peak_height)

    results = []

    # ---- 3) For each peak, compute SNR ----
    for p in peaks:
        f0 = freqs[p]
        peak_power = psd[p]

        # neighbor bands
        left_idx  = np.where((freqs >= f0 - neighbor_width) & (freqs < f0))[0]
        right_idx = np.where((freqs > f0) & (freqs <= f0 + neighbor_width))[0]

        if len(left_idx) == 0 or len(right_idx) == 0:
            continue  # skip if neighbor window incomplete

        noise_floor = np.mean(np.concatenate([psd[left_idx], psd[right_idx]]))

        # SNR in dB
        snr_db = 10 * np.log10(peak_power / noise_floor)

        results.append({
            "freq": float(f0),
            "peak_power": float(peak_power),
            "noise_power": float(noise_floor),
            "snr_db": float(snr_db)
        })

    return results, freqs, psd


#def save_psd_figure(psd_obj, pdf, title=None, dpi=90):
def save_psd_figure(psd_obj_all, raw, pdf, title=None, dpi=150):

    freqs_dec = psd_obj_all.freqs
    psd_dec   = psd_obj_all.get_data()
    ch_names  = psd_obj_all.ch_names

    # --- FIX: sampling frequency reconstruction ---
    fs = int(np.round(freqs_dec[-1] * 2))

    # Plot individual PSD for each channel
    for ch_idx, ch_name in enumerate(ch_names):
        fig, ax = plt.subplots(figsize=(10, 6))

        # PSD plot
        ch_psd = psd_dec[ch_idx]
        ax.plot(freqs_dec, 10 * np.log10(ch_psd + 1e-20),
                color='blue', linewidth=1.5, label=f'Channel {ch_name}')

        # Raw time-domain signal for this channel
        sig_data = raw.get_data(picks=[ch_name])[0]

        # --- Peak & SNR detection ---
        peak_results, _, _ = band_power_with_peaks(
            data=sig_data,
            fs=fs,
            fmin=freqs_dec.min(),
            fmax=freqs_dec.max(),
            peak_height=np.max(ch_psd) * 0.05,
            neighbor_width=2.0
        )

        # --- Annotate each detected peak ---
        for p in peak_results:
            f0  = p["freq"]
            snr = p["snr_db"]

            ax.axvline(f0, color='red', linestyle='--', alpha=0.6)
            ax.text(
                f0,
                ax.get_ylim()[1] * 0.9,
                f"{f0:.1f} Hz\nSNR={snr:.1f} dB",
                rotation=90,
                va='top',
                ha='center',
                fontsize=8,
                color='red'
            )

        # Formatting
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB µV²/Hz)")
        ax.set_title(f"PSD - {ch_name}" + (f" - {title}" if title else ""))
        ax.set_xlim(freqs_dec.min(), freqs_dec.max())
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)


def spectral_entropy(psd):
    psd = np.asarray(psd)
    if psd.size == 0:
        return 0.0
    total_power = np.sum(psd)
    if total_power <= 0 or not np.isfinite(total_power):
        return 0.0
    p = psd / total_power
    H = -np.sum(p * np.log2(p + 1e-20))
    return float(H)

def line_noise_ratio(freqs, psd, fline=50.0, fband=1.0):
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    idx_line = np.where((freqs >= fline - fband) &
                        (freqs <= fline + fband))[0]
    idx_baseline = np.where(
        ((freqs >= fline - 3*fband) & (freqs < fline - fband)) |
        ((freqs > fline + fband)  & (freqs <= fline + 3*fband))
    )[0]
    if len(idx_line) == 0 or len(idx_baseline) == 0:
        return 0.0
    line_p = np.mean(psd[idx_line])
    base_p = np.mean(psd[idx_baseline])
    if base_p <= 0 or not np.isfinite(base_p):
        return 0.0
    return float(10 * np.log10((line_p + 1e-20) / (base_p + 1e-20)))

def plot_spectrograms(raw, filename, pdf, channels=None, f_low=0, f_high=100, dpi=90):
    """Plot and save spectrograms per channel into the PDF (decimated & rasterized)."""
    if channels is None:
        picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, exclude='bads')
        channels = [raw.ch_names[i] for i in picks]

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

        # Downsample spectrogram to limit size
        max_time_bins = 300
        max_freq_bins = 200
        time_step = max(1, Sxx_db_rel.shape[1] // max_time_bins)
        freq_step = max(1, Sxx_db_rel.shape[0] // max_freq_bins)
        Sxx_db_rel_ds = Sxx_db_rel[::freq_step, ::time_step]
        f_show_ds = f_show[::freq_step]
        t_spec_ds = t_spec[::time_step]

        fig, ax = plt.subplots(figsize=(12, 5))
        pcm = ax.pcolormesh(t_spec_ds, f_show_ds, Sxx_db_rel_ds, shading='gouraud', cmap='inferno', rasterized=True)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(f_low, f_high)
        ax.set_title(f"Spectrogram: {os.path.basename(filename)} \nChannel: {ch}")
        fig.colorbar(pcm, ax=ax, label="Relative dB")

        # Overlay event markers if any
        for onset, desc in zip(ann_onsets, ann_descr):
            if t_spec_ds.min() <= onset <= t_spec_ds.max():
                ax.axvline(onset, linestyle='--', linewidth=1.0, color='white')
                y_text = f_low + (f_high - f_low) * 0.02
                ax.text(onset, y_text, str(desc), rotation=90, verticalalignment='bottom', fontsize=8, color='white')

        plt.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

def plot_spectrograms_mne(raw, filename, pdf, fmin=1, fmax=500, picks=None, dpi=90):
    """
    Fast Morlet spectrogram plotting for each EEG channel, decimated & rasterized.
    """
    import os

    # ----------------------------------------------------------
    # Pick channels
    # ----------------------------------------------------------
    if picks is None:
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

    # ----------------------------------------------------------
    # Frequencies (Morlet requires freq > 0!)
    # ----------------------------------------------------------
    fmin = max(fmin, 1)
    freqs = np.arange(fmin, fmax + 1)

    # n_cycles = freq / 2 → stable & reasonably fast
    n_cycles = freqs / 2.0

    # ----------------------------------------------------------
    # Loop over channels to keep memory small
    # ----------------------------------------------------------
    for ch_idx in picks:
        ch_name = raw.ch_names[ch_idx]

        # Compute TFR for THIS channel only; increase decim to reduce size (already used previously)
        tfr = tfr_morlet(
            raw,
            picks=[ch_idx],
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            decim=5,           # already aggressive decimation
            average=False
        )

        # Convert to dB
        data_db = 10 * np.log10(tfr.data[0] + 1e-20)

        # Downsample TFR for plotting
        max_time_bins = 400
        time_step = max(1, data_db.shape[1] // max_time_bins)
        data_db_ds = data_db[:, ::time_step]
        times_ds = tfr.times[::time_step]

        fig = plt.figure(figsize=(12, 5))
        pcm = plt.pcolormesh(times_ds, freqs, data_db_ds, shading="gouraud", cmap="inferno", rasterized=True)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Spectrogram (Morlet): {os.path.basename(filename)}\nChannel: {ch_name}")
        plt.colorbar(label="Power (dB)")

        # ------------------------------------------------------
        # Add annotations like events
        # ------------------------------------------------------
        if raw.annotations is not None:
            for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
                if times_ds.min() <= onset <= times_ds.max():
                    plt.axvline(onset, linestyle="--", color="white", linewidth=1)
                    y_text = freqs.min() + (freqs.max() - freqs.min()) * 0.02
                    plt.text(
                        onset,
                        y_text,
                        str(desc),
                        rotation=90,
                        verticalalignment="bottom",
                        fontsize=8,
                        color="white"
                    )

        plt.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

# def main():
#     args, config = parse_args()

#     # Prepare output PDF path
#     if args.out:
#         out_pdf = args.out
#     else:
#         base = os.path.splitext(os.path.basename(args.file))[0]
#         out_pdf = os.path.join(os.path.dirname(args.file), f"{base}_report.pdf")

#     # Compose and filter raw EEG
#     raw = mne_from_brainflow(args, config)
#     raw.set_montage("standard_1020", on_missing="ignore", verbose=args.verbose)

#     raw.drop_channels([
#         "P7"
#         # ch for ch in config["channels"]
#         # if ch.startswith("U") and ch not in ("U26", "U30")
#     ])

#     # NOTE: removed hard-coded pick_channels(["C4"]) to avoid limiting channels

#     if args.average:
#         raw.set_eeg_reference(ref_channels="average", verbose=args.verbose)
#     else:
#         # Add common reference (e.g., average of mastoids or specific channel)
#         # Option 1: Use specific reference channel(s)
#         # raw.set_eeg_reference(ref_channels=['A1', 'A2'], verbose=args.verbose)  # if you have mastoids
        
#         # Option 2: Common average reference (explicit)
#         raw.set_eeg_reference(ref_channels=None, verbose=args.verbose)  # This sets average reference
        


#     print("Remaining channels:", raw.ch_names)

#     if args.average:
#         raw.set_eeg_reference(ref_channels="average", verbose=args.verbose)

#     report_lines = []
#     # choose a moderate DPI for image pages
#     image_dpi = 90
#     text_dpi = 150

#     with PdfPages(out_pdf) as pdf:
#         # Save a raw timeseries overview (decimated)
#         save_raw_timeseries(raw, pdf, title="Raw EEG (after initial montage & channel drop)",
#                             max_points_per_channel=2000, dpi=image_dpi)

#         # Apply bandpass filter as specified in config file
#         raw.filter(l_freq=config["fmin"], h_freq=config["fmax"], verbose=args.verbose)

#         # Apply notch for mains
#         raw.notch_filter(freqs=config["freqs_main"], verbose=args.verbose)

#         # crop if requested
#         if args.crop and args.crop > 0:
#             tmin = args.crop
#             tmax = raw.times[-1] - args.crop
#             raw = raw.crop(tmin=tmin, tmax=tmax)

#         # Save timeseries after filtering
#         save_raw_timeseries(raw, pdf, title="Raw EEG (after filtering)",
#                             max_points_per_channel=2000, dpi=image_dpi)

#         # Compute and save PSD overview
#         psd_obj_all = raw.compute_psd(fmin=30, fmax=80, method="welch", verbose=args.verbose)
#         save_psd_figure(psd_obj_all, pdf, title=os.path.basename(args.file) + " - PSD (30-80 Hz)", dpi=image_dpi)

#         # Device signal quality check
#         report_lines.append("\n=== DEVICE SIGNAL QUALITY CHECK ===")
#         report_lines.append("Computing PSD…")

#         psd_obj = raw.compute_psd(fmin=30, fmax=80, method="welch", verbose=args.verbose)
#         report_lines.append("PSD done.")

#         freqs = psd_obj.freqs
#         psd_all = psd_obj.get_data()  # V²/Hz
#         data_all = raw.get_data()     # signal in V
#         ch_names = raw.ch_names

#         report_lines.append("Computing spectrograms (Morlet)...")
#         # Save TFR/Morlet spectrograms to PDF (each decimated & rasterized)
#         plot_spectrograms_mne(raw, filename=args.file, pdf=pdf, fmin=30, fmax=80, dpi=image_dpi)
#         report_lines.append("STG done.")

#         # Convert PSD from V²/Hz to µV²/Hz
#         psd_all *= 1e12

#         avg_bands = [0.0] * 7
#         valid_channel_count = 0

def main():
    args, config = parse_args()

    # Prepare output PDF path
    if args.out:
        out_pdf = args.out
    else:
        base = os.path.splitext(os.path.basename(args.file))[0]
        out_pdf = os.path.join(os.path.dirname(args.file), f"{base}_report.pdf")

    # Compose and filter raw EEG
    raw = mne_from_brainflow(args, config)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=args.verbose)

    raw.drop_channels([
        "P7"
    ])

    if args.average:
        raw.set_eeg_reference(ref_channels="average", verbose=args.verbose)
    else:
        raw.set_eeg_reference(ref_channels=None, verbose=args.verbose)

    print("Remaining channels:", raw.ch_names)

    report_lines = []
    image_dpi = 90
    text_dpi = 150

    with PdfPages(out_pdf) as pdf:
        # Save raw timeseries overview
        save_raw_timeseries(raw, pdf, title="Raw EEG (after initial montage & channel drop)",
                            max_points_per_channel=2000, dpi=image_dpi)

        # Apply bandpass filter
        raw.filter(l_freq=config["fmin"], h_freq=config["fmax"], verbose=args.verbose)

        # Apply notch filter for mains
        raw.notch_filter(freqs=config["freqs_main"], verbose=args.verbose)

        # Crop if requested
        if args.crop and args.crop > 0:
            tmin = args.crop
            tmax = raw.times[-1] - args.crop
            raw = raw.crop(tmin=tmin, tmax=tmax)

        # Save filtered timeseries
        save_raw_timeseries(raw, pdf, title="Raw EEG (after filtering)",
                            max_points_per_channel=2000, dpi=image_dpi)

        # Compute PSD
        psd_obj_all = raw.compute_psd(fmin=30, fmax=80, method="welch", verbose=args.verbose)
        save_psd_figure(psd_obj_all, raw, pdf, title=os.path.basename(args.file) + " - PSD (30-80 Hz)", dpi=image_dpi)

        # Device signal quality check
        report_lines.append("\n=== DEVICE SIGNAL QUALITY CHECK ===")
        report_lines.append("Computing PSD…")

        psd_obj = raw.compute_psd(fmin=30, fmax=80, method="welch", verbose=args.verbose)
        report_lines.append("PSD done.")

        freqs = psd_obj.freqs
        psd_all = psd_obj.get_data()  # V²/Hz
        data_all = raw.get_data()     # V
        ch_names = raw.ch_names

        # ---- Generate device report ----
        device_checks, band_powers, avg_band_powers, full_text = make_device_report(
            ch_names, data_all, freqs, psd_all
        )

        # ---- Save Morlet spectrograms ----
        report_lines.append("Computing spectrograms (Morlet)...")
        plot_spectrograms_mne(raw, filename=args.file, pdf=pdf, fmin=30, fmax=80, dpi=image_dpi)
        report_lines.append("STG done.")

        # ---- Render textual report as a PDF page ----
        fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4
        ax.axis('off')
        fig.text(0.01, 0.99, full_text, va='top', ha='left', family='monospace', wrap=True, fontsize=8)
        pdf.savefig(fig, dpi=text_dpi)
        plt.close(fig)

    print(f"Report saved to: {out_pdf}")


def band_power(freqs, psd, fmin, fmax):
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))  # µV²


def make_device_report(ch_names, data_all, freqs, psd_all):
    report_lines = []
    device_checks = []
    band_powers = []
    avg_band_powers = []

    avg_bands = [0, 0, 0, 0, 0, 0, 0]
    valid_channel_count = 0

    report_lines.append("=== DEVICE CHECKS AND BAND POWERS ===\n")

    for i, ch in enumerate(ch_names):
        sig = data_all[i]

        # ---- Time-domain metrics ----
        ptp = np.ptp(sig)
        rms = np.sqrt(np.mean(sig**2))
        dc  = np.mean(sig)
        diffs = np.diff(sig)
        flat = np.sum(np.abs(diffs) < 3e-6) / len(diffs)

        k = kurtosis(sig)
        s = skew(sig)
        is_bad = ptp < 10e-6 or flat > 0.95

        psd_ch = psd_all[i]
        entropy = spectral_entropy(psd_ch)
        ln      = line_noise_ratio(freqs, psd_ch)

        # ---- Band powers ----
        if is_bad:
            delta = theta = alpha = beta = h_beta = gamma = h_gamma = 0.0
        else:
            delta   = band_power(freqs, psd_ch, 1.0, 4.0)
            theta   = band_power(freqs, psd_ch, 4.0, 8.0)
            alpha   = band_power(freqs, psd_ch, 8.0, 13.0)
            beta    = band_power(freqs, psd_ch, 13.0, 20.0)
            h_beta  = band_power(freqs, psd_ch, 20.0, 30.0)
            gamma   = band_power(freqs, psd_ch, 30.0, 60.0)
            h_gamma = band_power(freqs, psd_ch, 60.0, 98.0)

            avg_bands = [sum(x) for x in zip(avg_bands,
                        [delta, theta, alpha, beta, h_beta, gamma, h_gamma])]
            valid_channel_count += 1

        muscle_ratio = (
            gamma / (alpha + beta + h_beta + 1e-20)
            if (alpha + beta + h_beta) > 0
            else 0.0
        )

        # ---- Status flags ----
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

        # ---- Formatted Lines ----
        line1 = (
            f"{ch:10s} | PTP={ptp*1e6:6.1f} µV | RMS={rms*1e6:6.1f} µV | "
            f"ET={entropy:4.2f} | LNR={ln:5.2f} dB | MR={muscle_ratio:4.2f} | "
            f"DC={dc*1e6:.1f} µV | FT={flat:.2f} | KT={k:.2f} | SK={s:.2f} | "
            f"{status} | {status2}"
        )

        line2 = (
            f"δ: {delta:.2f} µV² ({10*np.log10(delta + 1e-20):.1f} dB) | "
            f"θ: {theta:.2f} µV² ({10*np.log10(theta + 1e-20):.1f} dB) | "
            f"α: {alpha:.2f} µV² ({10*np.log10(alpha + 1e-20):.1f} dB) | "
            f"β: {beta:.2f} µV² ({10*np.log10(beta + 1e-20):.1f} dB) | "
            f"h-β: {h_beta:.2f} µV² ({10*np.log10(h_beta + 1e-20):.1f} dB) | "
            f"γ: {gamma:.2f} µV² ({10*np.log10(gamma + 1e-20):.1f} dB) | "
            f"h-γ: {h_gamma:.2f} µV² ({10*np.log10(h_gamma + 1e-20):.1f} dB)"
        )

        # Store in separate lists
        device_checks.append(line1)
        band_powers.append(line2)

    # ---- Averages ----
    if valid_channel_count > 0:
        avg_bands = [b / valid_channel_count for b in avg_bands]

        avg_band_powers = [
            f"δ:   {avg_bands[0]:.2f} µV² ({10*np.log10(avg_bands[0] + 1e-20):.1f} dB)",
            f"θ:   {avg_bands[1]:.2f} µV² ({10*np.log10(avg_bands[1] + 1e-20):.1f} dB)",
            f"α:   {avg_bands[2]:.2f} µV² ({10*np.log10(avg_bands[2] + 1e-20):.1f} dB)",
            f"β:   {avg_bands[3]:.2f} µV² ({10*np.log10(avg_bands[3] + 1e-20):.1f} dB)",
            f"h-β: {avg_bands[4]:.2f} µV² ({10*np.log10(avg_bands[4] + 1e-20):.1f} dB)",
            f"γ:   {avg_bands[5]:.2f} µV² ({10*np.log10(avg_bands[5] + 1e-20):.1f} dB)",
            f"h-γ: {avg_bands[6]:.2f} µV² ({10*np.log10(avg_bands[6] + 1e-20):.1f} dB)",
        ]
    else:
        avg_band_powers = ["No valid channels for band-power computation."]

    # ---- Final joined text for PDF ----
    report_lines.append("=== DEVICE CHECKS ===")
    report_lines.extend(device_checks)

    report_lines.append("\n=== BAND POWERS ===")
    report_lines.extend(band_powers)

    report_lines.append("\n=== AVERAGE BAND POWERS ===")
    report_lines.extend(avg_band_powers)

    report_lines.append("\n=== END DEVICE CHECK ===")
    full_report_text = "\n".join(report_lines)

    return (
        device_checks,
        band_powers,
        avg_band_powers,
        full_report_text
    )


    #     def band_power(freqs, psd, fmin, fmax):
    #         mask = (freqs >= fmin) & (freqs <= fmax)
    #         if not any(mask):
    #             return 0.0
    #         return float(np.trapz(psd[mask], freqs[mask]))  # µV²

    #     report_lines.append("\nPer-channel device checks and band powers:")
    #     for i, ch in enumerate(ch_names):
    #         sig = data_all[i]

    #         # Time-domain stats
    #         ptp = np.ptp(sig)
    #         rms = np.sqrt(np.mean(sig**2))
    #         dc  = np.mean(sig)
    #         diffs = np.diff(sig)
    #         flat = np.sum(np.abs(diffs) < 3e-6) / len(diffs)

    #         k = kurtosis(sig)
    #         s = skew(sig)

    #         is_bad = ptp < 10e-6 or flat > 0.95

    #         psd_ch = psd_all[i]
    #         entropy = spectral_entropy(psd_ch)
    #         ln      = line_noise_ratio(freqs, psd_ch)

    #         if is_bad:
    #             delta = theta = alpha = beta = h_beta = gamma = h_gamma = 0.0
    #         else:
    #             delta   = band_power(freqs, psd_ch, 1.0, 4.0)
    #             theta   = band_power(freqs, psd_ch, 4.0, 8.0)
    #             alpha   = band_power(freqs, psd_ch, 8.0, 13.0)
    #             beta = band_power(freqs, psd_ch, 13.0, 20.0)
    #             h_beta = band_power(freqs, psd_ch, 20.0, 30.0)
    #             gamma   = band_power(freqs, psd_ch, 30.0, 60.0)
    #             h_gamma = band_power(freqs, psd_ch, 60.0, 98.0)

    #             avg_bands = [sum(x) for x in zip(avg_bands, [delta, theta, alpha, beta, h_beta, gamma, h_gamma])]
    #             valid_channel_count += 1

    #         muscle_ratio = gamma / (alpha + beta + h_beta + 1e-20) if (alpha + beta + h_beta) > 0 else 0.0

    #         status = "GOOD"
    #         if ptp < 10e-6:
    #             status = "SHORTED"
    #         elif flat > 0.95:
    #             status = "FLATLINE"
    #         elif ptp > 800e-6:
    #             status = "NOISY / POSSIBLE SATURATION"
    #         elif ln > 20:
    #             status = "HIGH LINE-NOISE"
    #         elif muscle_ratio > 1.0:
    #             status = "EMG / MUSCLE CONTAMINATION"
    #         elif entropy > 5.0:
    #             status = "RANDOM NOISE"

    #         status2 = "UNKNOWN"
    #         if 2.5 < entropy < 4.8:
    #             status2 = "HUMAN EEG"
    #         elif entropy > 4.8:
    #             status2 = "RANDOM NOISE"

    #         line1 = (f"{ch:10s} | PTP={ptp*1e6:6.1f} µV | RMS={rms*1e6:6.1f} µV | "
    #             f"ET={entropy:4.2f} | LNR={ln:5.2f} dB | MR={muscle_ratio:4.2f} | "
    #             f"DC={dc*1e6:.1f} µV | FT={flat:.2f} | KT={k:.2f} | SK={s:.2f} | {status} | {status2}")

    #         line2 = (f"δ: {delta:.2f} µV² ({10*np.log10(delta + 1e-20):.1f} dB) | "
    #             f"θ: {theta:.2f} µV² ({10*np.log10(theta + 1e-20):.1f} dB) | "
    #             f"α: {alpha:.2f} µV² ({10*np.log10(alpha + 1e-20):.1f} dB) | "
    #             f"β: {beta:.2f} µV² ({10*np.log10(beta + 1e-20):.1f} dB) | "
    #             f"h-β: {h_beta:.2f} µV² ({10*np.log10(h_beta + 1e-20):.1f} dB) | "
    #             f"γ: {gamma:.2f} µV² ({10*np.log10(gamma + 1e-20):.1f} dB) | "
    #             f"h-γ: {h_gamma:.2f} µV² ({10*np.log10(h_gamma + 1e-20):.1f} dB)")

    #         print(line1)
    #         print(line2)
    #         report_lines.append(line1)
    #         report_lines.append(line2)

    #     if valid_channel_count > 0:
    #         avg_bands = [b/valid_channel_count for b in avg_bands]
    #         avg_lines = ("\nAverage band powers over valid channels (µV² | dB):",
    #             f"δ: {avg_bands[0]:.2f} µV² ({10*np.log10(avg_bands[0] + 1e-20):.1f} dB)",
    #             f"θ: {avg_bands[1]:.2f} µV² ({10*np.log10(avg_bands[1] + 1e-20):.1f} dB)",
    #             f"α: {avg_bands[2]:.2f} µV² ({10*np.log10(avg_bands[2] + 1e-20):.1f} dB)",
    #             f"β: {avg_bands[3]:.2f} µV² ({10*np.log10(avg_bands[3] + 1e-20):.1f} dB)",
    #             f"h-β: {avg_bands[4]:.2f} µV² ({10*np.log10(avg_bands[4] + 1e-20):.1f} dB)",
    #             f"γ: {avg_bands[5]:.2f} µV² ({10*np.log10(avg_bands[5] + 1e-20):.1f} dB)",
    #             f"h-γ: {avg_bands[6]:.2f} µV² ({10*np.log10(avg_bands[6] + 1e-20):.1f} dB)",
    #         )
    #         for l in avg_lines:
    #             print(l)
    #             report_lines.append(l)
    #     else:
    #         report_lines.append("\nNo valid channels for band power computation.")
    #         print("\nNo valid channels for band power computation.")

    #     report_lines.append("=== END DEVICE CHECK ===\n")

    #     # Render the textual summary into a dedicated PDF page
    #     txt = "\n".join(report_lines)
    #     fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape-ish
    #     ax.axis('off')
    #     # Put text block on the page. Use monospace for alignment.
    #     fig.text(0.01, 0.99, txt, va='top', ha='left', family='monospace', wrap=True, fontsize=8)
    #     pdf.savefig(fig, dpi=text_dpi)
    #     plt.close(fig)

    # print(f"Report saved to: {out_pdf}")

if __name__ == "__main__":
    main()