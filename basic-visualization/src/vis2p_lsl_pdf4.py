#!/usr/bin/env python3
"""
EEG Report Generator for SynAmps RT 32-channel data
Reads CSV, applies PREP filtering, and generates comprehensive PDF report
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import welch, find_peaks
from scipy.stats import kurtosis, skew
import mne
from mne.time_frequency import tfr_morlet
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class EEGReportGenerator:
    """Main class for generating EEG PDF reports"""
    
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.raw = None
        self.raw_filtered = None
        
    def read_yaml_config(self):
        """Read YAML configuration file"""
        if self.args.verbose:
            print(f"* Reading config from {self.args.config}")

        with open(self.args.config, encoding="utf8") as file:
            return yaml.safe_load(file)

    def load_data_from_csv(self):
        """Read CSV and return as MNE Raw object"""
        if self.args.verbose:
            print(f"* Reading CSV from {self.args.file}")

        csv_file = self.args.file
        data_in = pd.read_csv(csv_file, sep=",", header=None).values.T

        # Convert from CSV (V) to MNE (uV)
        data = data_in[1:len(self.config["channels"]) + 1] #* 1e6  # Convert to µV

        ch_types = ["eeg"] * len(self.config["channels"])
        sfreq = self.config["sampling_rate"]
        info = mne.create_info(
            ch_names=self.config["channels"],
            sfreq=sfreq,
            ch_types=ch_types,
            verbose=self.args.verbose,
        )

        raw = mne.io.RawArray(data, info, verbose=self.args.verbose)

        # Handle annotations if marker column exists
        if data_in.shape[0] > len(self.config["channels"]) + 1:
            events = data_in[len(self.config["channels"]) + 1]
            mask = (~np.isnan(events)) & (events != 0)
            onsets = np.arange(len(events)) / sfreq
            onsets_masked = onsets[mask]
            descriptions = [str(int(e)) for e in events[mask]]

            annotations = mne.Annotations(
                onset=onsets_masked,
                duration=[1.0 / sfreq] * len(onsets_masked),
                description=descriptions
            )
            raw.set_annotations(annotations)
            if self.args.verbose:
                print(f"Annotations loaded: {len(onsets_masked)} events")

        # Drop unwanted channels and pick specific channels from config
        try:
            drop_channels = self.config.get("drop_channels", [])
            pick_channels = self.config.get("pick_channels", [])
            
            if drop_channels:
                raw.drop_channels(drop_channels)
                if self.args.verbose:
                    print(f"Dropped channels: {drop_channels}")
            
            if pick_channels:
                raw.pick_channels(pick_channels)
                if self.args.verbose:
                    print(f"Picked channels: {pick_channels}")
            
            if self.args.verbose:
                print("Remaining channels:", raw.ch_names)
                
        except ValueError as e:
            print(f"Warning: Channel operation failed: {e}")
            print("Available channels:", raw.ch_names)

        return raw

    def apply_prep_filtering(self):
        """Apply PREP pipeline filtering"""
        if self.args.verbose:
            print("* Applying PREP filtering pipeline")
        
        # Create a copy for filtered data
        self.raw_filtered = self.raw.copy()
        
        # Apply bandpass filter
        self.raw_filtered.filter(
            l_freq=self.config["fmin"], 
            h_freq=self.config["fmax"], 
            verbose=self.args.verbose
        )
        
        # Apply notch filters for mains frequencies
        if "freqs_main" in self.config and self.config["freqs_main"]:
            self.raw_filtered.notch_filter(
                freqs=self.config["freqs_main"], 
                verbose=self.args.verbose
            )
        
        # Crop data if requested
        if self.args.crop and self.args.crop > 0:
            tmin = self.args.crop
            tmax = self.raw_filtered.times[-1] - self.args.crop
            if tmax > tmin:
                self.raw_filtered = self.raw_filtered.crop(tmin=tmin, tmax=tmax)
                if self.args.verbose:
                    print(f"* Cropped data: {tmin:.1f}s to {tmax:.1f}s")

    def plot_timeseries(self, raw, title, pdf, max_points_per_channel=2000, dpi=150):
        """Plot timeseries for all channels"""
        if self.args.verbose:
            print(f"* Plotting timeseries: {title}")
            
        # Downsample for plotting
        sfreq = raw.info['sfreq']
        decim = max(1, int(len(raw.times) / max_points_per_channel))
        times = raw.times[::decim]
        data = raw.get_data()[:, ::decim]
        
        n_channels = len(raw.ch_names)
        fig, axes = plt.subplots(n_channels, 1, figsize=(16, 2 * n_channels))
        if n_channels == 1:
            axes = [axes]
        
        for i, (ax, ch_name) in enumerate(zip(axes, raw.ch_names)):
            ax.plot(times, data[i], linewidth=0.5, color='blue')
            ax.set_ylabel(f'{ch_name}\n(µV)', rotation=0, ha='right')
            ax.set_ylim([np.percentile(data[i], 1), np.percentile(data[i], 99)])
            ax.grid(True, alpha=0.3)
            
            # Add annotations
            if raw.annotations and i == 0:  # Only on first channel to avoid clutter
                for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
                    if times[0] <= onset <= times[-1]:
                        ax.axvline(onset, color='red', linestyle='--', alpha=0.7)
                        ax.text(onset, ax.get_ylim()[1], desc, 
                               rotation=90, verticalalignment='top', fontsize=8, color='red')
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{title}\n{os.path.basename(self.args.file)}', fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

    def plot_psd_with_peaks(self, raw, pdf, dpi=150):
        """Plot PSD with configurable scale (linear/log) and peak detection"""
        if self.args.verbose:
            print("* Computing PSD with peak detection")
            
        sfreq = raw.info['sfreq']
        fmin = self.config.get("fmin", 1)
        fmax = self.config.get("fmax", 100)
        
        # Get configuration
        min_snr_db = self.config.get("peak_snr_threshold", 1.0)
        scale_type = self.config.get("psd_scale", "linear")  # "linear" or "log"
        peak_distance = self.config.get("peak_min_distance", 5)  # Minimum frequency bins between peaks
        
        if self.args.verbose:
            print(f"* Using SNR threshold: {min_snr_db} dB")
            print(f"* Using {scale_type.upper()} scale")
            print(f"* Peak minimum distance: {peak_distance} frequency bins")
        
        # Compute PSD
        psd_obj = raw.compute_psd(
            fmin=fmin, 
            fmax=fmax, 
            method="welch", 
            verbose=self.args.verbose
        )
        psds, freqs = psd_obj.get_data(return_freqs=True)
        
        # Calculate global y-axis limits based on scale type - FIXED VERSION
        all_psd_data = psds.flatten()
        
        if scale_type == "linear":
            # For linear scale: use actual data range with small margins
            y_min = 0
            y_max = np.max(all_psd_data) * 1.05  # 5% margin above max value
            
            # If there are extreme outliers, cap at 99.9th percentile
            p999 = np.percentile(all_psd_data, 99.9)
            if y_max > p999 * 5:  # If max is way beyond 99.9th percentile
                y_max = p999 * 1.1
                if self.args.verbose:
                    print(f"* Capping y_max due to outliers: {y_max:.2e}")
                    
        else:  # log scale
            # For log scale: use wider range to ensure all data is visible
            y_min = np.min(all_psd_data) * 0.9  # 10% margin below min
            y_max = np.max(all_psd_data) * 1.1  # 10% margin above max
            
            # Ensure reasonable limits for log scale
            y_min = max(y_min, 1e-12)
            y_max = max(y_max, y_min * 10)  # Ensure minimum range
            
            # If range is too extreme, use percentiles
            if y_max / y_min > 1e6:  # Extreme dynamic range
                y_min = np.percentile(all_psd_data, 1)  # 1st percentile
                y_max = np.percentile(all_psd_data, 99) * 2  # 99th percentile with margin
                if self.args.verbose:
                    print(f"* Using percentiles due to extreme range: {y_min:.2e} to {y_max:.2e}")
        
        # FINAL SAFETY CHECK: Ensure all data fits within bounds
        actual_min = np.min(all_psd_data)
        actual_max = np.max(all_psd_data)
        
        if actual_min < y_min:
            y_min = actual_min * 0.95  # Ensure min data is visible
        if actual_max > y_max:
            y_max = actual_max * 1.05  # Ensure max data is visible
        
        if self.args.verbose:
            print(f"* Data range: {actual_min:.2e} to {actual_max:.2e} µV²/Hz")
            print(f"* {scale_type.upper()} PSD y-axis range: {y_min:.2e} to {y_max:.2e} µV²/Hz")
            print(f"* Dynamic range: {y_max/y_min:.1f}x")
        
        n_channels = len(raw.ch_names)
        
        # Create figure with adjusted height to accommodate title
        fig_height = 2 * n_channels + 1  # Extra space for title
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, fig_height))
        if n_channels == 1:
            axes = [axes]
        
        total_peaks_all_channels = 0
        peak_neighborhood = self.config.get("peak_neighborhood_hz", 4.0) # Hz around peak to exclude
        
        for i, (ax, ch_name) in enumerate(zip(axes, raw.ch_names)):
            psd_data = psds[i]
            
            # Find all local peaks first
            peaks, _ = find_peaks(psd_data, distance=peak_distance)
            
            # Calculate SNR in dB for each peak and filter by threshold
            significant_peaks = []
            
            for peak_idx in peaks:
                freq_val = freqs[peak_idx]
                peak_power = psd_data[peak_idx]
                
                # Define neighborhood around peak (excluding the peak itself)
                neighbor_low = max(fmin, freq_val - peak_neighborhood)
                neighbor_high = min(fmax, freq_val + peak_neighborhood)
                
                # Exclude the peak frequency ±1 Hz
                neighbor_mask = (
                    (freqs >= neighbor_low) & 
                    (freqs <= neighbor_high) &
                    (np.abs(freqs - freq_val) > 1.0)  # Exclude peak region
                )
                
                if np.sum(neighbor_mask) >= 3:  # Need at least 3 neighbor points
                    neighbor_powers = psd_data[neighbor_mask]
                    noise_floor = np.median(neighbor_powers)
                    
                    if noise_floor > 0:
                        snr_db = 10 * np.log10(peak_power / noise_floor)
                        
                        # Apply configurable dB threshold
                        if snr_db >= min_snr_db:
                            significant_peaks.append({
                                'index': peak_idx,
                                'freq': freq_val,
                                'power': peak_power,
                                'snr_db': snr_db
                            })
            
            # Plot PSD with selected scale type
            if scale_type == "linear":
                ax.plot(freqs, psd_data, linewidth=1, color='blue', label='PSD')
            else:
                ax.semilogy(freqs, psd_data, linewidth=1, color='blue', label='PSD')
            
            # Set uniform y-limits for all channels
            ax.set_ylim([y_min, y_max])
            
            # Only show full y-label on first subplot
            if i == 0:
                ax.set_ylabel(f'{ch_name}\n(µV²/Hz)', rotation=0, ha='right', fontsize=8)
            else:
                #ax.set_ylabel(ch_name, rotation=0, ha='right', fontsize=8)
                ax.set_ylabel(f'{ch_name}\n(µV²/Hz)', rotation=0, ha='right', fontsize=8)
                    
            ax.grid(True, alpha=0.3)
            
            # Plot significant peaks
            if len(significant_peaks) > 0:
                peak_indices = [p['index'] for p in significant_peaks]
                peak_freqs = [p['freq'] for p in significant_peaks]
                peak_powers = [p['power'] for p in significant_peaks]
                
                ax.plot(peak_freqs, peak_powers, 'ro', markersize=5, 
                    label=f'SNR≥{min_snr_db}dB (n={len(significant_peaks)})')
                
                total_peaks_all_channels += len(significant_peaks)
                
                # Annotate peaks with frequency and SNR
                for peak_info in significant_peaks:
                    freq_val = peak_info['freq']
                    peak_power = peak_info['power']
                    snr_db = peak_info['snr_db']
                    
                    # Only annotate if within visible y-range and SNR is decent
                    if y_min <= peak_power <= y_max and snr_db >= min_snr_db:
                        # Use different colors based on SNR strength
                        if snr_db >= 6:
                            color = 'darkred'
                            bbox_color = 'red'
                        elif snr_db >= 3:
                            color = 'red' 
                            bbox_color = 'orange'
                        else:
                            color = 'coral'
                            bbox_color = 'yellow'
                        
                        ax.annotate(f'{freq_val:.1f}Hz\n{snr_db:.1f}dB', 
                                xy=(freq_val, peak_power),
                                xytext=(8, 8), textcoords='offset points',
                                fontsize=6, alpha=0.8, color=color,
                                bbox=dict(boxstyle="round,pad=0.2", 
                                        facecolor="white", 
                                        alpha=0.8,
                                        edgecolor=bbox_color,
                                        linewidth=1.0))
                
            ax.legend(fontsize=7, loc='upper right')
            ax.set_xlim([fmin, fmax])
        
        axes[-1].set_xlabel('Frequency (Hz)')
        
        # Create informative title based on scale type
        if scale_type == "linear":
            scale_advantage = "Better Peak Visibility"
        else:
            scale_advantage = "Wide Dynamic Range"
            
        title = (f'Power Spectral Density - {scale_type.upper()} Scale ({scale_advantage})\n'
                f'Peaks with SNR ≥ {min_snr_db} dB | Uniform Y-Scale: {y_min:.2e} to {y_max:.2e} µV²/Hz\n'
                f'Total peaks detected: {total_peaks_all_channels}\n'
                f'{os.path.basename(self.args.file)}')
        
        # Set the suptitle with adjusted position
        fig.suptitle(title, fontsize=11, y=0.98)  # y=0.98 moves it down slightly
        
        # Use constrained_layout instead of tight_layout for better title spacing
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # rect=[left, bottom, right, top] - reserve top 4% for title
        
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)
        
        if self.args.verbose:
            print(f"* Total peaks detected across all channels: {total_peaks_all_channels}")
        
        return psds, freqs

    def plot_spectrograms_multitaper(self, raw, pdf, dpi=100):
        """Use multitaper method for faster computation"""
        if self.args.verbose:
            print("* Computing spectrograms (multitaper - fast)")
        
        # Pre-downsample
        raw_resampled = raw.copy()
        if raw.info['sfreq'] > 200:
            raw_resampled.resample(200)
        
        # Define frequencies
        fmin = max(self.config.get("fmin", 1), 1)
        fmax = self.config.get("fmax", 100)
        freqs = np.linspace(fmin, fmax, 60)  # Even fewer frequencies
        
        n_channels = len(raw.ch_names)
        
        for ch_idx, ch_name in enumerate(raw.ch_names):
            # Use multitaper method - much faster than Morlet
            tfr = mne.time_frequency.tfr_multitaper(
                raw_resampled,
                picks=[ch_idx],
                freqs=freqs,
                n_cycles=freqs / 2,
                time_bandwidth=4.0,  # Standard setting
                use_fft=True,
                return_itc=False,
                decim=10,
                average=False,
                verbose=False
            )
            
            data_db = 10 * np.log10(tfr.data[0] + 1e-20)
            
            # Downsample for plotting
            max_time_bins = 150
            time_step = max(1, data_db.shape[1] // max_time_bins)
            data_db_ds = data_db[:, ::time_step]
            times_ds = tfr.times[::time_step]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            pcm = ax.pcolormesh(times_ds, freqs, data_db_ds, 
                            shading='gouraud', cmap='viridis', rasterized=True)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Spectrogram: {ch_name}')
            plt.colorbar(pcm, ax=ax, label='Power (dB)')
            plt.tight_layout()
            
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    def plot_spectrograms_stft(self, raw, pdf, dpi=100):
        """Use STFT for fastest spectrogram computation"""
        if self.args.verbose:
            print("* Computing spectrograms (STFT - fastest)")
        
        # Pre-process data
        raw_resampled = raw.copy()
        if raw.info['sfreq'] > 1250:
            raw_resampled.resample(250)
        
        data = raw_resampled.get_data()
        sfreq = raw_resampled.info['sfreq']
        times = raw_resampled.times
        
        # STFT parameters
        nperseg = int(sfreq * 1.0)  # 1-second windows
        noverlap = int(sfreq * 0.5)  # 50% overlap
        
        fmin = self.config.get("fmin", 1)
        fmax = self.config.get("fmax", 499)
        
        for ch_idx, ch_name in enumerate(raw.ch_names):
            # Compute STFT
            from scipy.signal import stft
            f, t, Zxx = stft(data[ch_idx], fs=sfreq, nperseg=nperseg, 
                            noverlap=noverlap, boundary=None)
            
            # Filter frequencies
            freq_mask = (f >= fmin) & (f <= fmax)
            f_filtered = f[freq_mask]
            Zxx_filtered = Zxx[freq_mask]
            
            # Convert to power (dB)
            power = np.abs(Zxx_filtered) ** 2
            power_db = 10 * np.log10(power + 1e-20)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            pcm = ax.pcolormesh(t, f_filtered, power_db, 
                            shading='gouraud', cmap='viridis', rasterized=True)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Spectrogram (STFT): {ch_name}')
            plt.colorbar(pcm, ax=ax, label='Power (dB)')
            plt.tight_layout()
            
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    def plot_spectrograms_ultrafast(self, raw, pdf, dpi=100):
        """Ultra-fast spectrograms with strategic trade-offs"""
        if self.args.verbose:
            print("* Computing spectrograms (ultra-fast)")
        
        # Aggressive resampling
        raw_fast = raw.copy()
        target_sfreq = min(100, raw.info['sfreq'])  # Max 100 Hz
        raw_fast.resample(target_sfreq)
        
        # Very limited frequency range
        freqs = np.array([2, 4, 8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100])
        n_cycles = freqs / 2.0
        
        # Process all channels at once (if memory permits)
        try:
            tfr = tfr_morlet(
                raw_fast,
                picks='eeg',
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                decim=10,
                average=False,
                verbose=False,
                n_jobs=4  # Use multiple cores
            )
            
            # Plot each channel
            for ch_idx, ch_name in enumerate(raw.ch_names):
                data_db = 10 * np.log10(tfr.data[ch_idx] + 1e-20)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                pcm = ax.pcolormesh(tfr.times, freqs, data_db, 
                                shading='gouraud', cmap='viridis', rasterized=True)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_title(f'Spectrogram: {ch_name}')
                plt.colorbar(pcm, ax=ax, label='Power (dB)')
                plt.tight_layout()
                
                pdf.savefig(fig, dpi=dpi)
                plt.close(fig)
                
        except MemoryError:
            # Fall back to sequential processing if memory issues
            self.plot_spectrograms_multitaper(raw, pdf, dpi)

    def plot_spectrograms(self, raw, pdf, dpi=100):
        """Plot spectrograms for each channel using Morlet wavelets"""
        if self.args.verbose:
            print("* Computing spectrograms")
            
        fmin = max(self.config.get("fmin", 1), 1)  # Morlet requires freq > 0
        fmax = self.config.get("fmax", 100)
        freqs = np.arange(fmin, fmax + 1)
        n_cycles = freqs / 2.0
        
        n_channels = len(raw.ch_names)
        
        for ch_idx, ch_name in enumerate(raw.ch_names):
            if self.args.verbose:
                print(f"  - Processing {ch_name}")
                
            # Compute TFR for single channel
            tfr = tfr_morlet(
                raw,
                picks=[ch_idx],
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                decim=10,  # Aggressive decimation for speed
                average=False,
                verbose=False
            )
            
            # Convert to dB and downsample for plotting
            data_db = 10 * np.log10(tfr.data[0] + 1e-20)
            max_time_bins = 300
            time_step = max(1, data_db.shape[1] // max_time_bins)
            data_db_ds = data_db[:, ::time_step]
            times_ds = tfr.times[::time_step]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            pcm = ax.pcolormesh(times_ds, freqs, data_db_ds, 
                              shading='gouraud', cmap='viridis', rasterized=True)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Spectrogram: {ch_name}\n{os.path.basename(self.args.file)}')
            
            # Add colorbar
            plt.colorbar(pcm, ax=ax, label='Power (dB)')
            
            # Add annotations
            if raw.annotations:
                for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
                    if times_ds[0] <= onset <= times_ds[-1]:
                        ax.axvline(onset, linestyle='--', color='white', linewidth=1, alpha=0.7)
                        y_text = freqs.min() + (freqs.max() - freqs.min()) * 0.02
                        ax.text(onset, y_text, str(desc), rotation=90,
                              verticalalignment='bottom', fontsize=8, color='white')
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    def band_power(self, freqs, psd, fmin, fmax):
        """Calculate band power"""
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return 0.0
        return float(np.trapz(psd[mask], freqs[mask]))

    def spectral_entropy(self, psd):
        """Calculate spectral entropy"""
        psd = np.asarray(psd)
        if psd.size == 0:
            return 0.0
        total_power = np.sum(psd)
        if total_power <= 0 or not np.isfinite(total_power):
            return 0.0
        p = psd / total_power
        H = -np.sum(p * np.log2(p + 1e-20))
        return float(H)

    def line_noise_ratio(self, freqs, psd, fline=50.0, fband=1.0):
        """Calculate line noise ratio"""
        freqs = np.asarray(freqs)
        psd = np.asarray(psd)
        idx_line = np.where((freqs >= fline - fband) & (freqs <= fline + fband))[0]
        idx_baseline = np.where(
            ((freqs >= fline - 3*fband) & (freqs < fline - fband)) |
            ((freqs > fline + fband) & (freqs <= fline + 3*fband))
        )[0]
        if len(idx_line) == 0 or len(idx_baseline) == 0:
            return 0.0
        line_p = np.mean(psd[idx_line])
        base_p = np.mean(psd[idx_baseline])
        if base_p <= 0 or not np.isfinite(base_p):
            return 0.0
        return float(10 * np.log10((line_p + 1e-20) / (base_p + 1e-20)))

    def generate_eeg_quality_report(self, raw, psds, freqs, pdf, dpi=150):
        """Generate comprehensive EEG quality report"""
        if self.args.verbose:
            print("* Generating EEG quality report")
            
        data_all = raw.get_data()
        ch_names = raw.ch_names
        
        # Prepare data for report
        eeg_checks = []
        band_powers = []
        avg_bands = np.zeros(7)  # delta, theta, alpha, beta, h_beta, gamma, h_gamma
        valid_channel_count = 0
        
        # Band definitions
        bands = [
            (1.0, 4.0, "Delta"),
            (4.0, 8.0, "Theta"),
            (8.0, 13.0, "Alpha"),
            (13.0, 20.0, "Beta"),
            (20.0, 30.0, "High Beta"),
            (30.0, 60.0, "Gamma"),
            (60.0, 98.0, "High Gamma")
        ]
        
        for i, ch_name in enumerate(ch_names):
            sig = data_all[i]
            psd_ch = psds[i]
            
            # Time-domain metrics
            ptp = np.ptp(sig)
            rms = np.sqrt(np.mean(sig**2))
            dc = np.mean(sig)
            diffs = np.diff(sig)
            flat = np.sum(np.abs(diffs) < 3e-6) / len(diffs) if len(diffs) > 0 else 0
            k = kurtosis(sig)
            s = skew(sig)
            
            # Frequency-domain metrics
            entropy = self.spectral_entropy(psd_ch)
            ln = self.line_noise_ratio(freqs, psd_ch)
            
            # Calculate band powers
            band_power_vals = []
            is_bad = ptp < 10 or flat > 0.95  # Simple bad channel detection
            
            for fmin, fmax, band_name in bands:
                if is_bad:
                    bp = 0.0
                else:
                    bp = self.band_power(freqs, psd_ch, fmin, fmax)
                band_power_vals.append(bp)
            
            if not is_bad:
                avg_bands += np.array(band_power_vals)
                valid_channel_count += 1
            
            # Muscle ratio
            gamma_power = band_power_vals[5]  # Gamma band
            alpha_beta_power = band_power_vals[2] + band_power_vals[3] + band_power_vals[4]
            muscle_ratio = gamma_power / (alpha_beta_power + 1e-20)
            
            # Channel status determination
            status = "GOOD"
            if ptp < 10:
                status = "SHORTED"
            elif flat > 0.95:
                status = "FLATLINE"
            elif ptp > 800:
                status = "NOISY/SATURATION"
            elif ln > 20:
                status = "HIGH LINE-NOISE"
            elif muscle_ratio > 1.0:
                status = "EMG/MUSCLE"
            elif entropy > 5.0:
                status = "RANDOM NOISE"
            
            status2 = "UNKNOWN"
            if 2.5 < entropy < 4.8:
                status2 = "HUMAN EEG"
            elif entropy > 4.8:
                status2 = "RANDOM NOISE"
            
            # Store results
            eeg_checks.append({
                'channel': ch_name,
                'ptp': ptp,
                'rms': rms,
                'entropy': entropy,
                'lnr': ln,
                'muscle_ratio': muscle_ratio,
                'dc': dc,
                'flat': flat,
                'kurtosis': k,
                'skew': s,
                'status': status,
                'status2': status2
            })
            
            band_powers.append({
                'channel': ch_name,
                'bands': band_power_vals,
                'band_names': [b[2] for b in bands]
            })
        
        # Calculate averages
        if valid_channel_count > 0:
            avg_bands /= valid_channel_count
        
        # Create visualization
        self._plot_eeg_quality_report(eeg_checks, band_powers, avg_bands, bands, pdf, dpi)
        
        return eeg_checks, band_powers, avg_bands

    def _plot_eeg_quality_report(self, eeg_checks, band_powers, avg_bands, bands, pdf, dpi):
        """Plot EEG quality assessment"""
        fig = plt.figure(figsize=(16, 12))
        
        # Channel quality overview
        ax1 = plt.subplot(2, 1, 1)
        channels = [ec['channel'] for ec in eeg_checks]
        ptp_vals = [ec['ptp'] for ec in eeg_checks]
        entropy_vals = [ec['entropy'] for ec in eeg_checks]
        lnr_vals = [ec['lnr'] for ec in eeg_checks]
        
        x = np.arange(len(channels))
        width = 0.25
        
        ax1.bar(x - width, ptp_vals, width, label='PTP (µV)', alpha=0.7)
        ax1.bar(x, entropy_vals, width, label='Spectral Entropy', alpha=0.7)
        ax1.bar(x + width, lnr_vals, width, label='Line Noise (dB)', alpha=0.7)
        
        ax1.set_xlabel('Channels')
        ax1.set_ylabel('Metrics')
        ax1.set_title('EEG Channel Quality Assessment')
        ax1.set_xticks(x)
        ax1.set_xticklabels(channels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Band power overview
        ax2 = plt.subplot(2, 1, 2)
        band_names = [b[2] for b in bands]
        x_band = np.arange(len(band_names))
        
        # Plot average band powers
        bars = ax2.bar(x_band, avg_bands, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Frequency Bands')
        ax2.set_ylabel('Power (µV²)')
        ax2.set_title('Average Band Powers Across Channels')
        ax2.set_xticks(x_band)
        ax2.set_xticklabels(band_names)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_bands):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)
        
        # Create text summary
        fig = plt.figure(figsize=(16, 10))
        plt.axis('off')
        
        summary_text = "EEG QUALITY SUMMARY\n\n"
        summary_text += "CHANNEL STATUS:\n"
        for ec in eeg_checks:
            summary_text += f"{ec['channel']:8s}: {ec['status']:20s} | Entropy: {ec['entropy']:4.2f} | LNR: {ec['lnr']:5.1f} dB\n"
        
        summary_text += "\nAVERAGE BAND POWERS:\n"
        for (fmin, fmax, band_name), power in zip(bands, avg_bands):
            summary_text += f"{band_name:10s} ({fmin:3.0f}-{fmax:3.0f} Hz): {power:8.2f} µV² ({10*np.log10(power + 1e-20):6.1f} dB)\n"
        
        plt.text(0.02, 0.98, summary_text, fontfamily='monospace', fontsize=8, 
                verticalalignment='top', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

    def generate_report(self):
        """Main method to generate complete PDF report"""
        # Determine output path
        if self.args.out:
            output_path = self.args.out
        else:
            base_name = os.path.splitext(self.args.file)[0]
            output_path = f"{base_name}_report.pdf"
        
        if self.args.verbose:
            print(f"* Output will be saved to: {output_path}")
        
        # Load data
        self.raw = self.load_data_from_csv()
        
        # Apply PREP filtering
        self.apply_prep_filtering()
        
        # Generate PDF report
        with PdfPages(output_path) as pdf:
            if self.args.verbose:
                print("* Generating PDF report...")
            
            # 1. Raw timeseries
            self.plot_timeseries(self.raw, "Raw EEG Timeseries", pdf)
            
            # 2. Filtered timeseries  
            self.plot_timeseries(self.raw_filtered, "PREP Filtered EEG Timeseries", pdf)
            
            # 3. PSD with peaks
            psds, freqs = self.plot_psd_with_peaks(self.raw_filtered, pdf)
            
            # 4. Spectrograms
            if self.args.ultra_fast:
                self.plot_spectrograms_stft(self.raw_filtered, pdf)
            elif self.args.fast:
                self.plot_spectrograms_multitaper(self.raw_filtered, pdf)
            elif self.args.std:
                self.plot_spectrograms(self.raw_filtered, pdf)
            else:
                self.plot_spectrograms_fast(self.raw_filtered, pdf)  # Parallel version


            
            # 5. EEG quality report
            eeg_checks, band_powers, avg_bands = self.generate_eeg_quality_report(
                self.raw_filtered, psds, freqs, pdf
            )
            
            if self.args.verbose:
                print(f"* Report saved to: {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create EEG PDF report.")
    
    parser.add_argument("-f", "--file", type=str, required=True, 
                       help="Input CSV file path")
    parser.add_argument("-c", "--config", type=str, required=True,
                       help="Configuration YAML file path")
    parser.add_argument("-r", "--crop", type=int, required=False,
                       help="Leading/trailing seconds to drop")
    parser.add_argument("-a", "--average", action="store_true",
                       help="Use average reference (not implemented)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("-o", "--out", type=str, required=False,
                       help="Output PDF path. Default: <input_basename>_report.pdf")
    parser.add_argument("-s", "--fast-spectrograms", action="store_true",
                       help="Use faster multitaper method for spectrograms")
    parser.add_argument("-u", "--ultra-fast", action="store_true",
                       help="Use STFT for fastest spectrogram computation")
    parser.add_argument("-d", "--standard", action="store_true",
                       help="Use standard method for spectrogram computation")
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    # Check if files exist
    if not os.path.exists(args.file):
        print(f"Error: Input file {args.file} not found")
        return 1
        
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found")
        return 1
    
    try:
        # Initialize and run report generator
        generator = EEGReportGenerator(args, {})
        generator.config = generator.read_yaml_config()
        generator.generate_report()
        
        if args.verbose:
            print("✓ Report generation completed successfully")
            
    except Exception as e:
        print(f"Error during report generation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())