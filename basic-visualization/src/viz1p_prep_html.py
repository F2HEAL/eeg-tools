#!/usr/bin/env python3
"""Simplified EEG visualization for sweep_lsl.py output (NO HEADERS, 34 columns)"""

import os
import argparse
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import yaml
import json
import uuid
import html
from collections import defaultdict


class SimpleReport:
    """Simple HTML report generator"""
    
    def __init__(self, output_dir="reports", title="EEG Report"):
        self.output_dir = output_dir
        self.title = title
        self.sections = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, "img")):
            os.makedirs(os.path.join(output_dir, "img"))
    
    def add_section(self, title, content=None, plot=None, plot_caption=None):
        """Add a section to the report"""
        section = {
            'title': title,
            'content': content,
            'plot': plot,
            'plot_caption': plot_caption
        }
        self.sections.append(section)
    
    def save(self, filename="report.html"):
        """Save the HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                .section {{ margin: 20px 0; }}
                .plot {{ max-width: 100%; height: auto; margin: 20px 0; }}
                .caption {{ font-style: italic; color: #666; text-align: center; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>{self.title}</h1>
        """
        
        for section in self.sections:
            html_content += f"<div class='section'><h2>{html.escape(section['title'])}</h2>"
            
            # FIXED: Check if content exists properly
            content = section['content']
            if content is not None:
                if isinstance(content, pd.DataFrame):
                    if not content.empty:  # Check if DataFrame is not empty
                        html_content += content.to_html(classes='dataframe')
                    else:
                        html_content += "<p>No data</p>"
                elif isinstance(content, dict):
                    html_content += f"<pre>{html.escape(json.dumps(content, indent=2))}</pre>"
                elif isinstance(content, str):
                    html_content += f"<p>{html.escape(content)}</p>"
                else:
                    html_content += f"<p>{html.escape(str(content))}</p>"
            
            if section['plot'] is not None:
                # Save the plot
                plot_filename = f"plot_{uuid.uuid4().hex}.png"
                plot_path = os.path.join(self.output_dir, "img", plot_filename)
                section['plot'].savefig(plot_path, bbox_inches='tight', dpi=100)
                plt.close(section['plot'])
                
                html_content += f'<img src="img/{plot_filename}" class="plot">'
                if section['plot_caption']:
                    html_content += f'<p class="caption">{html.escape(section["plot_caption"])}</p>'
            
            html_content += "</div>"
        
        html_content += "</body></html>"
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to: {output_path}")
        return output_path


def load_sweep_lsl_csv(csv_file, verbose=False):
    """
    Load CSV from sweep_lsl.py (NO HEADERS format)
    Format: timestamp, Ch1, Ch2, ..., Ch32, marker (34 columns total)
    """
    if verbose:
        print(f"Loading CSV: {csv_file}")
    
    # Load without headers
    df = pd.read_csv(csv_file, header=None)
    
    if verbose:
        print(f"CSV shape: {df.shape}")
        print(f"First few rows:")
        print(df.head(3))
        print(f"\nColumn count: {df.shape[1]}")
        print("Expected format: timestamp (col 0) + 32 EEG (col 1-32) + marker (col 33)")
    
    # Extract components based on 34 columns
    n_cols = df.shape[1]
    
    if n_cols == 34:
        # Correct format: timestamp + 32 EEG + marker
        timestamps = df.iloc[:, 0].values  # First column: timestamps
        eeg_data = df.iloc[:, 1:33].values.T  # Columns 1-32: EEG data (32 channels)
        markers = df.iloc[:, 33].values  # Last column: markers
        n_eeg_channels = 32
    elif n_cols == 35:
        # Alternative format: timestamp + 33 EEG + marker
        timestamps = df.iloc[:, 0].values
        eeg_data = df.iloc[:, 1:34].values.T  # Columns 1-33: EEG data (33 channels)
        markers = df.iloc[:, 34].values
        n_eeg_channels = 33
        if verbose:
            print("Note: Using 33 EEG channels (35 columns total)")
    elif n_cols == 33:
        # Possibly: 32 EEG + marker (no timestamp)
        timestamps = np.arange(len(df)) / 1000  # Create artificial timestamps
        eeg_data = df.iloc[:, 0:32].values.T  # First 32 columns: EEG
        markers = df.iloc[:, 32].values  # Last column: marker
        n_eeg_channels = 32
        if verbose:
            print("Note: No timestamp column found, using artificial timestamps")
    else:
        # Try to infer
        print(f"Warning: Unexpected number of columns ({n_cols}). Trying to infer...")
        # Assume last column is marker, first column is timestamp, rest are EEG
        timestamps = df.iloc[:, 0].values
        eeg_data = df.iloc[:, 1:-1].values.T
        markers = df.iloc[:, -1].values
        n_eeg_channels = eeg_data.shape[0]
        if verbose:
            print(f"Inferred: {n_eeg_channels} EEG channels")
    
    # Convert to microvolts (assuming data is in volts)
    eeg_data = eeg_data * 1e6
    
    if verbose:
        print(f"\nData extracted:")
        print(f"  Timestamps shape: {timestamps.shape}")
        print(f"  EEG data shape: {eeg_data.shape} ({n_eeg_channels} channels × {eeg_data.shape[1]} samples)")
        print(f"  Markers shape: {markers.shape}")
        
        # Show marker statistics
        valid_markers = markers[~np.isnan(markers)]
        if len(valid_markers) > 0:
            unique_markers = np.unique(valid_markers)
            print(f"\nUnique markers found: {unique_markers}")
            for marker in unique_markers:
                count = np.sum(markers == marker)
                print(f"  Marker {marker}: {count} occurrences")
        else:
            print("\nNo valid markers found")
    
    return timestamps, eeg_data, markers, n_eeg_channels


def create_mne_raw(eeg_data, timestamps, sfreq=1000, channel_names=None, n_channels=None):
    """Create MNE Raw object from EEG data"""
    if n_channels is None:
        n_channels = eeg_data.shape[0]
    
    if channel_names is None:
        # Create default channel names
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        # Adjust channel names if needed
        if len(channel_names) > n_channels:
            channel_names = channel_names[:n_channels]
        else:
            channel_names = channel_names + [f'Ch{i+1}' for i in range(len(channel_names), n_channels)]
    
    # Create info structure
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    # Create Raw object
    raw = mne.io.RawArray(eeg_data, info)
    
    return raw


def add_event_annotations(raw, timestamps, markers, verbose=False):
    """Add event annotations to MNE Raw object"""
    # Find indices where we have markers
    valid_mask = ~np.isnan(markers)
    if not np.any(valid_mask):
        if verbose:
            print("No valid markers found")
        return raw
    
    marker_indices = np.where(valid_mask)[0]
    marker_values = markers[valid_mask]
    
    # Calculate onsets in seconds
    if len(timestamps) > 0:
        first_timestamp = timestamps[0]
        onsets = timestamps[marker_indices] - first_timestamp
    else:
        onsets = marker_indices / raw.info['sfreq']  # Fallback
    
    # Map marker values to descriptions
    marker_descriptions = {
        3.0: 'Baseline1_Start_VHP_OFF',
        33.0: 'Baseline1_VHP_ON',
        31.0: 'Baseline2_StimON_NoContact',
        333.0: 'Baseline3_PreSweep_Contact',
        0.0: 'Stim_Ready',
        1.0: 'Stim_Active',
        11.0: 'Stim_Off',
    }
    
    descriptions = []
    for marker in marker_values:
        # Check for exact matches first
        if marker in marker_descriptions:
            descriptions.append(marker_descriptions[marker])
        else:
            # Try integer conversion
            try:
                marker_int = int(marker)
                if float(marker_int) in marker_descriptions:
                    descriptions.append(marker_descriptions[float(marker_int)])
                else:
                    descriptions.append(f'Event_{marker_int}')
            except:
                descriptions.append(f'Event_{marker}')
    
    # Create annotations (point events with minimal duration)
    durations = [0.001] * len(onsets)  # 1 ms duration
    
    annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions
    )
    
    raw.set_annotations(annotations)
    
    if verbose:
        print(f"Added {len(annotations)} event annotations")
    
    return raw


def calculate_sampling_rate(timestamps, verbose=False):
    """Calculate sampling rate from timestamps"""
    if len(timestamps) < 2:
        return 1000  # Default
    
    time_diffs = np.diff(timestamps)
    
    # Check for consistency
    if np.std(time_diffs) / np.mean(time_diffs) > 0.1:
        if verbose:
            print(f"Warning: Timestamps inconsistent (std/mean = {np.std(time_diffs)/np.mean(time_diffs):.2f})")
            print(f"Using median interval")
        avg_interval = np.median(time_diffs)
    else:
        avg_interval = np.mean(time_diffs)
    
    sfreq = 1.0 / avg_interval
    
    if verbose:
        print(f"Average time interval: {avg_interval:.6f} seconds")
        print(f"Calculated sampling rate: {sfreq:.2f} Hz")
        print(f"Min interval: {np.min(time_diffs):.6f}, Max interval: {np.max(time_diffs):.6f}")
    
    return sfreq


def create_timeseries_plot(raw, duration=10, start_time=0):
    """Create timeseries plot with event markers"""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Get data for the specified duration
    sfreq = raw.info['sfreq']
    start_sample = int(start_time * sfreq)
    end_sample = int((start_time + duration) * sfreq)
    
    if end_sample > len(raw.times):
        end_sample = len(raw.times)
        duration = (end_sample - start_sample) / sfreq
    
    if start_sample >= end_sample:
        # Not enough data
        ax.text(0.5, 0.5, 'Not enough data for timeseries plot',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Insufficient Data')
        plt.tight_layout()
        return fig
    
    # Get data
    try:
        data, times = raw[:, start_sample:end_sample]
    except Exception as e:
        ax.text(0.5, 0.5, f'Error getting data: {str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Data Error')
        plt.tight_layout()
        return fig
    
    # Plot each channel
    n_channels = data.shape[0]
    if n_channels > 0 and data.size > 0:
        spacing = np.std(data) * 3 if np.std(data) > 0 else 50  # Adjust spacing based on data
        
        # Limit to reasonable number of channels for clarity
        max_channels_to_plot = 20
        channels_to_plot = min(n_channels, max_channels_to_plot)
        
        for i in range(channels_to_plot):
            offset = i * spacing
            ax.plot(times, data[i] + offset, linewidth=0.5, color='black', alpha=0.7)
            ax.text(times[-1] + 0.1, offset, raw.ch_names[i], 
                    ha='left', va='center', fontsize=8)
    
    # Add event markers if available
    if raw.annotations:
        from matplotlib.lines import Line2D
        
        event_colors = {
            'Baseline1_Start_VHP_OFF': '#FF6B6B',
            'Baseline1_VHP_ON': '#4ECDC4',
            'Baseline2_StimON_NoContact': '#45B7D1',
            'Baseline3_PreSweep_Contact': '#96CEB4',
            'Stim_Ready': '#FFEAA7',
            'Stim_Active': '#FF9F43',
            'Stim_Off': '#55EFC4',
        }
        
        legend_handles = []
        for onset, description in zip(raw.annotations.onset, raw.annotations.description):
            if start_time <= onset <= start_time + duration:
                color = event_colors.get(description, 'gray')
                # Draw a vertical line at event time
                ax.axvline(x=onset - start_time, color=color, alpha=0.5, linestyle='--', linewidth=1)
                
                if description not in [h.get_label() for h in legend_handles]:
                    legend_handles.append(Line2D([0], [0], color=color, lw=2, 
                                               alpha=0.5, label=description))
        
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channels')
    ax.set_yticks([])
    ax.set_title(f'EEG Timeseries with Events ({duration}s segment)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_psd_plot(raw, fmin=1, fmax=45):
    """Create PSD plot"""
    try:
        psd = raw.compute_psd(fmin=fmin, fmax=fmax, method='welch')
        fig = psd.plot(average=True, show=False)
        fig.axes[0].set_title(f'Power Spectral Density ({fmin}-{fmax} Hz)')
        plt.tight_layout()
        return fig
    except Exception as e:
        # Create a simple error plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating PSD plot:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('PSD Plot Error')
        plt.tight_layout()
        return fig


def create_event_summary(raw):
    """Create event summary table"""
    if not raw.annotations:
        return pd.DataFrame({'Event Type': ['No events found'], 'Count': [0]})
    
    event_counts = {}
    for desc in raw.annotations.description:
        event_counts[desc] = event_counts.get(desc, 0) + 1
    
    summary_df = pd.DataFrame({
        'Event Type': list(event_counts.keys()),
        'Count': list(event_counts.values())
    })
    
    return summary_df


def create_erp_plot(raw, event_type='Stim_Active', tmin=-0.2, tmax=0.8):
    """Create ERP plot for specific event type"""
    if not raw.annotations:
        return None
    
    # Find events of specified type
    event_indices = []
    for i, description in enumerate(raw.annotations.description):
        if description == event_type:
            event_indices.append(i)
    
    if len(event_indices) == 0:
        return None
    
    # Create events array
    events = []
    for idx in event_indices:
        onset = raw.annotations.onset[idx]
        sample = int(onset * raw.info['sfreq'])
        events.append([sample, 0, 1])
    
    events = np.array(events)
    
    # Create epochs
    try:
        epochs = mne.Epochs(raw, events, event_id={event_type: 1},
                           tmin=tmin, tmax=tmax, baseline=(tmin, 0),
                           preload=True)
        
        if len(epochs) == 0:
            return None
        
        # Compute average
        evoked = epochs.average()
        
        # Plot
        fig = evoked.plot(show=False)
        fig.suptitle(f'ERP: {event_type} (n={len(epochs)})', y=1.02)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        # Create error plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating ERP for {event_type}:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'ERP Error: {event_type}')
        plt.tight_layout()
        return fig


def create_frequency_band_plot(raw, band_name, fmin, fmax):
    """Create plot for specific frequency band"""
    try:
        psd = raw.compute_psd(fmin=fmin, fmax=fmax, method='welch')
        fig = psd.plot(average=False, show=False)
        fig.axes[0].set_title(f'{band_name} Power by Channel')
        plt.tight_layout()
        return fig
    except Exception as e:
        # Create error plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating {band_name} plot:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{band_name} Plot Error')
        plt.tight_layout()
        return fig


def main():
    parser = argparse.ArgumentParser(description='EEG visualization for sweep_lsl.py output')
    parser.add_argument('-f', '--file', required=True, help='Input CSV file')
    parser.add_argument('-c', '--config', required=True, help='Config YAML file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--duration', type=float, default=30, help='Timeseries duration in seconds')
    parser.add_argument('--skip-psd', action='store_true', help='Skip PSD computation')
    parser.add_argument('--skip-erp', action='store_true', help='Skip ERP computation')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Starting EEG analysis of: {args.file}")
    print(f"{'='*60}")
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        config = {}
    
    # Load CSV data
    timestamps, eeg_data, markers, n_channels = load_sweep_lsl_csv(args.file, verbose=args.verbose)
    
    print(f"\n{'='*60}")
    
    # Calculate sampling rate
    sfreq = calculate_sampling_rate(timestamps, verbose=args.verbose)
    
    if args.verbose:
        print(f"\nUsing sampling rate: {sfreq:.2f} Hz")
        print(f"EEG data range: {eeg_data.min():.2f} to {eeg_data.max():.2f} μV")
        print(f"Total duration: {len(timestamps)/sfreq:.2f} seconds")
    
    # Create MNE Raw object
    channel_names = config.get('channels', None)
    raw = create_mne_raw(eeg_data, timestamps, sfreq=sfreq, 
                        channel_names=channel_names, n_channels=n_channels)
    
    # Add event annotations
    raw = add_event_annotations(raw, timestamps, markers, verbose=args.verbose)
    
    # Create report
    report = SimpleReport(
        output_dir=config.get('report_dir', 'reports'),
        title=f"EEG Analysis: {os.path.basename(args.file)}"
    )
    
    # File information
    file_info = {
        'File': args.file,
        'Samples': len(raw.times),
        'Duration (s)': f"{raw.times[-1]:.2f}" if len(raw.times) > 0 else "0",
        'Sampling Rate (Hz)': f"{sfreq:.2f}",
        'Channels': len(raw.ch_names),
        'Data Range (μV)': f"{eeg_data.min():.2f} to {eeg_data.max():.2f}",
        'Total Events': len(raw.annotations) if raw.annotations else 0
    }
    report.add_section("File Information", content=file_info)
    
    # Event summary
    event_summary = create_event_summary(raw)
    report.add_section("Event Summary", content=event_summary)
    
    # Timeseries plot
    print("\n1. Creating timeseries plot...")
    try:
        ts_fig = create_timeseries_plot(raw, duration=args.duration)
        report.add_section("Timeseries with Events", 
                          plot=ts_fig,
                          plot_caption=f"EEG timeseries with event markers (first {args.duration}s)")
        print("   ✓ Timeseries plot created")
    except Exception as e:
        print(f"   ✗ Error creating timeseries plot: {e}")
        report.add_section("Timeseries with Events", 
                          content=f"Error creating plot: {str(e)}")
    
    # PSD plot
    if not args.skip_psd:
        print("2. Creating PSD plot...")
        try:
            psd_fig = create_psd_plot(raw)
            report.add_section("Power Spectral Density", 
                              plot=psd_fig,
                              plot_caption="Average power spectral density across all channels")
            print("   ✓ PSD plot created")
        except Exception as e:
            print(f"   ✗ Error creating PSD plot: {e}")
            report.add_section("Power Spectral Density", 
                              content=f"Error creating PSD plot: {str(e)}")
    
    # ERP plots for stimulation events
    if not args.skip_erp and raw.annotations:
        print("3. Creating ERP plots...")
        stim_events = ['Stim_Active', 'Stim_Ready', 'Stim_Off']
        erp_created = False
        
        for event_type in stim_events:
            if event_type in raw.annotations.description:
                try:
                    erp_fig = create_erp_plot(raw, event_type=event_type)
                    if erp_fig:
                        report.add_section(f"ERP: {event_type}", 
                                          plot=erp_fig,
                                          plot_caption=f"Event-related potential for {event_type}")
                        print(f"   ✓ ERP plot created for {event_type}")
                        erp_created = True
                except Exception as e:
                    print(f"   ✗ Error creating ERP plot for {event_type}: {e}")
        
        if not erp_created:
            print("   No ERP plots created (no suitable events found)")
    
    # Frequency band PSDs
    if not args.skip_psd:
        print("4. Creating frequency band plots...")
        freq_bands = {
            'Delta (0.5-4 Hz)': (0.5, 4),
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-45 Hz)': (30, 45)
        }
        
        for band_name, (fmin, fmax) in freq_bands.items():
            try:
                band_fig = create_frequency_band_plot(raw, band_name, fmin, fmax)
                report.add_section(f"{band_name} Analysis", 
                                  plot=band_fig,
                                  plot_caption=f"Power in {band_name} frequency band")
                print(f"   ✓ {band_name} plot created")
            except Exception as e:
                if args.verbose:
                    print(f"   ✗ Error creating {band_name} plot: {e}")
    
    # Save report
    report_filename = f"{os.path.basename(args.file).replace('.csv', '')}_report.html"
    report_path = report.save(report_filename)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE!")
    print(f"Report saved to: {report_path}")
    
    if raw.annotations:
        print(f"\nEVENT SUMMARY:")
        event_counts = {}
        for desc in raw.annotations.description:
            event_counts[desc] = event_counts.get(desc, 0) + 1
        
        for desc, count in event_counts.items():
            print(f"  {desc}: {count} events")
    
    print(f"\nKey findings:")
    print(f"  • {len(raw.ch_names)} EEG channels")
    print(f"  • {len(raw.times)} samples ({raw.times[-1]:.1f} seconds)")
    print(f"  • {len(raw.annotations) if raw.annotations else 0} event markers")
    print(f"  • Sampling rate: {sfreq:.1f} Hz")
    print(f"  • Data range: {eeg_data.min():.1f} to {eeg_data.max():.1f} μV")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()