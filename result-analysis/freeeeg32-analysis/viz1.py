import argparse
import yaml
import numpy as np
import pandas as pd
import mne

import matplotlib.pyplot as plt


def read_yaml_config(args):
    if args.verbose:
        print(f"* Reading config from {args.config}")

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
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
        "-i", "--interpolate", action="store_true", help="Iterpolate based on timestamp"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    config = read_yaml_config(args)

    return args, config


def interpolate(df):
    values = df.iloc[:, 1:33].values  # columns 2-33
    timestamps = df.iloc[:, 33].values  # column 34
    events = df.iloc[:, 34].values  # column 35

    avg_sampling_period = np.mean(np.diff(timestamps))
    print(f"Original Sampling rate: {1/avg_sampling_period:.2f} Hz")
    uniform_timestamps = np.arange(timestamps[0], timestamps[-1], avg_sampling_period)
    print(f"Interpolation enabled. New Sampling rate: {1/avg_sampling_period:.2f} Hz")

    # ------------------------------
    # Interpolate measured values (columns 2-33)
    # ------------------------------
    interp_values = np.empty((len(uniform_timestamps), values.shape[1]))
    for c in range(values.shape[1]):
        interp_values[:, c] = np.interp(uniform_timestamps, timestamps, values[:, c])

    # ------------------------------
    # Interpolate events (nearest neighbor)
    # ------------------------------
    interp_events = np.zeros_like(uniform_timestamps)
    event_indices = np.where(events != 0)[0]
    for idx in event_indices:
        orig_time = timestamps[idx]
        new_idx = np.abs(uniform_timestamps - orig_time).argmin()
        interp_events[new_idx] = events[idx]

        # ------------------------------
    # Assemble result DataFrame
    # ------------------------------
    result_df = pd.DataFrame(
        np.hstack([interp_values, uniform_timestamps[:, None], interp_events[:, None]]),
        columns=[f"val{i}" for i in range(1, 33)] + ["timestamp", "event"],
    )

    return result_df


def mne_from_brainflow(args, config):
    if args.verbose:
        print(f"* Reading Brainflow CSV from {args.file}")

    csv_file = args.file
    df = pd.read_csv(csv_file, sep="\t", header=None)

    if args.interpolate:
        df = interpolate(df)

    data_in = df.values.T
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
    raw = mne_from_brainflow(args, config)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=args.verbose)

    # Apply bandpass filter as specified in config file
    raw.filter(l_freq=config["fmin"], h_freq=config["fmax"], verbose=args.verbose)

    # Apply notch filter for mains as specified in config file
    raw.notch_filter(freqs=config["freqs_main"], verbose=args.verbose)

    return raw


def main():
    args, config = parse_args()

    raw = compose_and_filter_raw(args, config)

    # crop if requested from cmd-line
    if args.crop and args.crop > 0:
        tmin = args.crop
        tmax = raw.times[-1] - args.crop
        raw = raw.crop(tmin=tmin, tmax=tmax)

    raw.drop_channels([ch for ch in config["channels"] if ch.startswith("U")])

    if args.average:
        raw.set_eeg_reference(ref_channels="average", verbose=args.verbose)

    # raw.plot(scalings="auto", bad_color="red", verbose=args.verbose)
    # raw.plot(scalings={"eeg": 50e-5}, bad_color="red", verbose=args.verbose)
    raw.compute_psd(fmin=0, fmax=200, verbose=args.verbose).plot()
    plt.show()


if __name__ == "__main__":
    main()
