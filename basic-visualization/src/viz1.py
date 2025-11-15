#!/usr/bin/env python3
"""Read CSV produced by brainflow (with FreeEEG32 device) and create timeseries
and PSD using MNE

"""


import os
import argparse
import yaml
import pandas as pd
import mne


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

    # Apply bandpass filter as specified in config file
    raw.filter(l_freq=config["fmin"], h_freq=config["fmax"], verbose=args.verbose)

    # Apply notch filter for mains as specified in config file
    raw.notch_filter(freqs=config["freqs_main"], verbose=args.verbose)

    # crop if requested from cmd-line
    if args.crop and args.crop > 0:
        tmin = args.crop
        tmax = raw.times[-1] - args.crop
        raw = raw.crop(tmin=tmin, tmax=tmax)

    raw.drop_channels([ch for ch in config["channels"] if ch.startswith("U")])

    if args.average:
        raw.set_eeg_reference(ref_channels="average", verbose=args.verbose)

    return raw


def main():
    """Main actions"""
    args, config = parse_args()

    raw = compose_and_filter_raw(args, config)

    raw.plot(
        scalings={"eeg": 50e-5},
        bad_color="red",
        title=os.path.basename(args.file),
        verbose=args.verbose,
    )
    raw.compute_psd(fmin=0, fmax=200, method="welch", verbose=args.verbose).plot()
    plt.show()


if __name__ == "__main__":
    main()
