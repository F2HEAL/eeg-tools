#!/usr/bin/env python3
"""Read CSV produced by brainflow (with FreeEEG32 device) and create timeseries
and PSD using MNE

"""

import os
import html
import uuid
import argparse
import yaml
import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt
from pyprep.prep_pipeline import PrepPipeline


class HTMLReport:
    """Handling HTML side of report"""

    def __init__(self, filename, output_dir="report_output", title="Analysis Report"):
        self.output_dir = output_dir
        self.filename = filename
        self.title = title
        self.body = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _tag(self, tag, content, attrs=""):
        return f"<{tag} {attrs}>{content}</{tag}>"

    def add_header(self, text, level=1):
        # Add 'collapsible' class only to level 1 headers
        attrs = 'class="collapsible"' if level == 1 else ""
        self.body.append(self._tag(f"h{level}", html.escape(text), attrs))
        self.body.append("\n")

    def add_text(self, text):
        self.body.append(self._tag("p", html.escape(text)))

    def add_object(self, obj):
        """Writes Python structures (Dict, List) or Pandas DataFrames to HTML."""
        if hasattr(obj, "to_html"):  # Pandas DataFrame support
            self.body.append(obj.to_html(classes="table table-striped"))
        elif isinstance(obj, (dict, list)):
            import json

            formatted = json.dumps(obj, indent=4, default=str)
            self.body.append(self._tag("pre", html.escape(formatted)))
        else:
            self.body.append(self._tag("pre", html.escape(str(obj))))
        self.body.append("\n")

    def add_plot(self, fig, caption=None):
        """Saves a matplotlib figure to file and embeds it."""
        img_filename = f"{self.title}-{uuid.uuid4().hex}.png"
        img_dir = os.path.join(self.output_dir, "img")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = os.path.join(img_dir, img_filename)

        fig.savefig(img_path, bbox_inches="tight")
        plt.close(fig)

        img_html = f'<img alt="{img_filename}" src="{os.path.join("img", img_filename)}" style="max-width:100%; height:auto;">'
        self.body.append(self._tag("div", img_html, 'class="plot-container"'))
        if caption:
            self.body.append(self._tag("p", html.escape(caption), 'class="caption"'))
        self.body.append("\n")

    def save(self):
        """Writes HTML file with JS/CSS for collapsible headers"""
        css = """
        body { font-family: sans-serif; margin: 40px auto; max-width: 800px; line-height: 1.6; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto;}
        .caption { font-style: italic; color: #666; text-align: center; }
        
        /* Collapsible Styling */
        .collapsible { cursor: pointer; user-select: none; }
        .collapsible::before { 
            content: '\\25B6'; /* Right-pointing arrow */
            color: #555; 
            display: inline-block; 
            margin-right: 10px; 
            transition: transform 0.2s;
        }
        .active::before { transform: rotate(90deg); } /* Down arrow */
        """

        script = """
        <script>
        document.addEventListener("DOMContentLoaded", function() {
            var headers = document.getElementsByClassName("collapsible");
            
            for (var i = 0; i < headers.length; i++) {
                // Add click listener
                headers[i].addEventListener("click", function() {
                    this.classList.toggle("active");
                    var content = this.nextElementSibling;
                    
                    // Toggle visibility of siblings until next h1
                    while (content && !content.classList.contains("collapsible")) {
                        if (content.style.display === "block") {
                            content.style.display = "none";
                        } else {
                            content.style.display = "block";
                        }
                        content = content.nextElementSibling;
                    }
                });

                // Default state: Collapse on load
                var content = headers[i].nextElementSibling;
                while (content && !content.classList.contains("collapsible")) {
                    content.style.display = "none";
                    content = content.nextElementSibling;
                }
            }
        });
        </script>
        """

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>{self.title}</title>
            <style>{css}</style>
        </head>
        <body>
            <h1>{self.title}</h1>
            {''.join(self.body)}
            {script}
        </body>
        </html>
        """

        full_path = os.path.join(self.output_dir, self.filename)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Report saved to: {full_path}")


class Report:
    """Report generation for Raw"""

    def __init__(self, args, config, prep):
        self.args = args
        self.config = config
        self.prep = prep

        self.html = HTMLReport(
            filename=f"{os.path.basename(args.file)}.html",
            output_dir=config["report_dir"],
            title=os.path.basename(args.file),
        )

    def __del__(self):
        self.html.save()

    def add_channel_quality(self):
        self.html.add_header("Channel quality")

        channel_info = {
            "Bad channels original": self.prep.noisy_channels_original["bad_all"],
            "Interpolated channels": self.prep.interpolated_channels,
            "Bad channels after interpolation": self.prep.still_noisy_channels,
        }

        self.html.add_object(channel_info)

        self.html.add_object(self.prep.noisy_channels_original)

    def add_timeseries(self):
        self.html.add_header("Timeseries")
        self.html.add_header("Timeseries 10s", level=2)

        fig = self.prep.raw.plot(
            n_channels=len(self.prep.raw.ch_names),
            show_scrollbars=False,
            show_scalebars=False,
            verbose=self.args.verbose,
        )
        self.html.add_plot(fig)

        self.html.add_header("Timeseries 1min", level=2)
        fig = self.prep.raw.plot(
            n_channels=len(self.prep.raw.ch_names),
            duration=60,
            show_scrollbars=False,
            show_scalebars=False,
            verbose=self.args.verbose,
        )

        self.html.add_plot(fig)

    def add_psd(self):
        self.html.add_header("PSD")
        self.html.add_header("PSD 0-100Hz", level=2)
        fig = self.prep.raw.compute_psd(
            fmin=0, fmax=100, method="welch", verbose=self.args.verbose
        ).plot()
        self.html.add_plot(fig)

        self.html.add_header(f"PSD 0-{self.prep.raw.info["sfreq"] / 2}Hz", level=2)
        fig = self.prep.raw.compute_psd(
            fmin=0,
            fmax=self.prep.raw.info["sfreq"] / 2,
            method="welch",
            verbose=self.args.verbose,
        ).plot()
        self.html.add_plot(fig)

    def add_spectogram(self, fmin=20, fmax=0):

        if fmax == 0:
            fmax = 0.8 * self.prep.raw.info["sfreq"] / 2

        freqs = np.arange(fmin, fmax, (fmax - fmin) / 100)
        n_cycles = freqs * 2.0

        raw = self.prep.raw.copy().filter(fmin, fmax)

        self.html.add_header(f"Spectogram {fmin}-{fmax}Hz")

        for ch in self.prep.raw.ch_names:
            power = raw.compute_tfr(
                method="morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                picks=ch,
            )

            fig, ax = plt.subplots()
            power.plot(axes=ax, show=False)

            self.html.add_header(ch, level=2)
            self.html.add_plot(fig)


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
    data_in = pd.read_csv(csv_file, header=None).values.T

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

    raw.set_montage(config["montage"], on_missing="ignore", verbose=args.verbose)

    return raw


def prep_raw(raw, config):
    montage = mne.channels.make_standard_montage(config["montage"])

    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(50, raw.info["sfreq"] / 2, 50),
    }

    prep = PrepPipeline(raw, prep_params, montage)
    prep.fit()

    return prep


# class FakePrep:
#     def __init__(self, fname):
#         self.raw = mne.io.Raw(fname, preload=True)


def main():
    """Main actions"""

    # mne.set_log_level("ERROR")
    args, config = parse_args()

    raw = mne_from_brainflow(args, config)
    prep = prep_raw(raw, config)

    # prep = FakePrep("/tmp/raw.fif")
    # prep.raw.save(fname="/tmp/raw.fif", fmt="double", verbose=True)

    report = Report(args, config, prep)
    report.add_timeseries()
    report.add_psd()
    report.add_spectogram()
    report.add_spectogram(50, 150)
    report.add_spectogram(100, 400)
    report.add_channel_quality()


if __name__ == "__main__":
    main()
