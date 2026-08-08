"""Re-render the stitched random-string BF figures locally from the CSVs.

The v2 pipeline already dumped one ``stitched_bf.csv`` (schema: model, series, x, bf)
next to every ``stitched_bf.pdf`` / ``stitched_bf.png``. To tweak visualization
details there is no need to re-fetch anything from the server: this script reads
those CSVs and re-plots, overwriting the PDF (and PNG) in place.

Plot styling lives in ``stitched_plot_utils.plot_rows`` so it stays identical to
what the server pipeline produces. Edit that function to change the look.

Examples (run from the repo root):
    python tmlr_additional_experiments/replot_stitched_from_csv.py
    python tmlr_additional_experiments/replot_stitched_from_csv.py --root tmlr_additional_experiments/stitched_plots__entropy_only --log_y
"""

import argparse
import glob
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from stitched_plot_utils import plot_rows, read_rows_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        default=os.path.join(THIS_DIR, "stitched_plots__entropy_only"),
        help="Directory to scan for stitched CSVs (default: the downloaded "
             "stitched_plots__entropy_only folder).",
    )
    parser.add_argument(
        "--pattern",
        default="**/stitched_bf.csv",
        help="Glob (relative to --root, recursive) for the CSVs to re-plot.",
    )
    parser.add_argument("--log_y", action="store_true", help="Use a log-scaled BF (y) axis.")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    parser.add_argument("--figsize", default="9,5", help="Figure size as 'width,height' in inches.")
    parser.add_argument("--no_pdf", action="store_true", help="Only write the PNG, leave the PDF as-is.")
    parser.add_argument("--no_png", action="store_true", help="Only write the PDF, do not touch the PNG.")
    return parser.parse_args()


def main():
    args = parse_args()
    width, height = (float(v) for v in args.figsize.split(","))
    csv_paths = sorted(glob.glob(os.path.join(args.root, args.pattern), recursive=True))
    if not csv_paths:
        raise SystemExit(f"No CSVs matching {args.pattern!r} under {args.root}")

    written = 0
    for csv_path in csv_paths:
        model, rows = read_rows_csv(csv_path)
        if not rows:
            print(f"Skipping empty CSV: {csv_path}")
            continue
        out_dir = os.path.dirname(csv_path)
        # Keep the original basename (stitched_bf.png / .pdf) so we overwrite in place.
        base = os.path.splitext(os.path.basename(csv_path))[0]
        png_path = os.path.join(out_dir, base + ".png")
        ok = plot_rows(
            png_path, model, rows,
            figsize=(width, height), dpi=args.dpi, log_y=args.log_y,
            also_pdf=not args.no_pdf,
        )
        if ok and args.no_png and os.path.exists(png_path):
            os.remove(png_path)
        written += int(bool(ok))

    print(f"Re-plotted {written}/{len(csv_paths)} figure(s) under {args.root}")


if __name__ == "__main__":
    main()
