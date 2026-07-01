import argparse
import re
import pandas as pd


def parse_vdd_in_w(tegrastats_log):
    values_mw = []

    with open(tegrastats_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Example: VDD_IN 7587mW/7520mW
            m = re.search(r"VDD_IN\s+(\d+)mW", line)
            if m:
                values_mw.append(int(m.group(1)))

    if not values_mw:
        return None, None, 0

    avg_w = sum(values_mw) / len(values_mw) / 1000
    peak_w = max(values_mw) / 1000
    return avg_w, peak_w, len(values_mw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Evaluation CSV")
    parser.add_argument("--tegrastats", required=True, help="tegrastats log file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if "wall_s" not in df.columns:
        raise ValueError("CSV must contain wall_s column")

    avg_power_w, peak_power_w, samples = parse_vdd_in_w(args.tegrastats)

    if avg_power_w is None:
        raise ValueError("No VDD_IN power values found in tegrastats log")

    avg_wall_s = df["wall_s"].mean()
    median_wall_s = df["wall_s"].median()
    p95_wall_s = df["wall_s"].quantile(0.95)

    cold_j_per_image = avg_power_w * avg_wall_s

    print("=" * 80)
    print("ENERGY SUMMARY")
    print("CSV:", args.csv)
    print("Tegrastats:", args.tegrastats)
    print("Images:", len(df))
    print("Power samples:", samples)

    print("\nPOWER")
    print("Avg VDD_IN W:", avg_power_w)
    print("Peak VDD_IN W:", peak_power_w)

    print("\nLATENCY")
    print("Avg wall_s/image:", avg_wall_s)
    print("Median wall_s/image:", median_wall_s)
    print("P95 wall_s/image:", p95_wall_s)

    print("\nENERGY")
    print("Cold-start Joules/image:", cold_j_per_image)

    if "mtmd_encode_ms" in df.columns:
        vals = pd.to_numeric(df["mtmd_encode_ms"], errors="coerce").dropna()
        if len(vals) > 0:
            avg_mtmd_s = vals.mean() / 1000
            print("Avg mtmd_encode_s/image:", avg_mtmd_s)
            print("Estimated image-encoding Joules/image:", avg_power_w * avg_mtmd_s)


if __name__ == "__main__":
    main()
