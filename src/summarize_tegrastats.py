import argparse
import re
import statistics


def mean_or_none(values):
    return statistics.mean(values) if values else None


def max_or_none(values):
    return max(values) if values else None


def print_metric(name, value, suffix=""):
    if value is None:
        print(f"{name}: no valid values")
    else:
        print(f"{name}: {value:.3f}{suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Path to tegrastats log file")
    args = parser.parse_args()

    ram_used_mb = []
    ram_total_mb = []
    swap_used_mb = []
    swap_total_mb = []
    gpu_util_pct = []
    cpu_utils = []
    vdd_in_mw = []
    temps_c = []

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # RAM 3902/7620MB
            m = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
            if m:
                ram_used_mb.append(int(m.group(1)))
                ram_total_mb.append(int(m.group(2)))

            # SWAP 0/3810MB
            m = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
            if m:
                swap_used_mb.append(int(m.group(1)))
                swap_total_mb.append(int(m.group(2)))

            # GR3D_FREQ 99%
            m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            if m:
                gpu_util_pct.append(int(m.group(1)))

            # CPU [5%@729,3%@729,off,off,4%@729,2%@729]
            m = re.search(r"CPU\s+\[([^\]]+)\]", line)
            if m:
                parts = m.group(1).split(",")
                for p in parts:
                    p = p.strip()
                    if p == "off":
                        continue
                    cm = re.search(r"(\d+)%@", p)
                    if cm:
                        cpu_utils.append(int(cm.group(1)))

            # VDD_IN 7587mW/7520mW
            # First number is current power, second is average since boot/window depending on tegrastats version.
            m = re.search(r"VDD_IN\s+(\d+)mW", line)
            if m:
                vdd_in_mw.append(int(m.group(1)))

            # Any temperature like CPU@45.5C, GPU@46C, tj@47.5C
            for tm in re.finditer(r"@([\d.]+)C", line):
                temps_c.append(float(tm.group(1)))

    print("=" * 80)
    print("TEGRSTATS SUMMARY")
    print("File:", args.log)
    print("Samples:", len(ram_used_mb))

    print("\nMEMORY")
    print_metric("Avg RAM used", mean_or_none(ram_used_mb), " MB")
    print_metric("Peak RAM used", max_or_none(ram_used_mb), " MB")
    if ram_total_mb:
        print_metric("RAM total", max_or_none(ram_total_mb), " MB")

    print_metric("Avg SWAP used", mean_or_none(swap_used_mb), " MB")
    print_metric("Peak SWAP used", max_or_none(swap_used_mb), " MB")
    if swap_total_mb:
        print_metric("SWAP total", max_or_none(swap_total_mb), " MB")

    print("\nUTILIZATION")
    print_metric("Avg GPU util", mean_or_none(gpu_util_pct), " %")
    print_metric("Peak GPU util", max_or_none(gpu_util_pct), " %")
    print_metric("Avg CPU util across active cores", mean_or_none(cpu_utils), " %")
    print_metric("Peak CPU util across active cores", max_or_none(cpu_utils), " %")

    print("\nPOWER")
    avg_power_w = mean_or_none(vdd_in_mw)
    peak_power_w = max_or_none(vdd_in_mw)

    if avg_power_w is not None:
        avg_power_w /= 1000
    if peak_power_w is not None:
        peak_power_w /= 1000

    print_metric("Avg VDD_IN", avg_power_w, " W")
    print_metric("Peak VDD_IN", peak_power_w, " W")

    print("\nTEMPERATURE")
    print_metric("Avg temp", mean_or_none(temps_c), " C")
    print_metric("Peak temp", max_or_none(temps_c), " C")


if __name__ == "__main__":
    main()
