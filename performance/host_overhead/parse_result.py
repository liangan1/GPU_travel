#!/usr/bin/env python3
import csv
from collections import defaultdict
import numpy as np

def parse_host_time(input_file, output_file):
    times = defaultdict(list)

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ', : start' in line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            op_name = parts[0].strip()
            time_str = parts[1].strip().split()[0]
            try:
                t = int(time_str)
                times[op_name].append(t)
            except ValueError:
                continue

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['op_name', 'min_time', 'max_time', 'Q25', 'Q75', 'avg_time'])
        for op, vals in times.items():
            if not vals:
                continue
            arr = np.array(vals)
            mn = int(arr.min())
            mx = int(arr.max())
            # 计算 25%、75% 分位数
            q25, q75 = np.percentile(arr, [25, 75])  # :contentReference[oaicite:1]{index=1}
            # 过滤仅保留中间 50%
            core = arr[(arr >= q25) & (arr <= q75)]
            avg = int(core.mean()) if len(core) > 0 else ''
            writer.writerow([op, mn, mx, int(q25), int(q75), avg])

def compute_overall_avg(input_file):
    times = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ', : start' in line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            time_str = parts[1].strip().split()[0]
            try:
                t = int(time_str)
                times.append(t)
            except ValueError:
                continue

    if not times:
        print("No timing data found.")
        return

    arr = np.array(times)
    q25, q75 = np.percentile(arr, [25, 75])
    core = arr[(arr >= q25) & (arr <= q75)]
    avg = core.mean() if len(core) > 0 else float('nan')
    
    print(f"Total samples: {len(arr)}")
    print(f"Q25 = {q25}, Q75 = {q75}, samples in core = {len(core)}")
    print(f"Overall average time (within [25%,75%]) = {avg:.2f} ns")


if __name__ == '__main__':
    compute_overall_avg('host_time_log.csv')
    parse_host_time('host_time_log.csv', 'host_time_summary.csv')
    print("Completed: output written to host_time_summary.csv")

