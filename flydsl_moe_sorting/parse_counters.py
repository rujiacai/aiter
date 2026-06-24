#!/usr/bin/env python3
# Parse rocprofv3 counter CSVs for stage2 (moe_gemm2_0) and compare two runs.
#   python parse_counters.py <scrambled_dir> <ordered_dir>
import csv, glob, sys, collections


def load(d):
    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    for fn in glob.glob(f"{d}/*/*.csv"):
        with open(fn) as fh:
            rd = csv.DictReader(fh)
            if "Counter_Name" not in (rd.fieldnames or []):
                continue
            for r in rd:
                kn = (r.get("Kernel_Name") or "").split("(")[0][:24]
                agg[kn][r["Counter_Name"]] += float(r.get("Counter_Value", 0) or 0)
    return agg


a = load(sys.argv[1])
b = load(sys.argv[2])
names = set()
for agg in (a, b):
    for k in agg:
        if "moe_gemm2" in k:
            names |= set(agg[k].keys())
for k in [x for x in a if "moe_gemm2" in x]:
    for cn in sorted(names):
        v0 = a[k].get(cn, 0)
        v1 = b.get(k, {}).get(cn, 0)
        d = (v0 / v1 - 1) * 100 if v1 else 0
        print(f"  {cn:26s} scrambled={v0:,.0f}  ordered={v1:,.0f}  (scrambled {d:+.1f}%)")
