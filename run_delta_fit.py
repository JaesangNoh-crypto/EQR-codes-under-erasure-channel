#!/usr/bin/env python3
"""
Find delta such that P_fail <= exp(-delta * sqrt(n))
for extended QR codes on BEC with eps=0.4, bit 0 always erased.

Primes p ≡ 7 (mod 8), code length n = p+1 in [192, 648].

Usage:
    python3 run_delta_fit.py [--frames 50000] [--threads 8] [--skip-gen]

Requirements:
    - gen_h.py in current directory
    - eqr_bec_sim compiled in current directory
"""

import subprocess, sys, os, math, argparse, time, csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def find_primes(n_min=190, n_max=1000, count=10):
    """Pick ~count primes p ≡ 7 (mod 8) evenly spread in [n_min, n_max]."""
    all_p = [p for p in range(n_min - 1, n_max)
             if is_prime(p) and p % 8 == 7]
    if len(all_p) <= count:
        return all_p
    if count == 1:
        return [all_p[len(all_p) // 2]]
    # Evenly spaced selection
    step = (len(all_p) - 1) / (count - 1)
    selected = [all_p[round(i * step)] for i in range(count)]
    return selected

H_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "H_matrices")

def h_filename(p):
    """Find the actual H matrix file in H_matrices/ directory."""
    n, k = p + 1, (p + 1) // 2
    os.makedirs(H_DIR, exist_ok=True)
    candidates = [
        os.path.join(H_DIR, f"H_{n}_{k}.bin"),
        os.path.join(H_DIR, f"H_{n}.bin"),
        os.path.join(H_DIR, f"H{n}_{k}.bin"),
        os.path.join(H_DIR, f"H{n}.bin"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]  # default for generation

def generate_h(p):
    n, k = p + 1, (p + 1) // 2
    existing = h_filename(p)
    if os.path.exists(existing):
        print(f"    [{existing}] exists, skip")
        return True
    print(f"    Generating H for p={p} ({n},{k}) ...", end=" ", flush=True)
    t0 = time.time()
    ret = subprocess.run(
        [sys.executable, "gen_h.py", str(p)],
        capture_output=True, text=True
    )
    dt = time.time() - t0
    if ret.returncode != 0:
        print(f"FAILED ({dt:.1f}s)")
        for line in (ret.stderr or ret.stdout).strip().split("\n")[-5:]:
            print(f"      {line}")
        return False
    print(f"OK ({dt:.1f}s)")
    return os.path.exists(h_filename(p))

def run_sim(p, nframes, nthr, eps):
    """Run C simulator at given eps. Returns P_amb or None."""
    fname = h_filename(p)
    if not os.path.exists(fname):
        print(f"    FILE NOT FOUND: tried {fname}")
        return None

    eps_str = f"{eps:.6f}"
    cmd = ["./eqr_bec_sim2", "-b", fname, "-f", str(nframes),
           "-s", eps_str, "-e", eps_str, "-d", "0.01", "-t", str(nthr)]
    ret = subprocess.run(cmd, capture_output=True, text=True)

    if ret.returncode != 0:
        print(f"    SIM ERROR p={p}: {ret.stderr[:200]}")
        return None

    # Parse output: find the data line (skip headers)
    for line in ret.stdout.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("ep") or line.startswith("--") or line.startswith("Code") or line.startswith("Done"):
            continue
        # Match data lines: first field should be a number close to eps
        parts = line.split()
        if len(parts) >= 2:
            try:
                val = float(parts[0])
                if abs(val - eps) < 0.01:
                    return float(parts[1])
            except ValueError:
                continue

    print(f"    PARSE FAIL p={p}, stdout:")
    for line in ret.stdout.strip().split("\n")[-5:]:
        print(f"      [{line}]")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=50000)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--nmin", type=int, default=190, help="min code length n")
    parser.add_argument("--nmax", type=int, default=1000, help="max code length n")
    parser.add_argument("--count", type=int, default=10, help="number of primes to sample")
    parser.add_argument("--eps", type=float, default=0.4, help="erasure probability (default 0.4)")
    args = parser.parse_args()

    nthr = args.threads if args.threads > 0 else os.cpu_count() or 4

    if not os.path.exists("./eqr_bec_sim2"):
        print("ERROR: ./eqr_bec_sim2 not found. Compile first:")
        print("  gcc -O3 -march=native -fopenmp -o eqr_bec_sim2 eqr_bec_sim2.c -lm")
        return 1

    primes = find_primes(args.nmin, args.nmax, args.count)
    print(f"Target: {len(primes)} primes, p ≡ 7 (mod 8), n ∈ [{args.nmin}, {args.nmax}]")
    print(f"  p = {primes}")
    print(f"  n = {[p+1 for p in primes]}")
    print(f"  ε = {args.eps}, frames = {args.frames}, threads = {nthr}\n")

    # Step 1: Generate H matrices
    if not args.skip_gen:
        print("STEP 1: Generate H matrices")
        print("-" * 50)
        for p in primes:
            generate_h(p)
        print()

    # Step 2: Run simulations
    print(f"STEP 2: Simulate at ε = {args.eps}")
    print("-" * 50)
    print(f"  {'p':>6}  {'n':>6}  {'P_amb':>14}  {'time':>8}")
    print(f"  {'---':>6}  {'---':>6}  {'---':>14}  {'---':>8}")

    results = []  # (p, n, P_amb)
    stopped_early = False
    for p in primes:
        n = p + 1
        if stopped_early:
            results.append((p, n, 0.0))
            print(f"  {p:>6}  {n:>6}  {'0 (skipped)':>14}  {'--':>8}")
            continue
        t0 = time.time()
        pamb = run_sim(p, args.frames, nthr, args.eps)
        dt = time.time() - t0
        results.append((p, n, pamb))
        ps = f"{pamb:.6e}" if pamb is not None else "N/A"
        print(f"  {p:>6}  {n:>6}  {ps:>14}  {dt:>7.1f}s")
        if pamb is not None and pamb == 0.0:
            print(f"  >>> P_amb = 0 at n={n}, skipping remaining (larger n will also be 0)")
            stopped_early = True

    # Save raw CSV
    with open("delta_raw.csv", "w") as f:
        f.write("p,n,k,P_amb\n")
        for p, n, pamb in results:
            k = n // 2
            ps = f"{pamb:.12e}" if pamb is not None else ""
            f.write(f"{p},{n},{k},{ps}\n")
    print(f"\nRaw data saved to delta_raw.csv\n")

    # Step 3: Fit delta
    print("STEP 3: Fit δ  (model: P_amb ≤ exp(-δ·√n))")
    print("-" * 50)

    points = []
    for p, n, pamb in results:
        if pamb is not None and pamb > 0:
            sqn = math.sqrt(n)
            secbits = -math.log2(2 * pamb)
            points.append((n, sqn, secbits, pamb))

    if len(points) == 0:
        print("  No data points with P_amb > 0.")
        print("  Try increasing --frames.")
        return 1

    print(f"\n  {'n':>6}  {'√n':>8}  {'P_amb':>14}  {'sec bits':>12}  {'bits/√n':>12}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*14}  {'-'*12}  {'-'*12}")
    for n, sqn, sb, pf in points:
        print(f"  {n:>6}  {sqn:>8.3f}  {pf:>14.6e}  {sb:>12.4f}  {sb/sqn:>12.6f}")

    # Save fit results
    with open("delta_fit.csv", "w") as f:
        f.write("n,sqrt_n,P_amb,sec_bits,sec_bits_over_sqrt_n\n")
        for n, sqn, sb, pf in points:
            f.write(f"{n},{sqn:.6f},{pf:.12e},{sb:.10f},{sb/sqn:.10f}\n")
    print(f"\n  Fit data saved to delta_fit.csv")

    if len(points) < 2:
        print(f"\n  Only {len(points)} data point(s) — skipping fit and plot.")
        return 0

    # δ₂_min: guaranteed bound for all tested n (in bits)
    delta2_min = min(sb / sqn for _, sqn, sb, _ in points)

    # Linear regression: -log₂(P) = δ₂·√n + c
    sx  = sum(sq for _, sq, _, _ in points)
    sy  = sum(sb for _, _, sb, _ in points)
    sxx = sum(sq*sq for _, sq, _, _ in points)
    sxy = sum(sq*sb for _, sq, sb, _ in points)
    nn  = len(points)

    denom = nn * sxx - sx * sx
    if abs(denom) > 1e-15:
        delta2_fit = (nn * sxy - sx * sy) / denom
        intercept = (sy - delta2_fit * sx) / nn
    else:
        delta2_fit, intercept = sxy / sxx, 0

    # R² for delta2_fit
    ss_res = sum((sb - delta2_fit * sq - intercept)**2 for _, sq, sb, _ in points)
    ss_tot = sum((sb - sy/nn)**2 for _, _, sb, _ in points)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  Results ({len(points)} data points):")
    print(f"  ─────────────────────────────────────────")
    print(f"  δ₂_min = {delta2_min:.6f}  (guaranteed: P ≤ 2^(-{delta2_min:.4f}·√n) ∀n)")
    print(f"  δ₂_fit = {delta2_fit:.6f} (c = {intercept:.4f}, R² = {r2:.6f})")
    print()

    # Target security levels
    print(f"  Target security (using δ₂_min = {delta2_min:.4f}):")
    for target in [40, 60, 80, 128]:
        n_req = (target / delta2_min) ** 2
        print(f"    {target:>3}-bit security:  n ≥ {n_req:.0f}  (√n ≥ {math.sqrt(n_req):.1f})")
    print()

    # ── Plot ──
    print(f"  Generating plot ...")

    sqns = np.array([d[1] for d in points])
    sbs  = np.array([d[2] for d in points])

    sq_max = max(max(sqns) * 1.15, 62 / delta2_min * 1.05)
    sq_range = np.linspace(0, sq_max, 200)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(sqns, sbs, s=60, c="#2563eb", zorder=5,
               label="Simulated", edgecolors="white", linewidth=0.5)
    ax.plot(sq_range, delta2_min * sq_range,
            "--", color="#ef4444", linewidth=2,
            label=f"δ₂_min = {delta2_min:.4f}")
    ax.plot(sq_range, delta2_fit * sq_range + intercept,
            "-.", color="#f59e0b", linewidth=2,
            label=f"δ₂_fit = {delta2_fit:.4f}, c={intercept:.2f} (R²={r2:.3f})")
    for ni, sq, sb, _ in points:
        ax.annotate(f"n={ni}", (sq, sb), fontsize=7,
                    textcoords="offset points", xytext=(5, 5), color="#64748b")
    for sec, col in [(20,"#cbd5e1"),(40,"#94a3b8"),(60,"#64748b")]:
        ax.axhline(sec, color=col, linewidth=0.8, linestyle=":", alpha=0.5)
        ax.text(sq_max*0.98, sec, f"{sec}-bit", fontsize=8, color=col, va="bottom", ha="right")
    ax.set_xlabel("√n", fontsize=12)
    ax.set_ylabel("Security bits  (−log₂ P_fail)", fontsize=12)
    ax.set_title(f"Security bits over EQR codes (ε = {args.eps})", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(0, 62)

    plt.tight_layout()
    plt.savefig("delta_fit.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved to delta_fit.png")

    return 0

if __name__ == "__main__":
    sys.exit(main())