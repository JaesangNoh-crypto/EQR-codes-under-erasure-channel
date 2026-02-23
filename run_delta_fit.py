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

def run_sim(p, nframes, nthr):
    """Run C simulator at eps=0.4 only. Returns P_amb or None."""
    fname = h_filename(p)
    if not os.path.exists(fname):
        print(f"    FILE NOT FOUND: tried {fname}")
        return None

    cmd = ["./eqr_bec_sim2", "-b", fname, "-f", str(nframes),
           "-s", "0.4", "-e", "0.4", "-d", "0.01", "-t", str(nthr)]
    ret = subprocess.run(cmd, capture_output=True, text=True)

    if ret.returncode != 0:
        print(f"    SIM ERROR p={p}: {ret.stderr[:200]}")
        return None

    # Parse output: handle both CSV and space-separated formats
    for line in ret.stdout.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("ep") or line.startswith("--") or line.startswith("Code") or line.startswith("Done"):
            continue
        if line.startswith("0.4") or line.startswith("0,4"):
            # Try comma-separated first, then space-separated
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass

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
    print(f"  ε = 0.4 fixed, frames = {args.frames}, threads = {nthr}\n")

    # Step 1: Generate H matrices
    if not args.skip_gen:
        print("STEP 1: Generate H matrices")
        print("-" * 50)
        for p in primes:
            generate_h(p)
        print()

    # Step 2: Run simulations
    print("STEP 2: Simulate at ε = 0.4")
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
        pamb = run_sim(p, args.frames, nthr)
        dt = time.time() - t0
        results.append((p, n, pamb))
        ps = f"{pamb:.8f}" if pamb is not None else "N/A"
        print(f"  {p:>6}  {n:>6}  {ps:>14}  {dt:>7.1f}s")
        if pamb is not None and pamb == 0.0:
            print(f"  >>> P_amb = 0 at n={n}, skipping remaining (larger n will also be 0)")
            stopped_early = True

    # Save raw CSV
    with open("delta_raw.csv", "w") as f:
        f.write("p,n,k,P_amb\n")
        for p, n, pamb in results:
            k = n // 2
            ps = f"{pamb:.10f}" if pamb is not None else ""
            f.write(f"{p},{n},{k},{ps}\n")
    print(f"\nRaw data saved to delta_raw.csv\n")

    # Step 3: Fit delta
    print("STEP 3: Fit δ  (model: P_amb ≤ exp(-δ·√n))")
    print("-" * 50)

    points = []
    for p, n, pamb in results:
        if pamb is not None and pamb > 0:
            sqn = math.sqrt(n)
            nlp = -math.log(pamb)
            points.append((n, sqn, nlp, pamb))

    if len(points) < 2:
        print("  Not enough data points with P_amb > 0.")
        print("  Try increasing --frames.")
        return 1

    print(f"\n  {'n':>6}  {'√n':>8}  {'P_amb':>14}  {'-ln(P)':>10}  {'-ln(P)/√n':>12}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*14}  {'-'*10}  {'-'*12}")
    for n, sqn, nlp, pf in points:
        print(f"  {n:>6}  {sqn:>8.3f}  {pf:>14.8f}  {nlp:>10.4f}  {nlp/sqn:>12.6f}")

    # δ_min: guaranteed bound for all tested n
    delta_min = min(nlp / sqn for _, sqn, nlp, _ in points)

    # Linear regression: -ln(P) = δ·√n + c
    sx  = sum(sq for _, sq, _, _ in points)
    sy  = sum(nl for _, _, nl, _ in points)
    sxx = sum(sq*sq for _, sq, _, _ in points)
    sxy = sum(sq*nl for _, sq, nl, _ in points)
    nn  = len(points)

    denom = nn * sxx - sx * sx
    if abs(denom) > 1e-15:
        delta_fit = (nn * sxy - sx * sy) / denom
        intercept = (sy - delta_fit * sx) / nn
    else:
        delta_fit, intercept = sxy / sxx, 0

    # R² for delta_fit
    ss_res = sum((nl - delta_fit * sq - intercept)**2 for _, sq, nl, _ in points)
    ss_tot = sum((nl - sy/nn)**2 for _, _, nl, _ in points)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  Results ({len(points)} data points):")
    print(f"  ─────────────────────────────────────────")
    print(f"  δ_min = {delta_min:.6f}  (guaranteed: P ≤ exp(-{delta_min:.4f}·√n) ∀n)")
    print(f"  δ_fit = {delta_fit:.6f}  (c = {intercept:.4f}, R² = {r2:.6f})")
    print()
    print(f"  Interpretation:")
    print(f"    For ε=0.4, self-dual EQR codes (p≡7 mod 8):")
    print(f"    P(bit 0 ambiguous) ≤ exp(-{delta_min:.4f} · √n)")
    print()

    # Save fit results
    with open("delta_fit.csv", "w") as f:
        f.write("n,sqrt_n,P_amb,neg_ln_P,neg_ln_P_over_sqrt_n\n")
        for n, sqn, nlp, pf in points:
            f.write(f"{n},{sqn:.6f},{pf:.10f},{nlp:.6f},{nlp/sqn:.6f}\n")
    print(f"  Fit data saved to delta_fit.csv")

    # ── Plot ──
    print(f"\n  Generating plots ...")

    ns   = np.array([d[0] for d in points])
    sqns = np.array([d[1] for d in points])
    nlps = np.array([d[2] for d in points])
    pambs= np.array([d[3] for d in points])

    sq_range = np.linspace(0, max(sqns) * 1.15, 200)
    n_range  = sq_range ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Self-Dual EQR Codes on BEC (ε=0.4) — Bit-MAP P(ambiguous)",
        fontsize=14, fontweight="bold", y=0.98)

    # Left: -ln(P) vs √n
    ax1.scatter(sqns, nlps, s=60, c="#2563eb", zorder=5,
                label="Simulated", edgecolors="white", linewidth=0.5)
    ax1.plot(sq_range, delta_min * sq_range,
             "--", color="#ef4444", linewidth=2,
             label=f"δ_min = {delta_min:.4f}")
    ax1.plot(sq_range, delta_fit * sq_range + intercept,
             "-.", color="#f59e0b", linewidth=2,
             label=f"δ_fit = {delta_fit:.4f}, c={intercept:.2f} (R²={r2:.3f})")
    for ni, sq, nl, _ in points:
        ax1.annotate(f"n={ni}", (sq, nl), fontsize=7,
                     textcoords="offset points", xytext=(5, 5), color="#64748b")
    ax1.set_xlabel("√n", fontsize=12)
    ax1.set_ylabel("−ln P(amb)", fontsize=12)
    ax1.set_title("−ln(P) vs √n", fontsize=12)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0); ax1.set_ylim(bottom=0)

    # Right: P(amb) vs n (log scale)
    ax2.semilogy(ns, pambs, "o", color="#2563eb", markersize=7, zorder=5,
                 label="Simulated", markeredgecolor="white", markeredgewidth=0.5)
    ax2.semilogy(n_range, np.exp(-delta_min * sq_range),
                 "--", color="#ef4444", linewidth=2,
                 label=f"exp(−{delta_min:.4f}·√n)")
    ax2.semilogy(n_range, np.exp(-delta_fit * sq_range - intercept),
                 "-.", color="#f59e0b", linewidth=2,
                 label=f"exp(−{delta_fit:.4f}·√n{intercept:+.2f})")
    for sec, col in [(40,"#94a3b8"),(60,"#64748b"),(80,"#475569")]:
        ax2.axhline(2**(-sec), color=col, linewidth=0.8, linestyle=":", alpha=0.5)
        ax2.text(max(ns)*1.02, 2**(-sec), f"2⁻{sec}", fontsize=8, color=col, va="center")
    ax2.set_xlabel("Code length n", fontsize=12)
    ax2.set_ylabel("P(ambiguous)", fontsize=12)
    ax2.set_title("P(amb) vs n", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim(left=0)

    plt.tight_layout()
    plt.show()

    return 0

if __name__ == "__main__":
    sys.exit(main())