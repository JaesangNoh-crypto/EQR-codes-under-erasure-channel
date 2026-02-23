#!/usr/bin/env python3
"""
Generate parity-check matrix for self-dual Extended QR codes.
Pure Python, no SageMath. Works for ANY prime p (with p ≡ ±1 mod 8).

Usage:
    python3 gen_h.py <prime> [prime2 ...]
    python3 gen_h.py -t <prime>           # also export .txt

Examples:
    python3 gen_h.py 23
    python3 gen_h.py 7 17 23 31 47 71 151 191
"""

import sys, struct, os, time, random

# ================================================================
#  GF(2) polynomial arithmetic — polynomials as Python ints
#  bit i of integer = coefficient of x^i
# ================================================================

def poly_deg(a):
    return a.bit_length() - 1 if a else -1

def poly_mod(a, b):
    db = poly_deg(b)
    if db < 0: raise ZeroDivisionError
    da = poly_deg(a)
    while da >= db:
        a ^= b << (da - db)
        da = poly_deg(a)
    return a

def poly_mul(a, b):
    r = 0
    while b:
        if b & 1: r ^= a
        b >>= 1; a <<= 1
    return r

def poly_mulmod(a, b, f):
    return poly_mod(poly_mul(a, b), f)

def poly_powmod(base, exp, f):
    r = 1
    base = poly_mod(base, f)
    while exp > 0:
        if exp & 1: r = poly_mulmod(r, base, f)
        exp >>= 1
        if exp > 0: base = poly_mulmod(base, base, f)
    return r

def poly_gcd(a, b):
    while b:
        a, b = b, poly_mod(a, b)
    return a

# ================================================================
#  Number theory helpers
# ================================================================

def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def prime_factors(n):
    facs = set()
    d = 2
    while d * d <= n:
        while n % d == 0: facs.add(d); n //= d
        d += 1
    if n > 1: facs.add(n)
    return facs

def ord_p(p):
    v, k = 2 % p, 1
    while v != 1:
        v = (v * 2) % p; k += 1
    return k

# ================================================================
#  Find irreducible polynomial of degree m over GF(2)
#
#  Uses Rabin irreducibility test:
#    1. x^(2^m) ≡ x (mod f)   — all roots lie in GF(2^m)
#    2. For each prime q | m:
#       gcd(x^(2^(m/q)) - x, f) = 1  — no root in smaller subfield
# ================================================================

def check_irreducible(f, m):
    # Test 1: x^(2^m) ≡ x mod f
    h = 2  # x
    for _ in range(m):
        h = poly_mulmod(h, h, f)
    if h != 2:
        return False
    # Test 2: for each prime q dividing m, gcd(x^(2^(m/q)) - x, f) == 1
    for q in prime_factors(m):
        d = m // q
        h2 = 2
        for _ in range(d):
            h2 = poly_mulmod(h2, h2, f)
        # h2 = x^(2^d) mod f; h2 ^ 2 = x^(2^d) + x  (in GF(2))
        g = poly_gcd(h2 ^ 2, f)
        # BUG FIX: old code had "g != 1 and poly_deg(g) < m" which
        # missed the case g == f (reducible into equal-degree factors).
        # Correct condition: any nontrivial GCD means f is reducible.
        if g != 1:
            return False
    return True

def find_irreducible(m):
    """Find an irreducible polynomial of degree m over GF(2)."""
    base = 1 << m
    # Try trinomials x^m + x^k + 1 first (fastest to work with)
    for k in range(1, m):
        f = base | (1 << k) | 1
        if check_irreducible(f, m):
            return f
    # Try pentanomials x^m + x^a + x^b + x^c + 1
    for a in range(3, m):
        for b in range(2, a):
            for c in range(1, b):
                f = base | (1 << a) | (1 << b) | (1 << c) | 1
                if check_irreducible(f, m):
                    return f
    # Try random odd-weight polynomials
    rng = random.Random(99999)
    for _ in range(100000):
        f = base | 1
        for _ in range(rng.randint(1, m // 2)):
            f |= 1 << rng.randint(1, m - 1)
        if check_irreducible(f, m):
            return f
    raise RuntimeError(f"No irreducible poly found for degree {m}")

# ================================================================
#  Find p-th root of unity in GF(2^m) = GF(2)[x] / irr(x)
# ================================================================

def find_pth_root(p, m, irr):
    """Find element of order exactly p in GF(2^m)."""
    q = ((1 << m) - 1) // p
    rng = random.Random(12345)
    for _ in range(500):
        a = rng.randint(2, (1 << m) - 1)
        alpha = poly_powmod(a, q, irr)
        if alpha <= 1:
            continue
        # Verify: alpha^p must equal 1
        if poly_powmod(alpha, p, irr) != 1:
            continue
        return alpha
    raise RuntimeError(f"Failed to find {p}-th root of unity in GF(2^{m})")

# ================================================================
#  Compute generator polynomial g(x) of QR code
# ================================================================

def compute_gen_poly(p, verbose=True):
    m = ord_p(p)
    if verbose:
        print(f"    ord_{p}(2) = {m}, working in GF(2^{m})")

    # Step 1: irreducible polynomial
    if verbose: print(f"    Finding irreducible poly of degree {m} ...", end=" ", flush=True)
    t0 = time.time()
    irr = find_irreducible(m)
    if verbose: print(f"found 0x{irr:X} ({time.time()-t0:.2f}s)")

    # Step 2: p-th root of unity
    if verbose: print(f"    Finding {p}-th root of unity ...", end=" ", flush=True)
    alpha = find_pth_root(p, m, irr)
    if verbose: print(f"alpha = 0x{alpha:X}")

    # Verify Frobenius: alpha^2 is also a root
    alpha2 = poly_mulmod(alpha, alpha, irr)
    alpha_2 = poly_powmod(alpha, 2, irr)
    assert alpha2 == alpha_2, "Frobenius check failed"

    # Step 3: QR set
    qr_set = set()
    for i in range(1, p):
        qr_set.add((i * i) % p)
    if verbose:
        print(f"    |QR| = {len(qr_set)}, |QNR| = {(p-1)//2 - len(qr_set) + (p-1)//2}")

    # Step 4: build g(x) = product of minimal polys
    visited = set()
    g = 1  # polynomial as int
    n_cosets = 0

    for r in range(1, p):
        if r not in qr_set or r in visited:
            continue

        # Cyclotomic coset of r under ×2 mod p
        coset = []
        c = r
        while c not in visited:
            visited.add(c)
            coset.append(c)
            c = (c * 2) % p

        # Compute roots in GF(2^m)
        roots = [poly_powmod(alpha, ci, irr) for ci in coset]

        # Verify Frobenius closure: root^2 = next root (cyclically)
        for idx in range(len(roots)):
            sq = poly_mulmod(roots[idx], roots[idx], irr)
            nxt = poly_powmod(alpha, (coset[idx] * 2) % p, irr)
            assert sq == nxt, f"Frobenius closure failed at coset element {coset[idx]}"

        # Minimal polynomial = product of (x + root_i) over GF(2^m)
        mp = [roots[0], 1]
        for i in range(1, len(roots)):
            rt = roots[i]
            nmp = [0] * (len(mp) + 1)
            for j in range(len(mp)):
                nmp[j] ^= poly_mulmod(mp[j], rt, irr)
                nmp[j + 1] ^= mp[j]
            mp = nmp

        # Verify all coefficients are in GF(2)
        min_poly = 0
        for i, coeff in enumerate(mp):
            if coeff == 0:
                pass
            elif coeff == 1:
                min_poly |= (1 << i)
            else:
                raise RuntimeError(
                    f"Minimal poly coeff not in GF(2): degree {i}, "
                    f"val=0x{coeff:x}, coset of {r}, coset_size={len(coset)}\n"
                    f"  irr=0x{irr:X}, alpha=0x{alpha:X}\n"
                    f"  This likely means the irreducible polynomial is wrong.")

        g = poly_mul(g, min_poly)
        n_cosets += 1
        if verbose:
            print(f"    Coset of {r:>4d} (size {len(coset):>3d}), "
                  f"min_poly deg={poly_deg(min_poly)}, g deg={poly_deg(g)}")

    if verbose:
        print(f"    Total: {n_cosets} QR cosets, g(x) degree = {poly_deg(g)}")
    return g

# ================================================================
#  Build extended QR parity-check matrix
# ================================================================

def build_extended_h(p, verbose=True):
    g_poly = compute_gen_poly(p, verbose)
    n = p + 1
    k = (p + 1) // 2
    deg = poly_deg(g_poly)
    assert deg == (p - 1) // 2, f"g(x) degree {deg} != expected {(p-1)//2}"

    # Convert to coefficient list
    g = []
    tmp = g_poly
    for i in range(deg + 1):
        g.append(tmp & 1)
        tmp >>= 1

    # Systematic encoding: row i of G
    Grows = []
    for i in range(k):
        rem = [0] * p
        rem[i + deg] = 1
        for j in range(p - 1, deg - 1, -1):
            if rem[j]:
                for l in range(deg + 1):
                    rem[j - deg + l] ^= g[l]
        row = 0
        for j in range(deg):
            if rem[j]:
                row |= (1 << j)
        row |= (1 << (i + deg))
        Grows.append(row)

    # H_ext: nk = k rows, n = p+1 columns
    nk = k
    all_ones_bit = 1 << (nk - 1)

    hcols = [0] * n
    for j in range(p):
        if j < deg:
            col = (1 << j)
        else:
            col = 0
            for i in range(deg):
                if (Grows[j - deg] >> i) & 1:
                    col |= (1 << i)
        col |= all_ones_bit
        hcols[j] = col

    # Parity column
    par_col = 0
    for i in range(deg):
        w = sum(1 for j in range(p) if (hcols[j] >> i) & 1)
        if w & 1:
            par_col |= (1 << i)
    par_col |= all_ones_bit
    hcols[p] = par_col

    return hcols, n, k, nk

# ================================================================
#  Quick verification: H · G^T = 0
# ================================================================

def quick_verify(p, hcols, n, k, nk):
    g_poly = compute_gen_poly(p, verbose=False)
    deg = poly_deg(g_poly)
    g = []
    tmp = g_poly
    for i in range(deg + 1):
        g.append(tmp & 1)
        tmp >>= 1

    rng = random.Random(999)
    n_checks = min(500, 1 << k)
    for trial in range(n_checks):
        msg = rng.randint(0, (1 << k) - 1) if trial > 0 else 1
        poly = [0] * p
        for i in range(k):
            if (msg >> i) & 1:
                poly[i + deg] = 1
        rem = list(poly)
        for j in range(p - 1, deg - 1, -1):
            if rem[j]:
                for l in range(deg + 1):
                    rem[j - deg + l] ^= g[l]

        cw = 0
        par = 0
        for j in range(p):
            bit = rem[j] if j < deg else poly[j]
            if bit:
                cw |= (1 << j)
                par ^= 1
        if par:
            cw |= (1 << p)

        for i in range(nk):
            dot = 0
            for j in range(n):
                if ((cw >> j) & 1) and ((hcols[j] >> i) & 1):
                    dot ^= 1
            if dot:
                return False
    return True

# ================================================================
#  Export
# ================================================================

def export_code(p, text_mode=False):
    if not is_prime(p):
        print(f"  [SKIP] {p} is not prime"); return None
    pm8 = p % 8
    if pm8 not in (1, 7):
        print(f"  [WARN] p={p} not ≡ ±1 (mod 8), code may not be self-dual")

    n, k = p + 1, (p + 1) // 2
    print(f"  Building ({n}, {k}) extended QR code for p={p} ...")
    t0 = time.time()

    hcols, n, k, nk = build_extended_h(p, verbose=True)
    nw = (nk + 63) // 64
    dt = time.time() - t0
    print(f"  Construction time: {dt:.2f}s")

    print(f"  Verifying H·G^T = 0 ...", end=" ", flush=True)
    if quick_verify(p, hcols, n, k, nk):
        print("OK ✓")
    else:
        print("FAILED ✗"); return None

    # Binary export
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "H_matrices")
    os.makedirs(outdir, exist_ok=True)
    fname_bin = os.path.join(outdir, f"H_{n}.bin")
    with open(fname_bin, "wb") as f:
        f.write(struct.pack("<III", nk, n, nw))
        for j in range(n):
            val = hcols[j]
            for w in range(nw):
                word = (val >> (64 * w)) & 0xFFFFFFFFFFFFFFFF
                f.write(struct.pack("<Q", word))
    sz = os.path.getsize(fname_bin)
    print(f"  Saved {fname_bin} ({sz} bytes, {nw} words/col)")

    if text_mode:
        fname_txt = os.path.join(outdir, f"H_{n}.txt")
        with open(fname_txt, "w") as f:
            f.write(f"{nk} {n}\n")
            for i in range(nk):
                f.write(" ".join(
                    str((hcols[j] >> i) & 1) for j in range(n)) + "\n")
        print(f"  Saved {fname_txt}")

    return n, k

# ================================================================
#  Main
# ================================================================

def main():
    args = sys.argv[1:]
    text_mode = "-t" in args
    if text_mode: args.remove("-t")

    if not args:
        print("Usage: python3 gen_h.py [-t] <prime> [prime2 ...]")
        print("  -t  Also export human-readable .txt file")
        print()
        print("Self-dual EQR primes (p ≡ ±1 mod 8):")
        primes = [p for p in range(7, 300) if is_prime(p) and p % 8 in (1, 7)]
        print(" ", ", ".join(str(p) for p in primes))
        sys.exit(1)

    print(f"Generating H matrices for {len(args)} prime(s):\n")
    results = []
    for a in args:
        p = int(a)
        r = export_code(p, text_mode)
        if r: results.append(r)
        print()

    if results:
        print("Summary:")
        for n, k in results:
            print(f"  ({n}, {k}), R = {k/n:.4f} -> H_{n}_{k}.bin")

if __name__ == "__main__":
    main()