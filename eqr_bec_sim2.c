/*
 * eqr_bec_sim.c
 *
 * High-performance Monte Carlo simulation of P(ambiguous) for bit 0
 * of extended quadratic residue codes on Binary Erasure Channel.
 *
 * Optimizations:
 *   - 64-bit packed GF(2) Gaussian elimination (1 XOR = 64 rows)
 *   - Incremental echelon basis (no full matrix copy per trial)
 *   - OpenMP parallel trials with thread-local workspace
 *   - xoshiro256** PRNG per thread (no locks, no false sharing)
 *   - Geometric random variate for fast erasure pattern generation
 *   - Built-in code construction for small primes (ord_p(2) <= 23)
 *   - Binary H-matrix file input for large codes (from SageMath)
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -o eqr_bec_sim eqr_bec_sim.c -lm
 *
 * Usage:
 *   ./eqr_bec_sim -p <prime> -f <nframes> [-e start end step] [-t threads] [-o file.csv]
 *   ./eqr_bec_sim -b <H.bin> -f <nframes> [-e start end step] [-t threads] [-o file.csv]
 *
 * Examples:
 *   ./eqr_bec_sim -p 23 -f 100000
 *   ./eqr_bec_sim -p 47 -f 50000 -e 0.35 0.65 0.005 -t 8
 *   ./eqr_bec_sim -b H_152_76.bin -f 50000 -o results.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <getopt.h>

/* ================================================================
 *  xoshiro256** PRNG
 * ================================================================ */
typedef struct { uint64_t s[4]; } rng_t;

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s;
    uint64_t res = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl64(s[3], 45);
    return res;
}

static inline double rng_double(rng_t *r) {
    return (rng_next(r) >> 11) * 0x1.0p-53;
}

static void rng_seed(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        r->s[i] = z ^ (z >> 31);
    }
}

/* Geometric random variate: number of non-erased bits before next erasure */
static inline int rng_geometric(rng_t *r, double log1meps) {
    return (int)(log(rng_double(r)) / log1meps);
}

/* ================================================================
 *  H matrix: column-major, 64-bit packed rows
 * ================================================================ */
typedef struct {
    int       n;        /* code length                    */
    int       nk;       /* # check rows = n - k           */
    int       nw;       /* ceil(nk / 64)                  */
    uint64_t *data;     /* flat: col[j] = data + j * nw   */
} HMat;

static inline uint64_t *hcol(const HMat *H, int j) {
    return H->data + (size_t)j * H->nw;
}

static HMat *hmat_alloc(int n, int nk) {
    HMat *H = malloc(sizeof *H);
    H->n = n; H->nk = nk; H->nw = (nk + 63) >> 6;
    H->data = calloc((size_t)n * H->nw, sizeof(uint64_t));
    return H;
}

static void hmat_free(HMat *H) { free(H->data); free(H); }

static inline void hmat_set(HMat *H, int r, int c) {
    hcol(H, c)[r >> 6] |= 1ULL << (r & 63);
}

/* ---- Read binary H from SageMath export ---- */
static HMat *hmat_read_bin(const char *fn) {
    FILE *f = fopen(fn, "rb");
    if (!f) { perror(fn); return NULL; }
    uint32_t hdr[3];
    if (fread(hdr, 4, 3, f) != 3) { fclose(f); return NULL; }
    int nk = hdr[0], n = hdr[1], nw = hdr[2];
    if (nw != (nk + 63) / 64) {
        fprintf(stderr, "Header mismatch: nw=%d vs expected %d\n", nw, (nk+63)/64);
        fclose(f); return NULL;
    }
    HMat *H = hmat_alloc(n, nk);
    size_t total = (size_t)n * nw;
    if (fread(H->data, sizeof(uint64_t), total, f) != total) {
        fprintf(stderr, "Truncated file\n");
        hmat_free(H); fclose(f); return NULL;
    }
    fclose(f);
    return H;
}

/* ================================================================
 *  GF(2^m) arithmetic for built-in code construction
 * ================================================================ */
static const uint32_t PRIM_POLY[] = {
    0,0,0,
    0xB,0x13,0x25,0x43,0x83,0x11D,0x203,0x409,0x805,
    0x1053,0x201B,0x4443,0x8003,0x1002D,0x20009,
    0x40009,0x80027,0x100009,0x200005,0x400003,0x800021
};
#define MAX_M 23

static uint32_t gf_mul(uint32_t a, uint32_t b, int m, uint32_t pp) {
    uint32_t r = 0, hi = 1u << m;
    while (b) { if (b & 1) r ^= a; b >>= 1; a <<= 1; if (a & hi) a ^= pp; }
    return r;
}

static uint32_t gf_pow(uint32_t base, uint32_t exp, int m, uint32_t pp) {
    uint32_t r = 1;
    while (exp) { if (exp & 1) r = gf_mul(r, base, m, pp); exp >>= 1; base = gf_mul(base, base, m, pp); }
    return r;
}

static int ord_p(int p) {
    int v = 2 % p, k = 1;
    while (v != 1) { v = (v * 2) % p; k++; }
    return k;
}

static HMat *build_eqr_hmat(int p) {
    int pm8 = p % 8;
    if (pm8 != 1 && pm8 != 7)
        fprintf(stderr, "Warning: p=%d not ≡ ±1 (mod 8)\n", p);

    int m = ord_p(p);
    if (m > MAX_M) {
        fprintf(stderr, "ord_%d(2) = %d > %d, use SageMath + binary file instead.\n", p, m, MAX_M);
        return NULL;
    }
    uint32_t pp = PRIM_POLY[m];
    uint32_t q  = ((1u << m) - 1) / p;
    uint32_t alpha = gf_pow(2, q, m, pp);

    /* QR set */
    int *is_qr = calloc(p, sizeof(int));
    for (int i = 1; i < p; i++) is_qr[(int)((long long)i * i % p)] = 1;

    /* Generator polynomial */
    int glen = 1, gcap = p + 2;
    int *g = calloc(gcap, sizeof(int));
    g[0] = 1;

    int *visited = calloc(p, sizeof(int));
    uint32_t *mp  = malloc((m + 2) * sizeof *mp);
    uint32_t *nmp = malloc((m + 2) * sizeof *nmp);

    for (int r = 1; r < p; r++) {
        if (!is_qr[r] || visited[r]) continue;
        int coset[64], csz = 0;
        int c = r;
        while (!visited[c]) { visited[c] = 1; coset[csz++] = c; c = (c * 2) % p; }

        mp[0] = gf_pow(alpha, coset[0], m, pp); mp[1] = 1;
        int mplen = 2;
        for (int i = 1; i < csz; i++) {
            uint32_t rt = gf_pow(alpha, coset[i], m, pp);
            memset(nmp, 0, (mplen + 1) * sizeof *nmp);
            for (int j = 0; j < mplen; j++) {
                nmp[j] ^= gf_mul(mp[j], rt, m, pp);
                nmp[j + 1] ^= mp[j];
            }
            mplen++; memcpy(mp, nmp, mplen * sizeof *mp);
        }
        int newlen = glen + mplen - 1;
        int *ng = calloc(newlen, sizeof(int));
        for (int i = 0; i < glen; i++)
            for (int j = 0; j < mplen; j++)
                ng[i + j] ^= g[i] & (mp[j] & 1);
        free(g); g = ng; glen = newlen;
    }
    free(is_qr); free(visited); free(mp); free(nmp);

    int n = p + 1, k = (p + 1) / 2, deg = glen - 1;
    if (deg != (p - 1) / 2) {
        fprintf(stderr, "gen poly degree %d != expected %d\n", deg, (p-1)/2);
        free(g); return NULL;
    }

    /* Systematic G rows */
    uint64_t *Grows = malloc(k * sizeof(uint64_t));
    int *rem   = malloc(p * sizeof(int));
    for (int i = 0; i < k; i++) {
        memset(rem, 0, p * sizeof(int));
        rem[i + deg] = 1;
        for (int j = p - 1; j >= deg; j--)
            if (rem[j]) { for (int l = 0; l <= deg; l++) rem[j-deg+l] ^= g[l]; }
        uint64_t row = 0;
        for (int j = 0; j < deg; j++) if (rem[j]) row |= (1ULL << j);
        row |= (1ULL << (i + deg));
        Grows[i] = row;
    }
    free(rem); free(g);

    /* Build H_ext: nk = k rows, n columns */
    int nk = k;
    HMat *H = hmat_alloc(n, nk);

    for (int j = 0; j < p; j++) {
        if (j < deg) {
            hmat_set(H, j, j);
        } else {
            for (int i = 0; i < deg; i++)
                if ((Grows[j - deg] >> i) & 1) hmat_set(H, i, j);
        }
        hmat_set(H, nk - 1, j);
    }
    /* Parity column */
    for (int i = 0; i < deg; i++) {
        int w = 0;
        uint64_t *ci;
        for (int j = 0; j < p; j++) {
            ci = hcol(H, j);
            w += (ci[i >> 6] >> (i & 63)) & 1;
        }
        if (w & 1) hmat_set(H, i, p);
    }
    hmat_set(H, nk - 1, p);

    free(Grows);
    printf("Built (%d, %d) EQR code, H: %d×%d, %d words/col\n", n, k, nk, n, H->nw);
    return H;
}

/* ================================================================
 *  GF(2) Incremental Echelon Basis (per-thread workspace)
 *
 *  basis[i] has pivot at piv[i].
 *  Invariant: basis[i] has bit piv[i] set but NOT piv[j] for j != i.
 *  → O(rank * nw) per insert, O(rank * nw) for span check.
 * ================================================================ */
typedef struct {
    int       nw, nk;
    int       rank;
    int       cap;          /* allocated slots      */
    uint64_t *pool;         /* flat: basis[i] at pool + i*nw */
    int      *piv;          /* pivot bit index      */
    uint64_t *tmp;          /* scratch              */
} GEWork;

static GEWork *ge_alloc(int nk, int nw, int cap) {
    GEWork *w = malloc(sizeof *w);
    w->nw = nw; w->nk = nk; w->rank = 0; w->cap = cap;
    w->pool = malloc((size_t)cap * nw * sizeof(uint64_t));
    w->piv  = malloc(cap * sizeof(int));
    w->tmp  = malloc(nw * sizeof(uint64_t));
    return w;
}

static void ge_free(GEWork *w) {
    free(w->pool); free(w->piv); free(w->tmp); free(w);
}

static inline void ge_reset(GEWork *w) { w->rank = 0; }

static inline uint64_t *ge_basis(GEWork *w, int i) {
    return w->pool + (size_t)i * w->nw;
}

/* Insert vector v into echelon basis. Modifies nothing if linearly dependent. */
static void ge_insert(GEWork *w, const uint64_t *v) {
    int nw = w->nw;
    memcpy(w->tmp, v, nw * sizeof(uint64_t));

    /* Reduce against existing basis */
    for (int i = 0; i < w->rank; i++) {
        int pb = w->piv[i];
        if ((w->tmp[pb >> 6] >> (pb & 63)) & 1) {
            uint64_t *bi = ge_basis(w, i);
            for (int u = 0; u < nw; u++) w->tmp[u] ^= bi[u];
        }
    }

    /* Find pivot (lowest set bit) */
    int p = -1;
    for (int u = 0; u < nw && p < 0; u++)
        if (w->tmp[u]) p = (u << 6) | __builtin_ctzll(w->tmp[u]);
    if (p < 0 || p >= w->nk) return;  /* zero vector */

    /* Back-reduce: clear bit p in all existing basis vectors */
    for (int i = 0; i < w->rank; i++) {
        uint64_t *bi = ge_basis(w, i);
        if ((bi[p >> 6] >> (p & 63)) & 1)
            for (int u = 0; u < nw; u++) bi[u] ^= w->tmp[u];
    }

    /* Store */
    uint64_t *dst = ge_basis(w, w->rank);
    memcpy(dst, w->tmp, nw * sizeof(uint64_t));
    w->piv[w->rank] = p;
    w->rank++;
}

/* Check if v ∈ span(basis) */
static inline int ge_in_span(const GEWork *w, const uint64_t *v) {
    int nw = w->nw;
    uint64_t *t = w->tmp;
    memcpy(t, v, nw * sizeof(uint64_t));
    for (int i = 0; i < w->rank; i++) {
        int pb = w->piv[i];
        if ((t[pb >> 6] >> (pb & 63)) & 1) {
            const uint64_t *bi = w->pool + (size_t)i * nw;
            for (int u = 0; u < nw; u++) t[u] ^= bi[u];
        }
    }
    for (int u = 0; u < nw; u++) if (t[u]) return 0;
    return 1;
}

/* ================================================================
 *  Simulation
 * ================================================================ */
static double simulate_eps(const HMat *H, double eps, long long nframes, int nthr) {
    long long tot_amb = 0;
    const double log1meps = (eps < 1.0 - 1e-15) ? log(1.0 - eps) : -40.0;

    #pragma omp parallel num_threads(nthr) reduction(+:tot_amb)
    {
        int tid = omp_get_thread_num();
        rng_t rng;
        rng_seed(&rng, 42ULL + (uint64_t)tid * 1000003ULL +
                 (uint64_t)(eps * 1e9));
        GEWork *ws = ge_alloc(H->nk, H->nw, H->nk);

        #pragma omp for schedule(dynamic, 256)
        for (long long f = 0; f < nframes; f++) {
            ge_reset(ws);

            if (eps < 1e-15) {
                /* No erasures besides bit 0 */
            } else if (eps > 1.0 - 1e-15) {
                /* All erased */
                for (int j = 1; j < H->n; j++)
                    ge_insert(ws, hcol(H, j));
            } else {
                /* Geometric skip for sparse erasures (eps < ~0.3) */
                if (eps < 0.35) {
                    int j = 1;
                    while (j < H->n && ws->rank < ws->nk) {
                        double u = rng_double(&rng);
                        if (u < 1e-300) u = 1e-300;
                        int skip = (int)(log(u) / log1meps);
                        j += skip;
                        if (j < H->n) {
                            ge_insert(ws, hcol(H, j));
                            j++;
                        }
                    }
                } else {
                    for (int j = 1; j < H->n && ws->rank < ws->nk; j++)
                        if (rng_double(&rng) < eps)
                            ge_insert(ws, hcol(H, j));
                }
            }

            if (ge_in_span(ws, hcol(H, 0)))
                tot_amb++;
        }
        ge_free(ws);
    }
    return (double)tot_amb / nframes;
}

/* ================================================================
 *  Main
 * ================================================================ */
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s -p <prime> -f <nframes> [-e start end step] [-t threads] [-o file.csv]\n"
        "  %s -b <H.bin> -f <nframes> [-e start end step] [-t threads] [-o file.csv]\n"
        "\nOptions:\n"
        "  -p prime     Build EQR code (p ≡ ±1 mod 8, ord_p(2) ≤ 23)\n"
        "  -b file.bin  Load H from binary file (from gen_h.sage)\n"
        "  -f frames    Monte Carlo frames per epsilon point\n"
        "  -e s e d     Epsilon range: start end step (default 0 1 0.025)\n"
        "  -t threads   OpenMP threads (default: all)\n"
        "  -o file.csv  Write CSV output\n",
        prog, prog);
}

int main(int argc, char **argv) {
    int    prime   = 0;
    char  *binfile = NULL;
    char  *csvfile = NULL;
    long long nframes = 0;
    double eps_s = 0.4, eps_e = 0.5, eps_d = 0.02;
    int    nthr = omp_get_max_threads();

    int opt;
    while ((opt = getopt(argc, argv, "p:b:f:s:e:d:t:o:h")) != -1) {
        switch (opt) {
        case 'p': prime   = atoi(optarg);  break;
        case 'b': binfile = optarg;        break;
        case 'f': nframes = atoll(optarg);  break;
        case 's': eps_s   = atof(optarg);  break;
        case 'e': eps_e   = atof(optarg);  break;
        case 'd': eps_d   = atof(optarg);  break;
        case 't': nthr    = atoi(optarg);  break;
        case 'o': csvfile = optarg;        break;
        default: usage(argv[0]); return 1;
        }
    }
    if ((!prime && !binfile) || !nframes) { usage(argv[0]); return 1; }

    /* Build or load H */
    HMat *H = NULL;
    if (binfile) {
        printf("Loading H from %s ...\n", binfile);
        H = hmat_read_bin(binfile);
    } else {
        printf("Building EQR code for p = %d ...\n", prime);
        H = build_eqr_hmat(prime);
    }
    if (!H) { fprintf(stderr, "Failed to obtain H matrix.\n"); return 1; }

    int n = H->n, k = n - H->nk;
    printf("Code: (%d, %d), R = %.4f, %d words/col, %d threads\n",
           n, k, (double)k/n, H->nw, nthr);
    printf("Frames: %lld, eps: [%.4f, %.4f] step %.4f\n\n", nframes, eps_s, eps_e, eps_d);

    FILE *csv = NULL;
    if (csvfile) {
        csv = fopen(csvfile, "w");
        if (csv) fprintf(csv, "epsilon,P_ambiguous,frames,time_sec\n");
    }

    printf("%-12s %-18s %-14s %s\n", "epsilon", "P(ambiguous)", "frames", "time");
    printf("%-12s %-18s %-14s %s\n", "--------", "----------", "------", "----");

    for (double eps = eps_s; eps <= eps_e + eps_d * 0.01; eps += eps_d) {
        if (eps > 1.0) eps = 1.0;
        double t0 = omp_get_wtime();
        double pamb = simulate_eps(H, eps, nframes, nthr);
        double dt = omp_get_wtime() - t0;

        printf("%-12.4f %-18.6e %-14lld %.3fs\n", eps, pamb, nframes, dt);
        fflush(stdout);

        if (csv) {
            fprintf(csv, "%.6f,%.12e,%lld,%.4f\n", eps, pamb, nframes, dt);
            fflush(csv);
        }
        if (eps >= 1.0 - 1e-9) break;
    }

    if (csv) fclose(csv);
    hmat_free(H);
    printf("\nDone.\n");
    return 0;
}