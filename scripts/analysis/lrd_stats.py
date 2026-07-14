"""Defensible statistics for a two-arm PPL comparison at small n.

`docs/learned_restriction.md` reported its headline as "5.8 points (~6 sigma)" and
"74.36 +/- 0.06" from **three seeds**. Both statements overreach, and a referee will say so:

* **A "sigma" from n=3 is an estimate with 2 degrees of freedom.** Its own relative standard
  error is ~50%, so quoting a 6-sigma effect from it is quoting a ratio whose denominator is
  barely determined. The right instrument is Welch's t (which does not assume equal variances
  and pays for the small n through its degrees of freedom), reported with a confidence
  interval on the *difference* -- the quantity anyone actually cares about.

* **"Every seed of A beats every seed of B" is a real result, but it is worth p = 0.05, not
  p = 1e-9.** It is exactly the extreme case of the Wilcoxon rank-sum / exact permutation
  test, and with 3 vs 3 there are only C(6,3) = 20 ways to split six numbers into two labelled
  groups of three. Complete separation is the most extreme of them, so the smallest two-sided
  p-value the design can *ever* produce is 2/20 = 0.10 (one-sided 1/20 = 0.05). No amount of
  separation buys more evidence than that, because the design does not contain more. This is a
  fact about n, not about the effect.

The honest summary of a 3-seed win is therefore: a point estimate, a wide interval, and a
one-sided exact p at its floor of 0.05. That is still evidence -- it is simply not 6 sigma.
Raising n is the only way to buy more.

Usage::

    python -m scripts.analysis.lrd_stats results_a.json results_b.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Welch's t statistic, its Welch-Satterthwaite dof, and the two-sided p-value."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return float("inf"), float(na + nb - 2), 0.0
    t = (ma - mb) / math.sqrt(se2)
    dof = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, dof, 2 * _student_sf(abs(t), dof)


def _student_sf(t: float, dof: float) -> float:
    """P(T > t) for Student's t. Via the regularized incomplete beta, no SciPy dependency."""
    x = dof / (dof + t * t)
    return 0.5 * _betainc(dof / 2, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), by the standard continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    if x >= (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def exact_perm_p(a: list[float], b: list[float]) -> tuple[float, float]:
    """One-sided exact permutation p for ``mean(a) < mean(b)``, and the design's *floor*.

    The floor is the smallest p this design can produce however large the effect: with ``na``
    and ``nb`` seeds there are ``C(na+nb, na)`` labellings, so no result can be rarer than
    ``1 / C(na+nb, na)``. Quote it: it is the honest ceiling on the evidence n can carry.
    """
    na, nb = len(a), len(b)
    pool = a + b
    obs = sum(b) / nb - sum(a) / na
    total = hit = 0
    for combo in itertools.combinations(range(na + nb), na):
        ga = [pool[i] for i in combo]
        gb = [pool[i] for i in range(na + nb) if i not in combo]
        total += 1
        if sum(gb) / nb - sum(ga) / na >= obs:
            hit += 1
    return hit / total, 1.0 / math.comb(na + nb, na)


def summarize(name: str, v: list[float]) -> str:
    n = len(v)
    mu = sum(v) / n
    sd = (sum((x - mu) ** 2 for x in v) / max(n - 1, 1)) ** 0.5 if n > 1 else float("nan")
    return f"{name:>8}: {mu:7.2f} +/- {sd:5.2f}  (n={n})  {[round(x, 2) for x in v]}"


def compare(base: list[float], new: list[float], base_name="pca", new_name="lrd") -> None:
    print(summarize(base_name, base))
    print(summarize(new_name, new))
    nb_, nn = len(base), len(new)
    mb = sum(base) / nb_
    mn = sum(new) / nn
    diff = mn - mb
    t, dof, p = welch(new, base)

    # 95% CI on the difference of means, Welch. t* via a bisection on the survival function.
    vb = sum((x - mb) ** 2 for x in base) / max(nb_ - 1, 1)
    vn = sum((x - mn) ** 2 for x in new) / max(nn - 1, 1)
    se = math.sqrt(vb / nb_ + vn / nn)
    lo_t, hi_t = 0.0, 100.0
    for _ in range(200):
        mid = (lo_t + hi_t) / 2
        if 2 * _student_sf(mid, dof) > 0.05:
            lo_t = mid
        else:
            hi_t = mid
    tstar = (lo_t + hi_t) / 2

    print(f"\n  difference   {diff:+.2f} PPL  ({100 * diff / mb:+.1f}%)")
    print(f"  95% CI       [{diff - tstar * se:+.2f}, {diff + tstar * se:+.2f}]  "
          f"(Welch, dof={dof:.1f})")
    print(f"  Welch t      t={t:.2f}, p={p:.3f}")
    p1, floor = exact_perm_p(new, base)
    sep = max(new) < min(base) or max(base) < min(new)
    print(f"  exact perm   one-sided p={p1:.3f}   (design floor {floor:.3f} at "
          f"n={nn} vs {nb_}: no result can beat it)")
    print(f"  separation   {'complete' if sep else 'overlapping'}")
    if p1 <= floor + 1e-9 and sep:
        print(f"\n  Note: complete separation. That is the strongest outcome this design "
              f"admits,\n  and it is worth one-sided p={floor:.2f} -- not a 'sigma' count. "
              f"To claim more, raise n.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="JSON files from scripts/analysis/lrd_validate")
    ap.add_argument("--base", default="pca")
    ap.add_argument("--new", default="lrd")
    a = ap.parse_args()

    by_arm: dict[str, list[float]] = {}
    for f in a.results:
        for r in json.loads(Path(f).read_text())["rows"]:
            if "arm" in r and math.isfinite(r.get("ppl", float("inf"))):
                by_arm.setdefault(r["arm"], []).append(r["ppl"])
    if a.base in by_arm and a.new in by_arm:
        compare(by_arm[a.base], by_arm[a.new], a.base, a.new)
    else:
        for k, v in by_arm.items():
            print(summarize(k, v))


if __name__ == "__main__":
    main()
