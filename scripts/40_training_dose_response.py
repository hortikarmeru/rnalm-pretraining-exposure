"""Dose response of the training experiment across checkpoints, next to the observational exposure slope.

Usage: python scripts/40_training_dose_response.py <model key> [--simulate]   (environment-extra.yml; after script 28)
  --simulate: every checkpoint is the released model (script 28 "released") plus a known effect of
  0.05 F1 per unit of dose on its own set's families, a generic +0.01 shift of model B on every
  sequence and noise, to check the estimates recover them before real results are used.

Dose per eval sequence q at injected epoch e (n = injected homologs, c = clusters among its corpus
homologs, script 21; config training_experiment.dose):
  times seen (main): x_q(e) = ln(1 + P c + e n) - ln(1 + P c), P = pretraining_epochs. RiNALMo saw about
    one sequence per cluster per epoch, so P c approximates how often the model saw q's relatives;
    each injected homolog is seen once per injected epoch. Paired with script 19's cluster_count fit:
    at a fixed number of epochs, a log change in presentations is a log change in clusters.
  homolog count: delta_log_exposure from script 26 (no epoch term), paired with script 19's main fit.
Both are natural-log units, so every slope is F1 per ln unit, the units of script 19's log_homologs.

Unit of analysis, as in script 29: own-minus-other F1 per sequence (mean over seeds), mean per
family, mean per side of a pair, the two sides weighted equally; pairs with one side only are left
out. That gives V_p(e) (gain) and X_p(e) (dose) per pair. Estimates:
  (a) per checkpoint: beta(e) = mean_p V_p(e) / mean_p X_p(e) (through the origin: no dose, no difference);
  (b) pooled: through-origin least squares of V_p(e) on X_p(e) over pairs and checkpoints;
  (c) curvature: (b) with an X^2 term added;
  (d) trend over epochs: per pair, the least-squares slope of V_p(e) on log2(e), averaged over pairs
      (sign-flip test over pairs, as in script 29);
  (e) within-family slope at the last checkpoint: per-sequence gain on dose, both centred within family,
      on both dose scales (removes family traits and any difference between model A and model B).
Uncertainty: bootstrap over pairs (a resampled pair brings all its checkpoints and sequences). The
observational slope enters as a normal draw (estimate, SE) paired with each bootstrap draw, for the
difference and ratio of causal and observational slopes.

Also: the zero-dose point (seed 0: the other-set model minus the replay-only model on each family,
pair-balanced; should be about 0), and the effect by set. On one side S of the pairs, the own-minus-
other difference D_S also holds any generic difference g between model B and model A, so
tau_A = D_A + g and tau_B = D_B - g, with g = sentinel mean F1 of model B minus model A (mean over
seeds; held fixed in the bootstrap). Seed 0 also gives each side against the replay-only model. One
side is not a randomized contrast, so these get bootstrap intervals but no permutation test.

Writes results/40_dose_response_<model>[_simulated].json with its .meta.json.
"""
import json
import sys
from collections import defaultdict

import numpy as np

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.pair_balance import read_tsv, scores

SIM_SLOPE, SIM_GENERIC, SIM_SD = 0.05, 0.01, 0.03   # --simulate only

NOTES = [
    "Continued training at a constant learning rate of 1e-5, after pretraining ended, is not the original schedule.",
    "Injected homologs are recent sequences and are not cluster-subsampled, so each one counts as a draw per epoch.",
    "The observational slope spans the whole exposure range; eval sequences have at most max_corpus_homologs "
    "corpus homologs, and the observational slope carries confounding and error in measured exposure.",
    "The times-seen scale assumes six pretraining epochs of one sequence per cluster, on the approximated corpus.",
    "The replay-only comparison has one seed.",
]


def interval(draws):
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def summary(est, draws):
    lo, hi = interval(draws)
    return {"estimate": float(est), "ci_low": lo, "ci_high": hi}


def main():
    config = load_config()
    exp, results = config["training_experiment"], repo_path(config["paths"]["results_dir"])
    cfg, dcfg = exp["analysis"], exp["dose"]
    key = sys.argv[1]
    simulate = "--simulate" in sys.argv
    eval_rows = read_tsv(results / "26_eval_sequences.tsv")
    assign = {r["family"]: r for r in read_tsv(results / "26_assignment.tsv")}
    clusters = {r["seq_id"]: int(r["clusters"]) for r in read_tsv(results / dcfg["cluster_table"])}
    seeds = exp["training"]["seeds"]
    epochs = sorted(exp["training"]["checkpoints_at_epochs"])
    last = epochs[-1]
    P = dcfg["pretraining_epochs"]
    set_of = {r["seq_id"]: r["set"] for r in eval_rows}
    rng = np.random.default_rng(cfg["seed"])

    def dose(r, e):
        c, n = clusters.get(r["seq_id"], 0), int(r["injected_homologs"])
        return float(np.log1p(P * c + e * n) - np.log1p(P * c))

    cache = {}

    def f1(arm, seed, epoch, role="eval"):
        if (arm, seed, epoch, role) in cache:   # read (and, simulated, drawn) once, so every estimate sees the same values
            return cache[arm, seed, epoch, role]
        s = scores(results, key, "released" if simulate else f"{arm}_s{seed}_e{epoch}", role)
        if simulate:
            x = {r["seq_id"]: dose(r, epoch) for r in eval_rows}
            s = {q: v + (SIM_SLOPE * x[q] if set_of.get(q) == arm else 0.0) + (SIM_GENERIC if arm == "B" else 0.0)
                 + rng.normal(0, SIM_SD) for q, v in s.items()}
        cache[arm, seed, epoch, role] = s
        return s

    # per sequence: gain (own - other, mean over seeds) and dose, at every checkpoint
    gain = {e: defaultdict(list) for e in epochs}
    for e in epochs:
        for sd in seeds:
            a, b = f1("A", sd, e), f1("B", sd, e)
            for r in eval_rows:
                q = r["seq_id"]
                gain[e][q].append(a[q] - b[q] if r["set"] == "A" else b[q] - a[q])
    gain = {e: {q: float(np.mean(v)) for q, v in g.items()} for e, g in gain.items()}
    x = {e: {r["seq_id"]: dose(r, e) for r in eval_rows} for e in epochs}
    d_count = {r["seq_id"]: float(r["delta_log_exposure"]) for r in eval_rows}

    def side_means(values):
        """{pair: {set: mean over the side's families of the family mean}}."""
        fams = defaultdict(list)
        for r in eval_rows:
            fams[r["family"]].append(values[r["seq_id"]])
        sides = defaultdict(lambda: defaultdict(list))
        for f, v in fams.items():
            sides[assign[f]["pair"]][assign[f]["set"]].append(np.mean(v))
        return {p: {s: float(np.mean(v)) for s, v in ss.items()} for p, ss in sides.items()}

    first = side_means(gain[last])
    pairs = sorted(p for p, ss in first.items() if len(ss) == 2)
    n_pairs = len(pairs)
    boot = rng.integers(0, n_pairs, size=(cfg["bootstrap"], n_pairs))

    def balanced(values):
        sm = side_means(values)
        return np.array([(sm[p]["A"] + sm[p]["B"]) / 2 for p in pairs])

    V = np.stack([balanced(gain[e]) for e in epochs], axis=1)   # [pairs, checkpoints]
    X = np.stack([balanced(x[e]) for e in epochs], axis=1)
    Xc = balanced(d_count)
    out = {"model": key, "simulated": simulate, "pairs": n_pairs, "epochs": epochs, "notes": NOTES}

    # (a) per checkpoint, both scales
    beta_draws = {}
    out["per_checkpoint"] = {}
    for k, e in enumerate(epochs):
        bd = V[boot, k].mean(1) / X[boot, k].mean(1)
        cd = V[boot, k].mean(1) / Xc[boot].mean(1)
        beta_draws[str(e)] = {"times_seen": bd, "homolog_count": cd}
        out["per_checkpoint"][str(e)] = {
            "gain": summary(V[:, k].mean(), V[boot, k].mean(1)),
            "mean_dose_times_seen": float(X[:, k].mean()), "mean_dose_homolog_count": float(Xc.mean()),
            "slope_times_seen": summary(V[:, k].mean() / X[:, k].mean(), bd),
            "slope_homolog_count": summary(V[:, k].mean() / Xc.mean(), cd)}

    # (b) pooled through-origin slope and (c) curvature, times-seen scale
    def pooled(v, xx):
        return float((v * xx).sum() / (xx * xx).sum())

    def curved(v, xx):
        coef, *_ = np.linalg.lstsq(np.stack([xx.ravel(), xx.ravel() ** 2], 1), v.ravel(), rcond=None)
        return coef

    pooled_draws = np.array([pooled(V[i], X[i]) for i in boot])
    beta_draws["pooled"] = {"times_seen": pooled_draws}
    out["pooled_slope_times_seen"] = summary(pooled(V, X), pooled_draws)
    cd = np.array([curved(V[i], X[i]) for i in boot])
    c = curved(V, X)
    out["curvature_times_seen"] = {"linear": summary(c[0], cd[:, 0]), "quadratic": summary(c[1], cd[:, 1])}

    # (d) trend over epochs: per-pair slope on log2(epoch)
    t = np.log2(epochs)
    slopes = ((V - V.mean(1, keepdims=True)) * (t - t.mean())).sum(1) / ((t - t.mean()) ** 2).sum()
    null = rng.choice([-1, 1], size=(cfg["sign_flip_permutations"], n_pairs)) @ slopes / n_pairs
    k_extreme = int(np.sum(np.abs(null) >= abs(slopes.mean()) - 1e-12))
    out["trend_per_doubling"] = summary(slopes.mean(), slopes[boot].mean(1)) | {
        "p_sign_flip": (k_extreme + 1) / (len(null) + 1)}

    # (e) within-family slope at the last checkpoint, both scales
    out["within_family_last"] = {}
    pair_of_family = {f: r["pair"] for f, r in assign.items()}
    for scale, dv in (("times_seen", x[last]), ("homolog_count", d_count)):
        sxy, sxx = defaultdict(float), defaultdict(float)
        fams = defaultdict(list)
        for r in eval_rows:
            fams[r["family"]].append(r["seq_id"])
        for f, qs in fams.items():
            xs = np.array([dv[q] for q in qs])
            ys = np.array([gain[last][q] for q in qs])
            xs, ys = xs - xs.mean(), ys - ys.mean()
            p = pair_of_family[f]
            sxy[p] += float((xs * ys).sum())
            sxx[p] += float((xs * xs).sum())
        a = np.array([sxy[p] for p in pairs])
        b = np.array([sxx[p] for p in pairs])
        out["within_family_last"][scale] = summary(a.sum() / b.sum(), a[boot].sum(1) / b[boot].sum(1))

    # causal vs observational slope
    fits = json.loads((results / "19_analysis_real.json").read_text())["models"][key]
    out["observational_comparison"] = {}
    for scale, fit in dcfg["observational_fits"].items():
        term = fits[fit]["terms"]["log_homologs"]
        obs = rng.normal(term["estimate"], term["se"], size=dcfg["observational_draws"])
        comp = {"observational": {"fit": fit, "estimate": term["estimate"], "se": term["se"]}}
        for label, draws in beta_draws.items():
            if scale not in draws:
                continue
            bd = draws[scale]
            n = min(len(bd), len(obs))
            est = (out["pooled_slope_times_seen"]["estimate"] if label == "pooled"
                   else out["per_checkpoint"][label][f"slope_{scale}"]["estimate"])
            comp[label] = {"difference": summary(est - term["estimate"], bd[:n] - obs[:n]),
                           "ratio": summary(est / term["estimate"], bd[:n] / obs[:n])}
        out["observational_comparison"][scale] = comp

    # zero-dose point: other-set model minus replay-only, seed 0
    out["zero_dose_other_minus_replay_only_s0"] = {}
    for e in epochs:
        a, b, z = f1("A", 0, e), f1("B", 0, e), f1("0", 0, e)
        v = {r["seq_id"]: (b if r["set"] == "A" else a)[r["seq_id"]] - z[r["seq_id"]] for r in eval_rows}
        bal = balanced(v)
        out["zero_dose_other_minus_replay_only_s0"][str(e)] = summary(bal.mean(), bal[boot].mean(1))

    # by set
    out["by_set"] = {}
    for e in epochs:
        g = []
        for sd in seeds:
            sa, sb = f1("A", sd, e, "sentinel"), f1("B", sd, e, "sentinel")
            g.append(np.mean([sb[q] - sa[q] for q in sa]))
        g = float(np.mean(g))
        sm = side_means(gain[e])
        a0, b0, z = f1("A", 0, e), f1("B", 0, e), f1("0", 0, e)
        own0 = side_means({r["seq_id"]: (a0 if r["set"] == "A" else b0)[r["seq_id"]] - z[r["seq_id"]] for r in eval_rows})
        oth0 = side_means({r["seq_id"]: (b0 if r["set"] == "A" else a0)[r["seq_id"]] - z[r["seq_id"]] for r in eval_rows})
        xs = side_means(x[e])
        entry = {"generic_B_minus_A_sentinel": g}
        tau = {}
        for s, sign in (("A", 1), ("B", -1)):
            ps = sorted(p for p in sm if s in sm[p])
            bs = rng.integers(0, len(ps), size=(cfg["bootstrap"], len(ps)))
            D = np.array([sm[p][s] for p in ps])
            tau[s] = D + sign * g
            mean_x = float(np.mean([xs[p][s] for p in ps]))
            own = np.array([own0[p][s] for p in ps])
            oth = np.array([oth0[p][s] for p in ps])
            entry[s] = {"pairs": len(ps), "D": summary(D.mean(), D[bs].mean(1)),
                        "tau": summary(tau[s].mean(), tau[s][bs].mean(1)),
                        "mean_dose_times_seen": mean_x,
                        "tau_per_dose": summary(tau[s].mean() / mean_x, tau[s][bs].mean(1) / mean_x),
                        "own_minus_replay_only_s0": summary(own.mean(), own[bs].mean(1)),
                        "other_minus_replay_only_s0": summary(oth.mean(), oth[bs].mean(1))}
        diff = np.array([(sm[p]["B"] - g) - (sm[p]["A"] + g) for p in pairs])
        entry["tau_B_minus_tau_A"] = summary(diff.mean(), diff[boot].mean(1))
        out["by_set"][str(e)] = entry

    fam_of = defaultdict(list)
    for r in eval_rows:
        fam_of[r["family"]].append(r["seq_id"])

    def family_mean(values, s):
        return float(np.mean([np.mean([values[q] for q in qs]) for f, qs in fam_of.items() if assign[f]["set"] == s]))

    out["dose_by_set"] = {s: {"families": sum(assign[f]["set"] == s for f in fam_of),
                              "family_mean_times_seen": {str(e): family_mean(x[e], s) for e in epochs},
                              "family_mean_homolog_count": family_mean(d_count, s)} for s in ("A", "B")}

    name = f"40_dose_response_{key}{'_simulated' if simulate else ''}.json"
    path = results / name
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    write_meta(path, config, packages=["numpy"])


if __name__ == "__main__":
    main()
