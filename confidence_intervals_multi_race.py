import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from F1DataExtractor import F1DataExtractor
from RacePaceFormulator import RacePaceFormulator
from MonteCarlo import MonteCarloSimulator

DRIVER = "PIA"
YEAR = 2023
SESSION = "R"
RACES = [
    "Bahrain Grand Prix",
    "Australian Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Spanish Grand Prix",
    "British Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Japanese Grand Prix",
]

PARTIAL_LAPS = 5          # laps of partial data used to fit the model
TARGET_LAPS = 20          # full-stint length we extrapolate to
MIN_STINT_LAPS = PARTIAL_LAPS + 3  # need >= 3 clean laps left to test extrapolation
SIM_SIZE = 1000
SEED = 42

OUTPUT_PLOT = "stint_prediction_vs_real.png"
OUTPUT_BIAS_PLOT = "bias_analysis.png"

# ===================== CORE FUNCTIONS =====================


def fit_from_partial_stint(stint_df: pd.DataFrame, partial_laps: int):
    """
    Fit the quadratic degradation model using all cleaned laps with
    stint-relative index < partial_laps (i.e. the first `partial_laps`
    in-stint laps). Using LapNumber-based relative indexing keeps alignment
    correct even when pick_quicklaps() removed laps from the stint.
    """
    stint_rel = stint_df["LapNumber"].to_numpy() - stint_df["LapNumber"].min()
    partial_df = stint_df[stint_rel < partial_laps]
    formulator = RacePaceFormulator(partial_df)
    fitted = formulator.calculate_fitted_deg_rate()
    if fitted.empty:
        return None, None
    row = fitted.iloc[0]
    params = {
        "base_time": row["base_time"],
        "deg_rate": row["deg_rate"],
        "deg_curvature": row["deg_curvature"],
        "fuel_loss": row["fuel_loss"],
        "pcov": row["pcov"],
        "sigma": row["sigma"],
    }
    return row, params


def simulate_stint(params: dict, n_laps: int, sim_size: int, seed: int):
    """
    Monte-Carlo simulate n_laps for sim_size stints.
    Returns (simulated, model_mean, lower, upper) where model_mean/lower/upper
    are per-lap arrays (length n_laps).
    """
    mc = MonteCarloSimulator(seed=seed)
    simulated = mc.simulate_stint_from_covariance(params, n_laps, sim_size)
    # simulated shape: (n_laps, sim_size)
    # SEMANTIC FIX: percentile across simulations (axis=1), NOT across laps.
    model_mean = simulated.mean(axis=1)
    lower, upper = np.percentile(simulated, [2.5, 97.5], axis=1)
    return simulated, model_mean, lower, upper


def analyze_race(extractor: F1DataExtractor, year: int, weekend: str, driver: str):
    """Full pipeline for one race. Returns (result_dict, error_str)."""
    extractor.load_session(year, weekend, SESSION)
    driver_df = extractor.get_driver_session(driver)
    if driver_df.empty:
        return None, f"No lap data for {driver} at {weekend} {year}"

    # ---- select the longest stint that leaves extrapolation room ----
    stint_sizes = driver_df.groupby("Stint").size()
    candidates = stint_sizes[stint_sizes >= MIN_STINT_LAPS]
    if candidates.empty:
        return None, f"No stint with >= {MIN_STINT_LAPS} laps at {weekend} {year}"

    stint_id = int(candidates.idxmax())
    stint_df = (
        driver_df[driver_df["Stint"] == stint_id]
        .sort_values("LapNumber")
        .reset_index(drop=True)
    )
    compound = stint_df["Compound"].iloc[0]

    # ---- stint-relative lap index (0-based). pick_quicklaps() may have
    # removed laps (pit/VSC/SC), so LapNumber can have gaps; all alignment
    # is done through this relative index, never via row position. ----
    stint_rel = stint_df["LapNumber"].to_numpy(dtype=int) - stint_df["LapNumber"].min()
    real_laps = stint_df["LapTime_Seconds"].to_numpy(dtype=float)

    # ---- fit on partial data only (stint-relative index < PARTIAL_LAPS) ----
    fit_row, params = fit_from_partial_stint(stint_df, PARTIAL_LAPS)
    if fit_row is None:
        return None, f"Fit failed for stint {stint_id} at {weekend} {year}"

    # ---- extrapolate to a full TARGET_LAPS stint via Monte-Carlo ----
    simulated, model_mean, lower, upper = simulate_stint(
        params, TARGET_LAPS, SIM_SIZE, SEED
    )

    # ---- extrapolation window: real laps beyond the partial-fit window,
    # clipped at the 20-lap extrapolation horizon ----
    in_extrap = (stint_rel >= PARTIAL_LAPS) & (stint_rel < TARGET_LAPS)
    ext_idx = np.where(in_extrap)[0]  # row positions into stint_df

    real_ext = real_laps[ext_idx]
    rel_ext = stint_rel[ext_idx]
    mean_ext = model_mean[rel_ext]
    lower_ext = lower[rel_ext]
    upper_ext = upper[rel_ext]

    residual = real_ext - mean_ext  # > 0 => real slower than model

    # full-length alignment arrays (NaN where no real lap exists)
    full_real = np.full(TARGET_LAPS, np.nan)
    full_real[stint_rel[stint_rel < TARGET_LAPS]] = real_laps[stint_rel < TARGET_LAPS]

    return {
        "year": year,
        "weekend": weekend,
        "stint_id": stint_id,
        "compound": compound,
        "total_laps": len(stint_df),
        "partial_laps": PARTIAL_LAPS,
        "fit_row": fit_row,
        "params": params,
        "simulated": simulated,
        "model_mean": model_mean,
        "lower": lower,
        "upper": upper,
        "real_laps": real_laps,
        "stint_rel": stint_rel,
        "full_real": full_real,
        "ext_idx": ext_idx,
        "residual": residual,
        "mean_residual": float(residual.mean()) if len(residual) else np.nan,
        "n_extrapolated": len(ext_idx),
        "pct_above_ci": float((real_ext > upper_ext).mean() * 100),
        "pct_below_ci": float((real_ext < lower_ext).mean() * 100),
        "pct_within_ci": float(
            ((real_ext >= lower_ext) & (real_ext <= upper_ext)).mean() * 100
        ),
    }, None


# ===================== PLOTTING =====================


def plot_prediction_vs_real(results: list):
    n = len(results)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), squeeze=False)

    for ax, r in zip(axes.flat, results):
        lap_nums = np.arange(TARGET_LAPS) + 1  # 1-based stint-relative lap number

        # Real lap times (aligned by stint-relative lap index; NaN = no clean lap)
        ax.plot(lap_nums, r["full_real"], "ko-", label="Real", markersize=4)
        # Model mean (full 20-lap extrapolation)
        ax.plot(lap_nums, r["model_mean"], "b-", label="Model mean", linewidth=2)
        # 95% CI band
        ax.fill_between(lap_nums, r["lower"], r["upper"], color="b", alpha=0.2, label="95% CI")
        # Boundary: partial-fit data vs extrapolation
        ax.axvline(PARTIAL_LAPS + 0.5, color="gray", linestyle="--", linewidth=1)
        ax.axvspan(0.5, PARTIAL_LAPS + 0.5, color="gray", alpha=0.12)

        ymin, ymax = ax.get_ylim()
        ax.text(
            PARTIAL_LAPS / 2 + 0.5, ymin, "partial\nfit", ha="center",
            va="bottom", fontsize=7, color="dimgray",
        )
        ax.text(
            PARTIAL_LAPS + (TARGET_LAPS - PARTIAL_LAPS) / 2 + 0.5, ymin,
            "extrapolation", ha="center", va="bottom", fontsize=7, color="royalblue",
        )

        ax.set_title(
            f"{r['weekend']}\nStint {r['stint_id']} ({r['compound']})  "
            f"resid {r['mean_residual']:+.2f}s",
            fontsize=9,
        )
        ax.set_xlabel("Lap (stint-relative)")
        ax.set_ylabel("Lap time (s)")
        ax.grid(alpha=0.3)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"{DRIVER} {YEAR} — Stint Prediction vs Real Lap Time "
        f"(fit on first {PARTIAL_LAPS} laps, extrapolated to {TARGET_LAPS})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_PLOT, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {OUTPUT_PLOT}")


def plot_bias_analysis(results: list):
    races = [r["weekend"] for r in results]
    mean_resid = [r["mean_residual"] for r in results]
    colors = ["red" if v > 0 else "green" if v < 0 else "gray" for v in mean_resid]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(races, mean_resid, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Mean residual: real - model mean (s)")
    ax.set_title(
        f"{DRIVER} {YEAR} — Bias Direction per Race "
        f"(fit on first {PARTIAL_LAPS} laps, extrapolated to {TARGET_LAPS})"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    for bar, v in zip(bars, mean_resid):
        offset = 0.01 if v >= 0 else -0.04
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{v:+.2f}",
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_BIAS_PLOT, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {OUTPUT_BIAS_PLOT}")


# ===================== MAIN =====================


def main():
    print(f"Analyzing {len(RACES)} races for {DRIVER} ({YEAR} {SESSION})")
    print(
        f"Partial-fit laps: {PARTIAL_LAPS} | Target stint: {TARGET_LAPS} laps | "
        f"Simulations: {SIM_SIZE} | Min stint laps: {MIN_STINT_LAPS}"
    )

    extractor = F1DataExtractor()
    results, failures = [], []

    for weekend in RACES:
        print(f"\n--- {weekend} {YEAR} ---")
        try:
            result, err = analyze_race(extractor, YEAR, weekend, DRIVER)
        except Exception as e:  # noqa: BLE001 - surface any FastF1/network error
            failures.append((weekend, str(e)))
            print(f"  ERROR: {e}")
            continue

        if result is None:
            failures.append((weekend, err))
            print(f"  SKIP: {err}")
            continue

        results.append(result)
        print(
            f"  Stint {result['stint_id']} ({result['compound']}), "
            f"{result['total_laps']} real laps"
        )
        print(
            f"  Mean residual over {result['n_extrapolated']} extrapolated laps: "
            f"{result['mean_residual']:+.3f} s"
        )
        print(
            f"  Outside 95% CI: {result['pct_above_ci']:.1f}% above, "
            f"{result['pct_below_ci']:.1f}% below, "
            f"{result['pct_within_ci']:.1f}% within"
        )

    # ---------- aggregate results ----------
    print("\n" + "=" * 90)
    print("AGGREGATE RESULTS")
    print("=" * 90)
    if results:
        summary = pd.DataFrame(
            [
                {
                    "Race": r["weekend"],
                    "Stint": r["stint_id"],
                    "Compound": r["compound"],
                    "RealLaps": r["total_laps"],
                    "ExtrapLaps": r["n_extrapolated"],
                    "MeanResid_s": round(r["mean_residual"], 3),
                    "PctAboveCI": round(r["pct_above_ci"], 1),
                    "PctBelowCI": round(r["pct_below_ci"], 1),
                    "PctWithinCI": round(r["pct_within_ci"], 1),
                    "Direction": (
                        "real slower" if r["mean_residual"] > 0 else "real faster"
                    ),
                }
                for r in results
            ]
        )
        print(summary.to_string(index=False))

        all_residuals = np.concatenate([r["residual"] for r in results])
        n_above = int((all_residuals > 0).sum())
        n_below = int((all_residuals < 0).sum())

        # Proper within-CI count across all extrapolated laps
        within_count = 0
        total_extrap = 0
        for r in results:
            # ext_idx are row positions into stint_df; map to stint-relative
            # indices before indexing into the per-lap CI arrays.
            rel_ext = r["stint_rel"][r["ext_idx"]]
            within_count += int(
                ((r["real_laps"][r["ext_idx"]] >= r["lower"][rel_ext])
                 & (r["real_laps"][r["ext_idx"]] <= r["upper"][rel_ext])).sum()
            )
            total_extrap += r["n_extrapolated"]

        overall_mean = all_residuals.mean()
        print(
            f"\nOverall mean residual across {total_extrap} extrapolated laps: "
            f"{overall_mean:+.3f} s"
        )

    if failures:
        print("\nRaces skipped/failed:")
        for w, e in failures:
            print(f"  - {w}: {e}")

    # ---------- plots ----------
    if results:
        plot_prediction_vs_real(results)
        plot_bias_analysis(results)


if __name__ == "__main__":
    main()
