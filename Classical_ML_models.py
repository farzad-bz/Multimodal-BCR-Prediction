
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from lifelines.utils import concordance_index
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM

from lifelines import CoxPHFitter
import argparse


def run_survival_cv_benchmark(
    df: pd.DataFrame,
    feature_cols,
    time_col: str = "time_to_follow-up/BCR",
    event_col: str = "BCR",
    random_state: int = 42,
):
    """
    Runs K-fold CV for multiple survival models and reports C-index per model.
    - df: cleaned dataframe with numeric features
    - feature_cols: list of feature names to use
    - time_col: duration column
    - event_col: event indicator (1=event, 0=censored)
    """

    # Basic checks
    assert time_col in df.columns and event_col in df.columns, "Missing time/event columns"

    # Prepare X, T, E
    X = df[feature_cols].copy()
    T = pd.to_numeric(df[time_col], errors="coerce").values.astype(float)
    E = pd.to_numeric(df[event_col], errors="coerce").values.astype(int)

    # Safety: small epsilon for non-positive times
    T = np.where(T <= 0, 0.1, T)

    results = []  # list of dicts: {"model": name, "fold": k, "c_index": val}

    # -------------------------
    # Helper: evaluate c-index
    # -------------------------
    def cindex_sksurv(t_true, e_true, risk):
        # risk: higher = higher hazard/earlier event
        c = concordance_index_censored(event_indicator=e_true.astype(bool),
                                    event_time=t_true,
                                    estimate=risk)  # <-- no minus
        return float(c[0])

    def cindex_lifelines(t_true, e_true, risk):
        # lifelines' c-index expects prediction where *higher = longer time*
        # Cox partial hazard: higher hazard = shorter time -> pass -risk
        return float(concordance_index(t_true, -risk, e_true))

    # -------------------------
    # Model runners (per fold)
    # -------------------------
    def run_cox_lifelines(X_tr, T_tr, E_tr, X_va, T_va, E_va):
        # Impute+scale
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        Xtr = scl.fit_transform(imp.fit_transform(X_tr))
        Xva = scl.transform(imp.transform(X_va))
        # lifelines needs a single DataFrame with time/event
        train_df = pd.DataFrame(Xtr, columns=feature_cols)
        train_df["time"] = T_tr
        train_df["event"] = E_tr
        cph = CoxPHFitter(penalizer=0.01)  # small L2 for stability
        cph.fit(train_df, duration_col="time", event_col="event", show_progress=False)

        risk = cph.predict_partial_hazard(pd.DataFrame(Xva, columns=feature_cols)).values.ravel()
        return cindex_lifelines(T_va, E_va, risk)

    def run_coxnet(X_tr, T_tr, E_tr, X_va, T_va, E_va):
        y_tr = Surv.from_arrays(event=E_tr.astype(bool), time=T_tr)

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("coxnet", CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=5000))
        ])
        pipe.fit(X_tr, y_tr)


        # ---- risk prediction & c-index ----
        risk = pipe.predict(X_va)  # higher = higher risk
        return concordance_index_censored(E_va.astype(bool), T_va, risk)[0]


    def rsf_expected_time(rsf, X_np):
        """
        Compute expected survival time E[T] by integrating each predicted survival
        step function: E[T] ≈ ∫ S(t) dt. Returns risk = -E[T].
        Works across sksurv versions (no event_times_ needed).
        """
        surv_fns = rsf.predict_survival_function(X_np)  # list of StepFunction
        exp_T = []
        for fn in surv_fns:
            # fn.x: time grid, fn.y: survival values on that grid
            t = np.asarray(fn.x, dtype=float)
            s = np.asarray(fn.y, dtype=float)
            # ensure strictly increasing times and non-negative survival
            if t.ndim == 0 or s.ndim == 0 or len(t) != len(s):
                exp_T.append(np.nan)
                continue
            t = np.maximum.accumulate(t)  # monotone
            s = np.clip(s, 0.0, 1.0)
            exp_T.append(np.trapz(s, t))
        exp_T = np.array(exp_T, dtype=float)
        risk = -exp_T  # higher risk = shorter expected time
        return risk


    def run_rsf(X_tr, T_tr, E_tr, X_va, T_va, E_va):
        y_tr = Surv.from_arrays(event=E_tr.astype(bool), time=T_tr)

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rsf", RandomSurvivalForest(
                n_estimators=500,
                min_samples_split=10,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state
            ))
        ])
        pipe.fit(X_tr, y_tr)

        # get imputed validation matrix
        Xva_imp = pipe.named_steps["imp"].transform(X_va)
        # compute risk from survival curves
        risk = rsf_expected_time(pipe.named_steps["rsf"], Xva_imp)

        return cindex_sksurv(T_va, E_va, risk)

    def run_gbsa(X_tr, T_tr, E_tr, X_va, T_va, E_va):
        y_tr = Surv.from_arrays(event=E_tr.astype(bool), time=T_tr)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),  # GB often benefits from scaling for continuous splits
            ("gb", GradientBoostingSurvivalAnalysis(
                learning_rate=0.05,
                max_depth=3,
                n_estimators=800,
                random_state=random_state
            ))
        ])
        pipe.fit(X_tr, y_tr)

        
        risk = pipe.predict(X_va)  # higher risk -> shorter survival
        return cindex_sksurv(T_va, E_va, risk)

    # -------------------------
    # CV loop
    # -------------------------
    model_specs = [
        ("CoxPH (lifelines)", run_cox_lifelines),
        ("Coxnet (elastic-net Cox)", run_coxnet),
        ("Random Survival Forest", run_rsf),
        ("GB Survival Analysis", run_gbsa),
    ]

    # fold = 0
        
    for fold in df["fold"].unique():
        va_mask = (df["fold"].values == fold)
        tr_mask = ~va_mask

        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]
        
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        T_tr, T_va = T[tr_idx], T[va_idx]
        E_tr, E_va = E[tr_idx], E[va_idx]
        
        for name, runner in model_specs:
            try:
                c = runner(X_tr, T_tr, E_tr, X_va, T_va, E_va)
            except Exception as ex:
                c = None
                print(f"[Fold {fold}] {name} failed: {ex}")
            if c is not None:
                results.append({"model": name, "fold": fold, "c_index": c})

    if not results:
        raise RuntimeError("No models ran successfully. Check installations and inputs.")

    res_df = pd.DataFrame(results)
    summary = (res_df.groupby("model")["c_index"]
               .agg(["mean", "std", "count"])
               .sort_values("mean", ascending=False))
    print("\n=== Survival CV Results (C-index) ===")
    print(summary.round(3))
    return res_df, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicting biochemical recurrence (BCR) using classical ML models (clinical data only)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/base.yaml)"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    clinical_df = pd.read_csv(cfg.data.clinical_df_path, index_col=0)

    res_df, summary = run_survival_cv_benchmark(clinical_df, cfg.data.features, time_col="time_to_follow-up/BCR", event_col="BCR")