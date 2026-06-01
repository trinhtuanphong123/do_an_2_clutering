
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr   



from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, silhouette_samples
from sklearn.decomposition import PCA


# Phase 1:
df = pd.read_csv(r"D:\do_an_2_clustering\data\CC GENERAL.csv")


# PHASE 2. DATA AUDIT

# ------------------------------------------------------------
# 1. Basic validation
# ------------------------------------------------------------
def validate_input_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df is empty.")
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {dup_cols}")


# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------
def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def safe_skew(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 3:
        return np.nan
    return float(skew(s, bias=False))


def safe_kurtosis(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 4:
        return np.nan
    return float(kurtosis(s, bias=False, fisher=True))


def infer_frequency_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect likely frequency/probability columns.
    In this dataset, columns containing 'FREQUENCY' and PRC_FULL_PAYMENT
    are expected to be in [0, 1].
    """
    freq_cols = [c for c in df.columns if "FREQUENCY" in c.upper()]
    if "PRC_FULL_PAYMENT" in df.columns:
        freq_cols.append("PRC_FULL_PAYMENT")
    return sorted(set(freq_cols))


def infer_nonnegative_columns(df: pd.DataFrame) -> list[str]:
    """
    In the credit card clustering dataset, most numeric columns should be nonnegative.
    """
    numeric_cols = get_numeric_columns(df)
    return [c for c in numeric_cols if c != "TENURE"]


def infer_positive_expected_columns(df: pd.DataFrame) -> list[str]:
    """
    Columns that should generally be > 0 to avoid downstream ratio issues.
    Zero may exist in practice, so this is a soft audit check, not a hard filter.
    """
    candidates = ["TENURE", "CREDIT_LIMIT"]
    return [c for c in candidates if c in df.columns]


# ------------------------------------------------------------
# 3. Core audit tables
# ------------------------------------------------------------
def build_schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schema and health overview for every column.
    """
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null_count": df.notna().sum(),
        "missing_count": df.isna().sum(),
        "missing_rate_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
    })

    numeric_cols = get_numeric_columns(df)

    report["min"] = np.nan
    report["max"] = np.nan
    report["mean"] = np.nan
    report["std"] = np.nan
    report["median"] = np.nan
    report["skewness"] = np.nan
    report["kurtosis"] = np.nan

    for col in numeric_cols:
        s = df[col]
        report.loc[col, "min"] = s.min(skipna=True)
        report.loc[col, "max"] = s.max(skipna=True)
        report.loc[col, "mean"] = s.mean(skipna=True)
        report.loc[col, "std"] = s.std(skipna=True)
        report.loc[col, "median"] = s.median(skipna=True)
        report.loc[col, "skewness"] = safe_skew(s)
        report.loc[col, "kurtosis"] = safe_kurtosis(s)

    return report.sort_values(["missing_rate_pct", "dtype"], ascending=[False, True])


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Small summary table for the whole dataset.
    """
    summary = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_numeric_columns": len(get_numeric_columns(df)),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_row_rate_pct": round(df.duplicated().mean() * 100, 4),
        "any_missing": bool(df.isna().any().any()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "total_missing_rate_pct": round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 4),
    }
    return pd.DataFrame([summary])


def build_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_rate_pct": (df.isna().mean() * 100).round(2).values
    })
    return report.sort_values(["missing_rate_pct", "missing_count"], ascending=[False, False]).reset_index(drop=True)


def build_duplicate_report(df: pd.DataFrame) -> dict:
    dup_mask = df.duplicated(keep=False)
    duplicate_rows_df = df.loc[dup_mask].copy()
    return {
        "duplicate_count": int(df.duplicated().sum()),
        "duplicate_rate_pct": round(df.duplicated().mean() * 100, 4),
        "duplicate_rows_preview": duplicate_rows_df.head(10)
    }


# ------------------------------------------------------------
# 4. Logical audit
# ------------------------------------------------------------
def audit_frequency_constraints(df_subset: pd.DataFrame):
    rows = []

    for col in df_subset.columns:
        s = df_subset[col]

        invalid_low = int((s < 0).sum(skipna=True))
        invalid_high = int((s > 1).sum(skipna=True))
        total = len(s)

        rows.append({
            "column": col,
            "rule": "0 <= value <= 1",
            "invalid_low_count": invalid_low,
            "invalid_high_count": invalid_high,
            "invalid_total_count": invalid_low + invalid_high,
            "invalid_rate_pct": round((invalid_low + invalid_high) / total * 100, 4),
            "min_value": s.min(skipna=True),
            "max_value": s.max(skipna=True),
        })

    return pd.DataFrame(rows).sort_values("invalid_total_count", ascending=False)



def audit_nonnegative_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monetary/count-like variables should generally be >= 0.
    """
    rows = []
    for col in infer_nonnegative_columns(df):
        s = df[col]
        invalid_neg = int((s < 0).sum(skipna=True))
        rows.append({
            "column": col,
            "rule": "value >= 0",
            "invalid_negative_count": invalid_neg,
            "invalid_rate_pct": round(invalid_neg / len(df) * 100, 4),
            "min_value": s.min(skipna=True),
        })

    result = pd.DataFrame(rows)
    return result[result["invalid_negative_count"] > 0].sort_values(
        "invalid_negative_count", ascending=False
    ).reset_index(drop=True)


def audit_soft_positive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Soft check only. Zero does not necessarily mean invalid, but it matters
    for later feature engineering such as LIMIT_USAGE or monthly averages.
    """
    rows = []
    for col in infer_positive_expected_columns(df):
        s = df[col]
        zero_count = int((s == 0).sum(skipna=True))
        negative_count = int((s < 0).sum(skipna=True))
        rows.append({
            "column": col,
            "zero_count": zero_count,
            "negative_count": negative_count,
            "zero_rate_pct": round(zero_count / len(df) * 100, 4),
            "negative_rate_pct": round(negative_count / len(df) * 100, 4),
            "min_value": s.min(skipna=True),
            "median_value": s.median(skipna=True),
        })

    return pd.DataFrame(rows)

def audit_purchases_decomposition(df: pd.DataFrame, tol: float = 1.0) -> pd.DataFrame:
    """
    Check: PURCHASES == ONEOFF_PURCHASES + INSTALLMENTS_PURCHASES (within tol).
 
    Returns a one-row summary DataFrame so it fits naturally into
    run_phase_2_data_audit alongside the other audit tables.
    """
    required = ["PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return pd.DataFrame([{
            "check": "purchases_decomposition",
            "status": "skipped",
            "reason": f"missing columns: {missing_cols}",
            "violation_count": np.nan,
            "violation_rate_pct": np.nan,
            "max_abs_diff": np.nan,
        }])
 
    reconstructed = df["ONEOFF_PURCHASES"] + df["INSTALLMENTS_PURCHASES"]
    diff = (df["PURCHASES"] - reconstructed).abs()
 
    # Only rows where all three columns are non-null
    valid_mask = df[required].notna().all(axis=1)
    diff_valid = diff[valid_mask]
 
    violations = (diff_valid > tol).sum()
    total_valid = valid_mask.sum()
 
    return pd.DataFrame([{
        "check": "purchases_decomposition",
        "status": "ok" if violations == 0 else "VIOLATIONS FOUND",
        "reason": "PURCHASES == ONEOFF + INSTALLMENTS",
        "violation_count": int(violations),
        "violation_rate_pct": round(violations / total_valid * 100, 4) if total_valid > 0 else np.nan,
        "max_abs_diff": round(diff_valid.max(), 4),
    }])



# ------------------------------------------------------------
# 5. Distribution and outlier audit
# ------------------------------------------------------------
def build_distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize numeric distribution shape and IQR-based outlier burden.
    """
    rows = []
    for col in get_numeric_columns(df):
        s = df[col].dropna()
        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (s < lower) | (s > upper)

        rows.append({
            "column": col,
            "skewness": round(safe_skew(s), 4),
            "kurtosis": round(safe_kurtosis(s), 4),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count_iqr": int(outlier_mask.sum()),
            "outlier_rate_iqr_pct": round(outlier_mask.mean() * 100, 2),
        })

    report = pd.DataFrame(rows)

    def shape_label(x: float) -> str:
        if pd.isna(x):
            return "unknown"
        if x > 1:
            return "high_right_skew"
        if x > 0.5:
            return "moderate_right_skew"
        if x < -1:
            return "high_left_skew"
        if x < -0.5:
            return "moderate_left_skew"
        return "roughly_symmetric"

    report["shape_label"] = report["skewness"].apply(shape_label)
    return report.sort_values(["outlier_rate_iqr_pct", "skewness"], ascending=[False, False]).reset_index(drop=True)


def suggest_log1p_candidates(distribution_report: pd.DataFrame, skew_threshold: float = 1.0) -> list[str]:
    """
    Conservative suggestion list for later preprocessing.
    This is only an audit output. It does not transform anything yet.
    """
    candidates = distribution_report.loc[
        distribution_report["skewness"] >= skew_threshold, "column"
    ].tolist()

    # In this dataset, log1p is usually more defensible for monetary totals than for bounded ratios.
    exclude_keywords = ["FREQUENCY", "RATIO", "PRC_"]
    filtered = []
    for col in candidates:
        upper_col = col.upper()
        if any(k in upper_col for k in exclude_keywords):
            continue
        filtered.append(col)
    return filtered


# ------------------------------------------------------------
# 6. Optional plotting utilities
# ------------------------------------------------------------
def plot_numeric_histograms(df: pd.DataFrame, columns: list[str] | None = None, n_cols: int = 3, bins: int = 30) -> None:
    if columns is None:
        columns = get_numeric_columns(df)

    if not columns:
        print("No numeric columns to plot.")
        return

    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columns):
        ax = axes[i]
        s = df[col].dropna()
        ax.hist(s, bins=bins)
        ax.set_title(f"{col}\nSkew={safe_skew(s):.2f}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame, columns: list[str] | None = None, n_cols: int = 3) -> None:
    if columns is None:
        columns = get_numeric_columns(df)

    if not columns:
        print("No numeric columns to plot.")
        return

    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.boxplot(df[col].dropna(), vert=True)
        ax.set_title(col)
        ax.set_xticks([])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_missingness_bar(missing_report: pd.DataFrame, top_n: int = 20) -> None:
    plot_df = missing_report.head(top_n).copy()
    if plot_df.empty:
        print("No missingness to plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["column"], plot_df["missing_rate_pct"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Missing rate (%)")
    plt.title(f"Top {min(top_n, len(plot_df))} columns by missingness")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 7. Main audit runner
# ------------------------------------------------------------
def run_phase_2_data_audit(
    df: pd.DataFrame,
    id_columns: list[str] | None = None,
    frequency_columns: list[str] | None = None,
    nonnegative_columns: list[str] | None = None,
    positive_columns: list[str] | None = None,
    purchases_decomp_tol: float = 1.0,
    show_plots: bool = True
):
    validate_input_dataframe(df)

    audit_df = df.copy()

    # -------------------------------
    # Handle default parameters
    # -------------------------------
    if id_columns is None:
        id_columns = []
    id_columns = [c for c in id_columns if c in audit_df.columns]

    if frequency_columns is None:
        frequency_columns = [c for c in audit_df.columns if "FREQUENCY" in c.upper()]
        if "PRC_FULL_PAYMENT" in audit_df.columns:
            frequency_columns.append("PRC_FULL_PAYMENT")

    if nonnegative_columns is None:
        nonnegative_columns = audit_df.select_dtypes(include=[np.number]).columns.tolist()

    if positive_columns is None:
        positive_columns = [c for c in ["TENURE", "CREDIT_LIMIT"] if c in audit_df.columns]

    # -------------------------------
    # Reports
    # -------------------------------
    numeric_cols_for_plots = [c for c in get_numeric_columns(audit_df) if c not in id_columns]

    dataset_summary = build_dataset_summary(audit_df)
    schema_report = build_schema_report(audit_df)
    missingness_report = build_missingness_report(audit_df)
    duplicate_report = build_duplicate_report(audit_df)

    frequency_audit = audit_frequency_constraints(audit_df[frequency_columns])
    nonnegative_audit = audit_nonnegative_constraints(audit_df[nonnegative_columns])
    positive_soft_audit = audit_soft_positive_columns(audit_df[positive_columns])
    purchases_decomp_audit = audit_purchases_decomposition(audit_df)

    distribution_report = build_distribution_report(
        audit_df[[c for c in audit_df.columns if c not in id_columns]]
    )

    log1p_candidates = suggest_log1p_candidates(distribution_report, skew_threshold=1.0)

    audit_results = {
        "dataset_summary": dataset_summary,
        "schema_report": schema_report,
        "missingness_report": missingness_report,
        "duplicate_report": duplicate_report,
        "frequency_audit": frequency_audit,
        "nonnegative_audit": nonnegative_audit,
        "positive_soft_audit": positive_soft_audit,
        "purchases_decomposition_audit": purchases_decomp_audit,
        "distribution_report": distribution_report,
        "log1p_candidates": log1p_candidates
        
    }

    # -------------------------------
    # Print results
    # -------------------------------
    print("=" * 80)
    print("PHASE 2. DATA AUDIT")
    print("=" * 80)

    print("\n[Dataset Summary]")
    print(dataset_summary.to_string(index=False))

    print("\n[Top Missing Columns]")
    print(missingness_report.head(15).to_string(index=False))

    print("\n[Frequency Constraint Audit]")
    print(frequency_audit.to_string(index=False))

    print("\n[Nonnegative Constraint Violations]")
    print(nonnegative_audit.to_string(index=False))

    print("\n[Soft Positive Columns Audit]")
    print(positive_soft_audit.to_string(index=False))

    print("\n[Purchases Decomposition Audit]")
    print(purchases_decomp_audit.to_string(index=False))

    print("\n[Most Skewed / Outlier-Heavy Features]")
    print(distribution_report.head(15).to_string(index=False))



    print("\n[Duplicate Rows]")
    print(f"duplicate_count = {duplicate_report['duplicate_count']}")
    print(f"duplicate_rate_pct = {duplicate_report['duplicate_rate_pct']}")

    if show_plots:
        plot_missingness_bar(missingness_report, top_n=20)
        plot_numeric_histograms(audit_df, columns=numeric_cols_for_plots)
        plot_boxplots(audit_df, columns=numeric_cols_for_plots)

    return audit_results



audit_results = run_phase_2_data_audit(
    df=df,
    id_columns=["CUST_ID"],
    show_plots=True
)

schema_report = audit_results["schema_report"]
missingness_report = audit_results["missingness_report"]
distribution_report = audit_results["distribution_report"]




def phase_3_clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --------------------------------------------------
    # 1. Drop ID
    # --------------------------------------------------
    if "CUST_ID" in df.columns:
        df = df.drop(columns=["CUST_ID"])

    # --------------------------------------------------
    # 2. Handle invalid frequency values
    # (convert out-of-range to NaN, not drop rows)
    # --------------------------------------------------
    freq_cols = [c for c in df.columns if "FREQUENCY" in c.upper()]
    if "PRC_FULL_PAYMENT" in df.columns:
        freq_cols.append("PRC_FULL_PAYMENT")

    for col in set(freq_cols):
        if col in df.columns:
            mask_invalid = (~df[col].isna()) & ((df[col] < 0) | (df[col] > 1))
            df.loc[mask_invalid, col] = np.nan

    # --------------------------------------------------
    # 3. Impute missing values (ONLY where needed)
    # --------------------------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns
    missing_cols = [c for c in num_cols if df[c].isna().any()]

    if missing_cols:
        imputer = SimpleImputer(strategy="median")
        df[missing_cols] = imputer.fit_transform(df[missing_cols])

    # --------------------------------------------------
    # 4. Safe denominators (important fix)
    # --------------------------------------------------
    safe_tenure = df["TENURE"].replace(0, np.nan) if "TENURE" in df.columns else np.nan
    safe_credit_limit = df["CREDIT_LIMIT"].replace(0, np.nan) if "CREDIT_LIMIT" in df.columns else np.nan
    safe_min_pay = df["MINIMUM_PAYMENTS"].replace(0, np.nan) if "MINIMUM_PAYMENTS" in df.columns else np.nan

    purchases = df["PURCHASES"] if "PURCHASES" in df.columns else 0
    cash_advance = df["CASH_ADVANCE"] if "CASH_ADVANCE" in df.columns else 0
    installments = df["INSTALLMENTS_PURCHASES"] if "INSTALLMENTS_PURCHASES" in df.columns else 0
    payments = df["PAYMENTS"] if "PAYMENTS" in df.columns else 0
    balance = df["BALANCE"] if "BALANCE" in df.columns else 0

    total_spend = purchases + cash_advance

    # --------------------------------------------------
    # 5. Feature engineering
    # --------------------------------------------------
    if "TENURE" in df.columns:
        df["MONTHLY_AVG_PURCHASES"] = (purchases / safe_tenure).fillna(0)
        df["MONTHLY_AVG_CASH_ADVANCE"] = (cash_advance / safe_tenure).fillna(0)

    if "PURCHASES" in df.columns and "CASH_ADVANCE" in df.columns:
        df["CASH_ADVANCE_RATIO"] = np.where(total_spend == 0, 0, cash_advance / total_spend)

    if "BALANCE" in df.columns and "CREDIT_LIMIT" in df.columns:
        df["LIMIT_USAGE"] = (balance / safe_credit_limit).replace([np.inf, -np.inf], np.nan).fillna(0)

    if "PAYMENTS" in df.columns and "MINIMUM_PAYMENTS" in df.columns:
        df["PAYMENT_MIN_RATIO"] = (payments / safe_min_pay).replace([np.inf, -np.inf], np.nan).fillna(0)

    if "INSTALLMENTS_PURCHASES" in df.columns and "PURCHASES" in df.columns:
        df["INSTALLMENT_PURCHASE_RATIO"] = np.where(purchases == 0, 0, installments / purchases)

    # --------------------------------------------------
    # 6. Stability fix (IMPORTANT)
    # Cap extreme ratios (based on data, not arbitrary)
    # --------------------------------------------------
    for col in ["PAYMENT_MIN_RATIO", "LIMIT_USAGE"]:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)

    return df



df_phase3 = phase_3_clean_and_engineer(df)
df_phase3.head()





def phase_3b_drop_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.7
):
    df = df.copy()
 
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["TENURE"]
    candidate_cols = [c for c in numeric_cols if c not in exclude_cols]
 
    X = df[candidate_cols].copy()
 
    # ── PATCH: Spearman instead of Pearson ───────────────────
    # spearmanr returns (correlation_matrix, pvalue_matrix) when
    # given a 2-D array; we only need the correlation matrix.
    corr_values, _ = spearmanr(X.values)
    corr_matrix = pd.DataFrame(
        np.abs(corr_values),
        index=candidate_cols,
        columns=candidate_cols
    )
    # ─────────────────────────────────────────────────────────
 
    graph = {col: set() for col in candidate_cols}
    for i in range(len(candidate_cols)):
        for j in range(i + 1, len(candidate_cols)):
            col_i = candidate_cols[i]
            col_j = candidate_cols[j]
            if corr_matrix.loc[col_i, col_j] > threshold:
                graph[col_i].add(col_j)
                graph[col_j].add(col_i)
 
    visited = set()
    components = []
    for col in candidate_cols:
        if col not in visited:
            stack = [col]
            component = set()
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.add(node)
                    stack.extend(graph[node] - visited)
            components.append(component)
 
    priority = [
        "LIMIT_USAGE", "PAYMENT_MIN_RATIO", "CASH_ADVANCE_RATIO",
        "INSTALLMENT_PURCHASE_RATIO", "MONTHLY_AVG_PURCHASES",
        "MONTHLY_AVG_CASH_ADVANCE", "PURCHASES_FREQUENCY",
        "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT", "BALANCE",
        "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
        "CASH_ADVANCE", "PAYMENTS", "MINIMUM_PAYMENTS", "CREDIT_LIMIT",
        "PURCHASES_TRX", "CASH_ADVANCE_TRX"
    ]
    rank = {col: i for i, col in enumerate(priority)}
 
    kept_cols = []
    dropped_info = []
    for comp in components:
        comp = list(comp)
        if len(comp) == 1:
            kept_cols.append(comp[0])
            continue
        best = min(comp, key=lambda c: rank.get(c, 10**6))
        kept_cols.append(best)
        for col in comp:
            if col != best:
                dropped_info.append({
                    "kept_feature": best,
                    "dropped_feature": col,
                    "group_size": len(comp)
                })
 
    df_reduced = df[kept_cols].copy()
    drop_log = pd.DataFrame(dropped_info)
 
    return df_reduced, kept_cols, drop_log



df_phase3_reduced, kept_cols, drop_log = phase_3b_drop_correlated_features(df_phase3, threshold=0.7)

print("Final feature count:", len(kept_cols))
drop_log.head()







def phase_4_prepare_matrix(
    df: pd.DataFrame,
    feature_cols: list[str]
):
    df = df.copy()

    # -----------------------------------
    # 1. Keep only selected features
    # -----------------------------------
    X = df[feature_cols].copy()

    # -----------------------------------
    # 2. Define log-transform columns
    # (only if they survived Phase 3b)
    # -----------------------------------
    log_candidates = [
        "BALANCE",
        "PURCHASES",
        "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES",
        "CASH_ADVANCE",
        "CREDIT_LIMIT",
        "PAYMENTS",
        "MINIMUM_PAYMENTS",
        "MONTHLY_AVG_PURCHASES",
        "MONTHLY_AVG_CASH_ADVANCE",
        "PAYMENT_MIN_RATIO",        # capped in Phase 3
        "PURCHASES_TRX",
        "CASH_ADVANCE_TRX"
    ]

    log_cols = [c for c in log_candidates if c in X.columns]

    # -----------------------------------
    # 3. Apply log1p safely
    # -----------------------------------
    if log_cols:
        X[log_cols] = np.log1p(X[log_cols].clip(lower=0))

    # -----------------------------------
    # 4. Standard scaling
    # -----------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=feature_cols,
        index=df.index
    )

    return X_scaled, feature_cols, log_cols, scaler


X_scaled, feature_cols, log_cols, scaler = phase_4_prepare_matrix( df_phase3_reduced, kept_cols)
X_scaled.head()




def _cluster_metrics(X, labels):
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}

    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


def phase_5a_scan_models(X_scaled, k_range=range(2, 9), random_state=42):
    kmeans_rows = []
    gmm_rows = []

    for k in k_range:
        # --- KMeans ---
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels_km = km.fit_predict(X_scaled)
        metrics_km = _cluster_metrics(X_scaled, labels_km)

        kmeans_rows.append({
            "k": k,
            "inertia": km.inertia_,
            **metrics_km
        })

        # --- GMM ---
        for cov in ["full", "diag", "tied", "spherical"]:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                n_init=10,
                random_state=random_state
            )
            labels_gmm = gmm.fit_predict(X_scaled)
            metrics_gmm = _cluster_metrics(X_scaled, labels_gmm)

            gmm_rows.append({
                "k": k,
                "covariance_type": cov,
                "bic": gmm.bic(X_scaled),
                "aic": gmm.aic(X_scaled),
                **metrics_gmm
            })

    return pd.DataFrame(kmeans_rows), pd.DataFrame(gmm_rows)



def phase_5b_summarize_candidates(
    kmeans_results: pd.DataFrame,
    gmm_results: pd.DataFrame,
    top_n: int = 5
) -> tuple[pd.DataFrame, int, str]:
    """
    Returns
    -------
    ranking_df   : wide table showing rank of each k per metric
    best_k       : recommended k (most consistent across metrics)
    best_cov     : best GMM covariance type for that k
    """
 
    # ── 1. KMeans rankings ───────────────────────────────────
    km = kmeans_results.copy()
 
    # Higher silhouette = better → rank ascending=False
    km["rank_silhouette"] = km["silhouette"].rank(ascending=False, method="min").astype(int)
    # Lower davies_bouldin = better → rank ascending=True
    km["rank_davies"]     = km["davies_bouldin"].rank(ascending=True,  method="min").astype(int)
    # Higher calinski = better → rank ascending=False
    km["rank_calinski"]   = km["calinski_harabasz"].rank(ascending=False, method="min").astype(int)
 
    km["kmeans_avg_rank"] = (
        km["rank_silhouette"] + km["rank_davies"] + km["rank_calinski"]
    ) / 3.0
 
    kmeans_ranking = (
        km[["k", "silhouette", "davies_bouldin", "calinski_harabasz",
            "rank_silhouette", "rank_davies", "rank_calinski", "kmeans_avg_rank"]]
        .sort_values("kmeans_avg_rank")
        .head(top_n)
        .reset_index(drop=True)
    )
 
    # ── 2. GMM rankings (all cov types pooled) ───────────────
    gm = gmm_results.copy()
    gm["rank_bic"]       = gm["bic"].rank(ascending=True,  method="min").astype(int)
    gm["rank_aic"]       = gm["aic"].rank(ascending=True,  method="min").astype(int)
    gm["rank_sil_gmm"]   = gm["silhouette"].rank(ascending=False, method="min").astype(int)
 
    gm["gmm_avg_rank"] = (
        gm["rank_bic"] + gm["rank_aic"] + gm["rank_sil_gmm"]
    ) / 3.0
 
    gmm_ranking = (
        gm[["k", "covariance_type", "bic", "aic", "silhouette",
            "rank_bic", "rank_aic", "rank_sil_gmm", "gmm_avg_rank"]]
        .sort_values("gmm_avg_rank")
        .head(top_n)
        .reset_index(drop=True)
    )
 
    # ── 3. Derive recommended k ──────────────────────────────
    # Strategy: k that appears most often in top-3 of KMeans
    # across the three metrics.  Ties broken by avg_rank.
    top3_per_metric = pd.concat([
        km.nsmallest(3, "rank_silhouette")[["k"]],
        km.nsmallest(3, "rank_davies")[["k"]],
        km.nsmallest(3, "rank_calinski")[["k"]],
    ])
    vote_counts = top3_per_metric["k"].value_counts()
    # Among k values with the most votes, take the one with
    # lowest kmeans_avg_rank as tiebreaker.
    top_vote = vote_counts[vote_counts == vote_counts.max()].index.tolist()
    best_k = int(
        km[km["k"].isin(top_vote)]
        .sort_values("kmeans_avg_rank")
        .iloc[0]["k"]
    )
 
    # ── 4. Best GMM covariance type for best_k ───────────────
    gm_at_best_k = gm[gm["k"] == best_k].sort_values("gmm_avg_rank")
    best_cov = str(gm_at_best_k.iloc[0]["covariance_type"]) if len(gm_at_best_k) > 0 else "full"
 
    # ── 5. Print ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL SELECTION — KMeans ranking (top {})".format(top_n))
    print("=" * 70)
    print(kmeans_ranking.round(4).to_string(index=False))
 
    print("\n" + "=" * 70)
    print("MODEL SELECTION — GMM ranking (top {}, all cov types)".format(top_n))
    print("=" * 70)
    print(gmm_ranking.round(4).to_string(index=False))
 
    print("\n>>> Recommended k = {}  (most votes in top-3 across KMeans metrics)".format(best_k))
    print(">>> Best GMM covariance for k={}: {}".format(best_k, best_cov))
 
    return kmeans_ranking, gmm_ranking, best_k, best_cov
    

def phase_5c_fit_final_model(
    X_scaled,
    model_type="kmeans",
    k=3,
    covariance_type="full",
    random_state=42
):
    if model_type == "kmeans":
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(X_scaled)

    elif model_type == "gmm":
        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=10,
            random_state=random_state
        )
        labels = model.fit_predict(X_scaled)

    else:
        raise ValueError("model_type must be 'kmeans' or 'gmm'")

    metrics = _cluster_metrics(X_scaled, labels)

    print("\n=== FINAL MODEL ===")
    print(f"Model: {model_type}, k={k}, cov={covariance_type if model_type=='gmm' else '-'}")
    print(metrics)

    return model, labels, metrics





def phase_5d_stability_kmeans(X_scaled, k, n_runs=10):
    labels_list = []

    for i in range(n_runs):
        km = KMeans(n_clusters=k, n_init=10, random_state=42 + i)
        labels = km.fit_predict(X_scaled)
        labels_list.append(labels)

    scores = []

    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            ari = adjusted_rand_score(labels_list[i], labels_list[j])
            scores.append(ari)

    stability = {
        "mean_ari": np.mean(scores),
        "std_ari": np.std(scores)
    }

    print("\n=== STABILITY (KMeans) ===")
    print(stability)

    return stability

kmeans_results, gmm_results = phase_5a_scan_models(X_scaled)

kmeans_ranking, gmm_ranking, best_k, best_cov = phase_5b_summarize_candidates(kmeans_results, gmm_results)

final_model, final_labels, final_metrics = phase_5c_fit_final_model( X_scaled, model_type="kmeans", k=best_k)

stability = phase_5d_stability_kmeans(X_scaled, k=3, n_runs=10)




def phase_6_validate_and_profile(
    df_for_profiling: pd.DataFrame,
    X_for_clustering,
    labels,
    feature_cols
):
    labels = np.asarray(labels)

    # -----------------------------------
    # 1. Cluster validation (silhouette)
    # -----------------------------------
    sil_samples = silhouette_samples(X_for_clustering, labels)

    sil_df = pd.DataFrame({
        "Cluster": labels,
        "silhouette_sample": sil_samples
    })

    cluster_validation = sil_df.groupby("Cluster").agg(
        cluster_size=("Cluster", "size"),
        silhouette_mean=("silhouette_sample", "mean"),
        silhouette_min=("silhouette_sample", "min"),
        silhouette_std=("silhouette_sample", "std")
    ).reset_index()

    # -----------------------------------
    # 2. Attach labels to full dataset
    # -----------------------------------
    df_labeled = df_for_profiling.copy()
    df_labeled["Cluster"] = labels

    # -----------------------------------
    # 3. Build cluster profile
    # (use original scale, not scaled data)
    # -----------------------------------
    profile_features = [c for c in feature_cols if c in df_labeled.columns]

    cluster_profile = df_labeled.groupby("Cluster")[profile_features].agg(["mean", "median"])
    cluster_profile.columns = [f"{col}_{stat}" for col, stat in cluster_profile.columns]
    cluster_profile = cluster_profile.reset_index()

    # -----------------------------------
    # 4. Population %
    # -----------------------------------
    cluster_pct = df_labeled["Cluster"].value_counts(normalize=True).sort_index() * 100
    cluster_profile["population_pct"] = cluster_profile["Cluster"].map(cluster_pct)

    return df_labeled, cluster_validation, cluster_profile


def phase_6_name_segments(cluster_profile: pd.DataFrame):
    df = cluster_profile.copy()
 
    metrics = {
        "spend":        "MONTHLY_AVG_PURCHASES_mean",
        "cash_volume":  "MONTHLY_AVG_CASH_ADVANCE_mean",
        "payment":      "PRC_FULL_PAYMENT_mean",
        "usage":        "LIMIT_USAGE_mean",
    }
    metrics = {k: v for k, v in metrics.items() if v in df.columns}
 
    # ── z-score each metric across clusters ──────────────────
    HIGH = 0.75    # z-score threshold to call a dimension "high"
    LOW  = -0.75   # z-score threshold to call a dimension "low"
 
    for key, col in metrics.items():
        mu  = df[col].mean()
        std = df[col].std(ddof=0)          # population std (few clusters)
        if std < 1e-9:
            df[f"z_{key}"] = 0.0           # constant across clusters → neutral
        else:
            df[f"z_{key}"] = (df[col] - mu) / std
 
    segment_names = {}
    for _, row in df.iterrows():
        c = row["Cluster"]
 
        z_spend   = row.get("z_spend",       0.0)
        z_cash    = row.get("z_cash_volume",  0.0)
        z_payment = row.get("z_payment",      0.0)
        z_usage   = row.get("z_usage",        0.0)
 
        # ── Decision rules (ordered by specificity) ──────────
        if z_cash >= HIGH:
            name = "Cash Advance Heavy"
 
        elif z_spend >= HIGH and z_payment >= HIGH:
            name = "Active Transactor"
 
        elif z_usage >= HIGH and z_payment <= LOW:
            name = "Revolving Credit User"
 
        elif z_spend <= LOW and z_usage <= LOW:
            name = "Low Engagement"
 
        else:
            name = "Mixed Behavior"
 
        segment_names[c] = name
 
    df["Segment_Name"] = df["Cluster"].map(segment_names)
 
    # ── drop helper z-score columns before returning ─────────
    z_cols = [c for c in df.columns if c.startswith("z_")]
    df = df.drop(columns=z_cols)
 
    return df, segment_names




def phase_6_attach_segments(df_labeled: pd.DataFrame, segment_names: dict):
    df_out = df_labeled.copy()
    df_out["Segment_Name"] = df_out["Cluster"].map(segment_names)
    return df_out

df_labeled, cluster_validation, cluster_profile = phase_6_validate_and_profile(
    df_for_profiling=df_phase3,        
    X_for_clustering=X_scaled,
    labels=final_labels,
    feature_cols=feature_cols
)

cluster_profile_named, segment_names = phase_6_name_segments(cluster_profile)

df_final = phase_6_attach_segments(df_labeled, segment_names)






def plot_phase_7_model_metrics(kmeans_results, gmm_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # KMeans
    axes[0, 0].plot(kmeans_results["k"], kmeans_results["inertia"], marker="o")
    axes[0, 0].set_title("KMeans Inertia")

    axes[0, 1].plot(kmeans_results["k"], kmeans_results["silhouette"], marker="o")
    axes[0, 1].set_title("KMeans Silhouette")

    # GMM
    for cov in gmm_results["covariance_type"].unique():
        sub = gmm_results[gmm_results["covariance_type"] == cov]

        axes[1, 0].plot(sub["k"], sub["bic"], marker="o", label=cov)
        axes[1, 1].plot(sub["k"], sub["aic"], marker="o", label=cov)

    axes[1, 0].set_title("GMM BIC")
    axes[1, 1].set_title("GMM AIC")

    for ax in axes.ravel():
        ax.set_xlabel("k")

    axes[1, 0].legend()
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()



def plot_phase_7_pca_clusters(X_scaled, labels):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        s=25,
        alpha=0.7
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Cluster Visualization (PCA Projection)")

    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()

    explained_var = pca.explained_variance_ratio_
    print("Explained variance:", explained_var)

def plot_phase_7_cluster_sizes(cluster_validation):
    plt.figure(figsize=(7, 5))

    plt.bar(
        cluster_validation["Cluster"].astype(str),
        cluster_validation["cluster_size"]
    )

    plt.xlabel("Cluster")
    plt.ylabel("Size")
    plt.title("Cluster Sizes")

    plt.tight_layout()
    plt.show()





def plot_phase_7_profile_heatmap(cluster_profile_named):
    df = cluster_profile_named.copy()

    # Keep only mean features
    feature_cols = [c for c in df.columns if c.endswith("_mean")]

    heatmap_df = df.set_index("Cluster")[feature_cols]

    # -----------------------------------
    # COLUMN-WISE MIN-MAX NORMALIZATION
    # -----------------------------------
    norm_df = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min() + 1e-9)

    data = norm_df.values

    fig, ax = plt.subplots(figsize=(max(10, len(feature_cols) * 0.6), 5))
    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(norm_df.index)))
    ax.set_yticklabels(norm_df.index.astype(str))

    # annotate values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title("Cluster Profile (Normalized Heatmap)")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

def display_cluster_profile_table(cluster_profile_named):
    print("\n=== RAW CLUSTER PROFILE (UNSCALED) ===\n")
    print(cluster_profile_named.to_string(index=False))


def run_phase_7_visuals(
    kmeans_results,
    gmm_results,
    X_scaled,
    final_labels,
    cluster_validation,
    cluster_profile_named
):
    plot_phase_7_model_metrics(kmeans_results, gmm_results)

    plot_phase_7_pca_clusters(X_scaled, final_labels)

    plot_phase_7_cluster_sizes(cluster_validation)

    plot_phase_7_profile_heatmap(cluster_profile_named)

    display_cluster_profile_table(cluster_profile_named)


run_phase_7_visuals(
    kmeans_results,
    gmm_results,
    X_scaled,
    final_labels,
    cluster_validation,
    cluster_profile_named
)