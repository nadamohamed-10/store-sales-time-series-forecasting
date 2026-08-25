from __future__ import annotations

import argparse
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm  # noqa: F401  (kept for users who want ACF/PACF diagnostics)
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("store_sales_pipeline")

REQUIRED_FILES = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
]

FEATURE_COLS = [
    "store_nbr", "onpromotion", "is_holiday", "oil_price", "cluster",
    "year", "month", "day", "dayofweek", "weekofyear", "is_weekend",
    "is_month_start", "is_month_end",
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_mean_28",
    "family_enc", "type_enc", "city_enc", "state_enc",
]


# 1. Extract

def extract(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the raw Kaggle CSVs into a dict of DataFrames."""
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required file(s) in {data_dir}: {missing}"
        )

    datasets = {}
    for filename in REQUIRED_FILES:
        key = filename.replace(".csv", "")
        df = pd.read_csv(data_dir / filename)
        log.info("Loaded %-16s shape=%s", key, df.shape)
        datasets[key] = df
    return datasets


# 2. Transform + Load (SQLite = single source of truth downstream)

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df.drop_duplicates()


def transform(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean, type-cast, and lightly enrich every table."""
    ds = {name: _clean_columns(df) for name, df in datasets.items()}

    train = ds["train"].dropna().copy()
    train["date"] = pd.to_datetime(train["date"])
    train["sales"] = pd.to_numeric(train["sales"])
    train["onpromotion"] = pd.to_numeric(train["onpromotion"])

    test = ds["test"].copy()
    test["date"] = pd.to_datetime(test["date"], errors="coerce")
    test["onpromotion"] = pd.to_numeric(test["onpromotion"], errors="coerce")

    stores = ds["stores"].copy()
    for col in ("city", "state", "type"):
        stores[col] = stores[col].str.strip()

    oil = ds["oil"].copy()
    oil["date"] = pd.to_datetime(oil["date"])
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"])
    oil = oil.sort_values("date")
    # Reindex to a full daily calendar so ffill/bfill has no gaps to skip.
    oil = (
        oil.set_index("date")
        .reindex(pd.date_range(oil["date"].min(), oil["date"].max(), freq="D"))
        .rename_axis("date")
        .reset_index()
    )
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate().ffill().bfill()

    holidays = ds["holidays_events"].copy()
    holidays["date"] = pd.to_datetime(holidays["date"])
    for col in ("type", "locale", "locale_name", "description"):
        holidays[col] = holidays[col].str.strip()
    holidays["transferred"] = holidays["transferred"].astype(bool)

    transactions = ds["transactions"].copy()
    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions["transactions"] = pd.to_numeric(transactions["transactions"])

    cleaned = {
        "train": train,
        "test": test,
        "stores": stores,
        "oil": oil,
        "holidays_events": holidays,
        "transactions": transactions,
    }
    for name, df in cleaned.items():
        log.info("Cleaned %-16s shape=%s", name, df.shape)
    return cleaned


def load_to_sqlite(cleaned: dict[str, pd.DataFrame], db_path: Path) -> sqlite3.Connection:
    """Load cleaned tables into SQLite, the single source of truth for
    everything downstream (feature engineering AND SQL reporting)."""
    conn = sqlite3.connect(db_path)
    for table_name, df in cleaned.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=100_000)
        log.info("Loaded table '%s' into %s", table_name, db_path.name)
    return conn


# 3. Feature engineering

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


@dataclass
class FeatureArtifacts:
    train_fe: pd.DataFrame
    test_fe: pd.DataFrame
    encoders: dict[str, LabelEncoder]
    scaler: StandardScaler


def engineer_features(cleaned: dict[str, pd.DataFrame]) -> FeatureArtifacts:
    train, test = cleaned["train"], cleaned["test"]
    stores, oil, holidays = cleaned["stores"], cleaned["oil"], cleaned["holidays_events"]

    train_fe = add_calendar_features(train)
    test_fe = add_calendar_features(test)

    holiday_dates = set(
        holidays.loc[
            (~holidays["transferred"]) & (holidays["type"] != "Work Day"), "date"
        ]
    )
    oil_lookup = oil.set_index("date")["dcoilwtico"]

    for df in (train_fe, test_fe):
        df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
        df["oil_price"] = df["date"].map(oil_lookup).ffill()

    train_fe = train_fe.merge(stores, on="store_nbr", how="left")
    test_fe = test_fe.merge(stores, on="store_nbr", how="left")

    train_fe = train_fe.sort_values(["store_nbr", "family", "date"])
    group_cols = ["store_nbr", "family"]
    for lag in (7, 14, 28):
        train_fe[f"sales_lag_{lag}"] = train_fe.groupby(group_cols)["sales"].shift(lag)
    for window in (7, 28):
        train_fe[f"sales_roll_mean_{window}"] = (
            train_fe.groupby(group_cols)["sales"]
            .transform(lambda s: s.shift(1).rolling(window).mean())
        )

    n_before = len(train_fe)
    lag_roll_cols = [c for c in train_fe.columns if "lag" in c or "roll" in c]
    train_fe_model = train_fe.dropna(subset=lag_roll_cols).copy()
    log.info("Dropped %d/%d rows with NaN lag/rolling features", n_before - len(train_fe_model), n_before)

    encoders: dict[str, LabelEncoder] = {}
    for col in ("family", "type", "city", "state"):
        le = LabelEncoder()
        all_vals = pd.concat([train_fe_model[col], test_fe[col]]).astype(str).unique()
        le.fit(all_vals)
        train_fe_model[f"{col}_enc"] = le.transform(train_fe_model[col].astype(str))
        test_fe[f"{col}_enc"] = le.transform(test_fe[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    train_fe_model["oil_price_scaled"] = scaler.fit_transform(train_fe_model[["oil_price"]])
    test_fe["oil_price_scaled"] = scaler.transform(test_fe[["oil_price"]])

    return FeatureArtifacts(train_fe_model, test_fe, encoders, scaler)


# 4/5. Modeling + fair evaluation

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


@dataclass
class ModelResults:
    comparison: pd.DataFrame
    lgb_model: lgb.Booster
    feature_importance: pd.DataFrame
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def run_models(
    raw_train: pd.DataFrame,
    train_fe_model: pd.DataFrame,
    validation_days: int = 30,
) -> ModelResults:
    cutoff_date = train_fe_model["date"].max() - pd.Timedelta(days=validation_days)
    tr = train_fe_model[train_fe_model["date"] <= cutoff_date]
    val = train_fe_model[train_fe_model["date"] > cutoff_date]
    log.info(
        "Validation window: %s -> %s (%d rows)",
        val["date"].min().date(), val["date"].max().date(), len(val),
    )

    #  Naive baselines 

    baseline_rmsle = rmsle(val["sales"].values, val["sales_lag_7"].values)
    ma_rmsle = rmsle(val["sales"].values, val["sales_roll_mean_28"].values)

    # SARIMA on the national aggregate 

    daily = raw_train.groupby("date")["sales"].sum()
    daily_train = daily[daily.index <= cutoff_date]
    daily_val = daily[daily.index > cutoff_date]

    sarima_fit = SARIMAX(
        daily_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarima_preds = sarima_fit.get_forecast(steps=len(daily_val)).predicted_mean
    sarima_rmsle_national = rmsle(daily_val.values, sarima_preds.values)

    # LightGBM on the full engineered feature set 

    X_tr, y_tr = tr[FEATURE_COLS], np.log1p(tr["sales"])
    X_val, y_val = val[FEATURE_COLS], np.log1p(val["sales"])

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    lgb_model = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    val_preds_log = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    val_preds = np.expm1(val_preds_log)
    lgb_rmsle_rowlevel = rmsle(val["sales"].values, val_preds)

    # Aggregate LightGBM's row-level predictions to national daily totals
    # so it can be fairly compared against SARIMA (apples-to-apples target).

    lgb_national = (
        val.assign(pred=val_preds)
        .groupby("date")["pred"].sum()
    )
    common_dates = lgb_national.index.intersection(daily_val.index)
    lgb_rmsle_national = rmsle(
        daily_val.loc[common_dates].values, lgb_national.loc[common_dates].values
    )

    comparison = pd.DataFrame(
        [
            {"model": "Naive (lag-7)", "level": "row (store x family)", "rmsle": baseline_rmsle},
            {"model": "28-day moving average", "level": "row (store x family)", "rmsle": ma_rmsle},
            {"model": "LightGBM", "level": "row (store x family)", "rmsle": lgb_rmsle_rowlevel},
            {"model": "SARIMA", "level": "national daily total", "rmsle": sarima_rmsle_national},
            {"model": "LightGBM (aggregated)", "level": "national daily total", "rmsle": lgb_rmsle_national},
        ]
    )
    log.info("\n%s", comparison.to_string(index=False))

    importance = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": lgb_model.feature_importance(importance_type="gain")}
    ).sort_values("importance", ascending=False)

    return ModelResults(
        comparison=comparison,
        lgb_model=lgb_model,
        feature_importance=importance,
        val_start=val["date"].min(),
        val_end=val["date"].max(),
    )


# 6. Reporting

def build_sql_reports(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    reports = {}
    reports["overall"] = pd.read_sql_query(
        """
        SELECT COUNT(*) AS total_records,
               ROUND(SUM(sales), 2) AS total_sales,
               ROUND(AVG(sales), 2) AS average_sales,
               MAX(sales) AS max_sales,
               MIN(sales) AS min_sales
        FROM train;
        """,
        conn,
    )
    reports["by_store"] = pd.read_sql_query(
        """
        SELECT t.store_nbr, s.city, s.state, s.type AS store_type,
               ROUND(SUM(t.sales), 2) AS total_sales,
               ROUND(AVG(t.sales), 2) AS average_sales,
               COUNT(*) AS record_count
        FROM train t JOIN stores s ON t.store_nbr = s.store_nbr
        GROUP BY t.store_nbr, s.city, s.state, s.type
        ORDER BY total_sales DESC;
        """,
        conn,
    )
    reports["by_family"] = pd.read_sql_query(
        """
        SELECT family, ROUND(SUM(sales), 2) AS total_sales,
               ROUND(AVG(sales), 2) AS average_sales, COUNT(*) AS record_count
        FROM train GROUP BY family ORDER BY total_sales DESC;
        """,
        conn,
    )
    reports["by_month"] = pd.read_sql_query(
        """
        SELECT strftime('%Y-%m', date) AS month,
               ROUND(SUM(sales), 2) AS total_sales,
               ROUND(AVG(sales), 2) AS average_sales
        FROM train GROUP BY month ORDER BY month;
        """,
        conn,
    )
    reports["by_promotion"] = pd.read_sql_query(
        """
        SELECT CASE WHEN onpromotion > 0 THEN 'Promoted' ELSE 'Not Promoted' END AS promotion_status,
               ROUND(AVG(sales), 2) AS average_sales,
               ROUND(SUM(sales), 2) AS total_sales,
               COUNT(*) AS record_count
        FROM train GROUP BY promotion_status;
        """,
        conn,
    )
    return reports


def export_reports(reports: dict[str, pd.DataFrame], comparison: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / "store_sales_report.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, df in reports.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    log.info("Reports written to %s", output_dir)


# Orchestration

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with the raw Kaggle CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Where reports/db get written")
    parser.add_argument("--validation-days", type=int, default=30, help="Validation window length in days")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Extract ===")
    raw = extract(args.data_dir)

    log.info("=== Transform ===")
    cleaned = transform(raw)

    log.info("=== Load (SQLite) ===")
    conn = load_to_sqlite(cleaned, args.output_dir / "store_sales_etl.db")

    log.info("=== Feature engineering ===")
    features = engineer_features(cleaned)

    log.info("=== Modeling & evaluation ===")
    results = run_models(cleaned["train"], features.train_fe, validation_days=args.validation_days)

    log.info("=== Reporting ===")
    reports = build_sql_reports(conn)
    export_reports(reports, results.comparison, args.output_dir)

    conn.close()
    log.info("Pipeline complete. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()