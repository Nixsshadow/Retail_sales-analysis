"""Clean and profile the Sample Superstore retail-sales dataset.

Input:  data/raw/sample_superstore.csv
Output: data/processed/retail_sales_clean.csv
        reports/data_quality_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INPUT_FILE = next(
    (path for path in [RAW_DATA_DIR / "sample_superstore.csv", RAW_DATA_DIR / "Superstore.csv"] if path.exists()),
    RAW_DATA_DIR / "sample_superstore.csv",
)
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "retail_sales_clean.csv"
SUMMARY_FILE = PROJECT_ROOT / "reports" / "data_quality_summary.json"

REQUIRED_COLUMNS = {
    "Row ID", "Order ID", "Order Date", "Ship Date", "Customer ID",
    "Segment", "Region", "Product ID", "Category", "Sub-Category",
    "Sales", "Quantity", "Discount", "Profit",
}


def to_snake_case(column: str) -> str:
    return column.strip().lower().replace("-", "_").replace(" ", "_")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Download Sample Superstore CSV and save it with this exact name."
        )

    raw = pd.read_csv(INPUT_FILE, encoding="latin-1")
    missing_columns = REQUIRED_COLUMNS - set(raw.columns)
    if missing_columns:
        raise ValueError(f"The dataset is missing required columns: {sorted(missing_columns)}")

    df = raw.copy()
    df.columns = [to_snake_case(column) for column in df.columns]

    # Make types explicit before calculating business metrics.
    for date_column in ["order_date", "ship_date"]:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce", dayfirst=False)
    for numeric_column in ["sales", "quantity", "discount", "profit"]:
        df[numeric_column] = pd.to_numeric(df[numeric_column], errors="coerce")

    initial_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Derived fields used consistently in SQL, Power BI, and Tableau.
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_month_name"] = df["order_date"].dt.month_name()
    df["order_year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["shipping_days"] = (df["ship_date"] - df["order_date"]).dt.days
    df["profit_margin"] = np.where(df["sales"] != 0, df["profit"] / df["sales"], np.nan)
    df["discount_band"] = pd.cut(
        df["discount"],
        bins=[-0.01, 0, 0.10, 0.20, 0.40, 1.00],
        labels=["No discount", "1-10%", "11-20%", "21-40%", "41%+"],
    )
    df["is_loss_making"] = df["profit"] < 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    summary = {
        "source_rows": initial_rows,
        "rows_after_duplicate_removal": len(df),
        "duplicate_rows_removed": duplicate_rows,
        "columns": len(df.columns),
        "missing_values_by_column": {
            column: int(count) for column, count in df.isna().sum().items() if count
        },
        "date_range": {
            "first_order_date": str(df["order_date"].min().date()),
            "last_order_date": str(df["order_date"].max().date()),
        },
        "total_sales": round(float(df["sales"].sum()), 2),
        "total_profit": round(float(df["profit"].sum()), 2),
        "profit_margin": round(float(df["profit"].sum() / df["sales"].sum()), 4),
        "loss_making_rows": int(df["is_loss_making"].sum()),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Cleaned dataset saved to: {OUTPUT_FILE}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
