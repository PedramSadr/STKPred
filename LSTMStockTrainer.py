import argparse
import os
import numpy as np
import pandas as pd
import duckdb

# --- CONFIGURATION ---
CONTRACT_MULTIPLIER = 100.0  # US equity option standard


def interpolate_linear(curve_df: pd.DataFrame, target_dte: int, col: str) -> float:
    """
    Linear interpolation on a (dte -> value) curve.
    If target is outside range, returns closest endpoint.
    """
    curve_df = curve_df.dropna(subset=["dte", col]).sort_values("dte")
    if curve_df.empty:
        return 0.0

    # Exact match
    exact = curve_df[curve_df["dte"] == target_dte]
    if not exact.empty:
        return float(exact.iloc[0][col])

    # Bracket
    near = curve_df[curve_df["dte"] < target_dte]
    far = curve_df[curve_df["dte"] > target_dte]

    # Out of range => closest
    if near.empty or far.empty:
        idx = (curve_df["dte"] - target_dte).abs().idxmin()
        return float(curve_df.loc[idx, col])

    r1 = near.iloc[-1]
    r2 = far.iloc[0]
    d1, d2 = float(r1["dte"]), float(r2["dte"])
    v1, v2 = float(r1[col]), float(r2[col])

    if d2 == d1:
        return float(v1)

    w = (target_dte - d1) / (d2 - d1)
    return float(v1 + w * (v2 - v1))


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def build_surface_vectors(
        daily_path: str,
        options_path: str,
        out_path: str,
        target_dte: int = 14,
        atm_delta_low: float = 0.40,
        atm_delta_high: float = 0.60,
        skew_delta_low: float = 0.20,
        skew_delta_high: float = 0.30
) -> None:
    print("--- STARTING SURFACE GENERATOR ---")
    print(f"Input Folder: {os.path.dirname(daily_path)}")

    # 1. Validation
    if not os.path.exists(daily_path):
        raise FileNotFoundError(f"Daily file not found: {daily_path}")
    if not os.path.exists(options_path):
        raise FileNotFoundError(f"Options file not found: {options_path}")

    print("1. Loading Raw Files...")
    df_daily = pd.read_csv(daily_path)
    df_opts = pd.read_csv(options_path, low_memory=False)

    # Normalize column names
    df_daily.columns = df_daily.columns.str.strip().str.lower()
    df_opts.columns = df_opts.columns.str.strip().str.lower()

    # Parse dates
    date_col = 'date' if 'date' in df_daily.columns else 'Date'
    if date_col not in df_daily.columns:
        raise ValueError(f"Daily file missing 'date' column. Found: {df_daily.columns.tolist()}")

    if date_col != 'date':
        df_daily.rename(columns={date_col: "date"}, inplace=True)

    _ensure_datetime(df_daily, "date")
    _ensure_datetime(df_opts, "date")
    _ensure_datetime(df_opts, "expiration")

    df_daily = df_daily.dropna(subset=["date"]).sort_values("date")
    df_opts = df_opts.dropna(subset=["date", "expiration"]).sort_values("date")

    # 2. Mapping Price and VIX
    print("2. Mapping Context (Price & VIX)...")
    if "close" not in df_daily.columns:
        raise ValueError("Daily file must contain 'close' column.")

    df_daily = df_daily.set_index("date")
    price_map = df_daily["close"].to_dict()
    vix_map = df_daily["vix"].to_dict() if "vix" in df_daily.columns else {}

    if not vix_map:
        print("   WARNING: 'vix' column not found. 'vol_spread' will be 0.")

    # Prepare options
    df_opts["underlying_price"] = df_opts["date"].map(price_map)
    df_opts = df_opts.dropna(subset=["underlying_price"])

    df_opts["dte"] = (df_opts["expiration"] - df_opts["date"]).dt.days
    df_opts = df_opts[df_opts["dte"] >= 0]

    # Normalize Type
    if "type" in df_opts.columns:
        df_opts["type"] = df_opts["type"].astype(str).str.strip().str.lower()
    else:
        df_opts["type"] = "call"

    # Fill Missing Greeks
    for c in ["delta", "gamma", "vega", "open_interest", "implied_volatility"]:
        if c not in df_opts.columns:
            df_opts[c] = 0.0
        df_opts[c] = pd.to_numeric(df_opts[c], errors="coerce").fillna(0.0)

    # 3. Calculating Physics (Net Exposures)
    print("3. Calculating Net Greek Exposures (GEX, DEX, VEX)...")

    # DEX ($ per point move): delta * OI * 100 * Spot
    df_opts["dex"] = df_opts["delta"] * df_opts["open_interest"] * CONTRACT_MULTIPLIER * df_opts["underlying_price"]

    # VEX ($ per vol point): vega * OI * 100
    df_opts["vex"] = df_opts["vega"] * df_opts["open_interest"] * CONTRACT_MULTIPLIER

    # GEX ($ per 1% move): gamma * OI * 100 * Spot^2
    df_opts["gex"] = (
            df_opts["gamma"] * df_opts["open_interest"] * CONTRACT_MULTIPLIER * (df_opts["underlying_price"] ** 2)
    )

    # 4. Aggregation
    print("4. Aggregating Surface via DuckDB...")
    con = duckdb.connect()
    con.register('df_opts', df_opts)

    query = f"""
    WITH base AS (
        SELECT
            date, dte, type, delta, implied_volatility,
            dex, gex, vex
        FROM df_opts
    ),
    curves AS (
        SELECT
            date, dte,
            AVG(CASE
                    WHEN ABS(delta) BETWEEN {atm_delta_low} AND {atm_delta_high}
                    THEN implied_volatility
                END) AS atm_iv,
            AVG(CASE
                    WHEN type = 'put' AND ABS(delta) BETWEEN {skew_delta_low} AND {skew_delta_high}
                    THEN implied_volatility
                END)
            -
            AVG(CASE
                    WHEN type = 'call' AND ABS(delta) BETWEEN {skew_delta_low} AND {skew_delta_high}
                    THEN implied_volatility
                END) AS skew_25d
        FROM base
        GROUP BY date, dte
    ),
    greeks AS (
        SELECT
            date,
            SUM(gex) AS total_gex,
            SUM(dex) AS total_dex,
            SUM(vex) AS total_vex
        FROM base
        GROUP BY date
    )
    SELECT c.*, g.total_gex, g.total_dex, g.total_vex
    FROM curves c
    LEFT JOIN greeks g ON c.date = g.date
    """

    df_curve = con.query(query).to_df()
    df_curve["date"] = pd.to_datetime(df_curve["date"], errors="coerce")
    df_curve = df_curve.dropna(subset=["date"])

    # 5. Interpolation Loop
    print("5. Interpolating Final Vectors...")
    surface_rows = []

    for day, day_curve in df_curve.groupby("date"):
        vec = {"trade_date": day}

        # Interpolate ATM IV @ target_dte
        vec["atm_iv_14d"] = interpolate_linear(day_curve, target_dte, "atm_iv")

        # Interpolate 25D skew @ target_dte
        vec["skew_25d_14d"] = interpolate_linear(day_curve, target_dte, "skew_25d")

        # Daily greek totals
        vec["net_gamma_exposure"] = float(day_curve["total_gex"].max()) if "total_gex" in day_curve else 0.0
        vec["net_delta_exposure"] = float(day_curve["total_dex"].max()) if "total_dex" in day_curve else 0.0
        vec["net_vega_exposure"] = float(day_curve["total_vex"].max()) if "total_vex" in day_curve else 0.0

        # Fear gap feature (IV - VIX)
        vix_val = vix_map.get(day, np.nan)
        if pd.notna(vix_val) and vec["atm_iv_14d"] > 0:
            vec["vol_spread"] = vec["atm_iv_14d"] - (float(vix_val) / 100.0)
        else:
            vec["vol_spread"] = 0.0

        surface_rows.append(vec)

    df_out = pd.DataFrame(surface_rows).sort_values("trade_date").fillna(0.0)

    # 6. Save
    df_out.to_csv(out_path, index=False)
    print(f"SUCCESS: Saved Surface Vector to {out_path}")
    print(f"Stats: {len(df_out)} rows generated.")
    print(f"Columns: {list(df_out.columns)}")


def main():
    ap = argparse.ArgumentParser()

    # --- HARDCODED DEFAULT PATHS ---
    # The 'r' prefix handles the backslashes in Windows paths correctly
    default_daily = r"C:\My Documents\Mics\Logs\tsla_daily.csv"
    default_opts = r"C:\My Documents\Mics\Logs\TSLA_Options_Chain_Historical_combined.csv"
    default_out = r"C:\My Documents\Mics\Logs\TSLA_Surface_Vector_Greeks.csv"

    ap.add_argument("--daily", default=default_daily, help="Daily stock file")
    ap.add_argument("--options", default=default_opts, help="Big options file")
    ap.add_argument("--out", default=default_out, help="Output file")

    args = ap.parse_args()

    build_surface_vectors(args.daily, args.options, args.out)


if __name__ == "__main__":
    main()