import argparse
import os
import numpy as np
import pandas as pd
import duckdb

# --- CONFIGURATION ---
CONTRACT_MULTIPLIER = 100.0


def interpolate_linear(curve_df: pd.DataFrame, target_dte: int, col: str) -> float:
    """ Linear interpolation helper """
    curve_df = curve_df.dropna(subset=["dte", col]).sort_values("dte")
    if curve_df.empty: return 0.0

    exact = curve_df[curve_df["dte"] == target_dte]
    if not exact.empty: return float(exact.iloc[0][col])

    near = curve_df[curve_df["dte"] < target_dte]
    far = curve_df[curve_df["dte"] > target_dte]

    if near.empty or far.empty:
        idx = (curve_df["dte"] - target_dte).abs().idxmin()
        return float(curve_df.loc[idx, col])

    r1, r2 = near.iloc[-1], far.iloc[0]
    d1, d2 = float(r1["dte"]), float(r2["dte"])
    v1, v2 = float(r1[col]), float(r2[col])

    if d2 == d1: return float(v1)
    w = (target_dte - d1) / (d2 - d1)
    return float(v1 + w * (v2 - v1))


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    """Safely converts to datetime, strips timezones, and normalizes to midnight."""
    if col in df.columns:
        s = pd.to_datetime(df[col], errors="coerce")
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_convert(None)
        df[col] = s.dt.normalize()


def build_surface_vectors(daily_path: str, options_path: str, surf_out_path: str, menu_out_path: str) -> None:
    print("--- STARTING COMPLETE GENERATOR (SURFACE + MENU) ---")
    print(f"Daily:   {daily_path}")
    print(f"Options: {options_path}")

    # 1. LOAD & CLEAN
    if not os.path.exists(daily_path): raise FileNotFoundError(f"Missing: {daily_path}")
    if not os.path.exists(options_path): raise FileNotFoundError(f"Missing: {options_path}")

    df_daily = pd.read_csv(daily_path)
    df_opts = pd.read_csv(options_path, low_memory=False)

    df_daily.columns = df_daily.columns.str.strip().str.lower()
    df_opts.columns = df_opts.columns.str.strip().str.lower()

    date_col = 'date' if 'date' in df_daily.columns else 'Date'
    if date_col in df_daily.columns: df_daily.rename(columns={date_col: "date"}, inplace=True)

    _ensure_datetime(df_daily, "date")
    _ensure_datetime(df_opts, "date")
    _ensure_datetime(df_opts, "expiration")

    df_daily = df_daily.dropna(subset=["date"]).sort_values("date")
    df_opts = df_opts.dropna(subset=["date", "expiration"]).sort_values("date")

    # 2. MAP CONTEXT (Using Foolproof String Mapping)
    print("2. Mapping Context...")
    if "close" not in df_daily.columns: raise ValueError("Daily file needs 'close'")

    df_daily['date_str'] = df_daily['date'].dt.strftime('%Y-%m-%d')
    df_opts['date_str'] = df_opts['date'].dt.strftime('%Y-%m-%d')

    price_map = df_daily.set_index('date_str')["close"].to_dict()
    vix_map = df_daily.set_index('date_str')["vix"].to_dict() if "vix" in df_daily.columns else {}
    spy_map = df_daily.set_index('date_str')["spy"].to_dict() if "spy" in df_daily.columns else {}
    dxy_map = df_daily.set_index('date_str')["dxy"].to_dict() if "dxy" in df_daily.columns else {}
    wti_map = df_daily.set_index('date_str')["wti"].to_dict() if "wti" in df_daily.columns else {}

    initial_rows = len(df_opts)
    df_opts["underlying_price"] = df_opts["date_str"].map(price_map)
    df_opts = df_opts.dropna(subset=["underlying_price"])
    print(f"   -> Mapped Underlying Price. Retained {len(df_opts)} of {initial_rows} option contracts.")

    df_opts["dte"] = (df_opts["expiration"] - df_opts["date"]).dt.days
    df_opts = df_opts[df_opts["dte"] >= 0]

    if "type" not in df_opts.columns:
        df_opts["type"] = "call"
    else:
        df_opts["type"] = df_opts["type"].astype(str).str.strip().str.lower()

    for c in ["delta", "gamma", "vega", "open_interest"]:
        if c not in df_opts.columns: df_opts[c] = 0.0
        df_opts[c] = pd.to_numeric(df_opts[c], errors="coerce").fillna(0.0)

    if "implied_volatility" in df_opts.columns:
        df_opts["implied_volatility"] = pd.to_numeric(df_opts["implied_volatility"], errors="coerce")
    else:
        df_opts["implied_volatility"] = np.nan

    # 3. GENERATE MENU
    print(f"3. Generating Options Menu -> {menu_out_path}")
    df_opts['moneyness'] = df_opts['strike'] / df_opts['underlying_price']

    if 'mark' in df_opts.columns:
        df_opts['opt_price'] = df_opts['mark']
    elif 'bid' in df_opts.columns and 'ask' in df_opts.columns:
        df_opts['opt_price'] = (df_opts['bid'] + df_opts['ask']) / 2
    elif 'last' in df_opts.columns:
        df_opts['opt_price'] = df_opts['last']
    else:
        df_opts['opt_price'] = np.nan

    contract_cols = [
        'date', 'expiration', 'dte', 'type', 'strike', 'moneyness', 'opt_price',
        'bid', 'ask', 'volume', 'open_interest',
        'implied_volatility',
        'delta', 'gamma', 'vega', 'underlying_price'
    ]
    final_cols = [c for c in contract_cols if c in df_opts.columns]
    df_opts[final_cols].to_csv(menu_out_path, index=False)

    # 4. CALCULATE PHYSICS
    print("4. Calculating Physics for Surface...")
    df_opts["dex"] = df_opts["delta"] * df_opts["open_interest"] * CONTRACT_MULTIPLIER * df_opts["underlying_price"]
    df_opts["vex"] = df_opts["vega"] * df_opts["open_interest"] * CONTRACT_MULTIPLIER
    df_opts["gex"] = df_opts["gamma"] * df_opts["open_interest"] * CONTRACT_MULTIPLIER * (
                df_opts["underlying_price"] ** 2)

    # 5. AGGREGATE
    print("5. Aggregating Surface...")
    con = duckdb.connect()
    con.register('df_opts', df_opts)

    query = """
    WITH base AS (
        SELECT date, dte, type, delta, implied_volatility, dex, gex, vex FROM df_opts
    ),
    curves AS (
        SELECT date, dte,
            AVG(CASE WHEN ABS(delta) BETWEEN 0.40 AND 0.60 THEN implied_volatility END) AS atm_iv,
            AVG(CASE WHEN type='put' AND ABS(delta) BETWEEN 0.20 AND 0.30 THEN implied_volatility END) -
            AVG(CASE WHEN type='call' AND ABS(delta) BETWEEN 0.20 AND 0.30 THEN implied_volatility END) AS skew_25d
        FROM base GROUP BY date, dte
    ),
    totals AS (
        SELECT date, SUM(gex) AS total_gex, SUM(dex) AS total_dex, SUM(vex) AS total_vex
        FROM base GROUP BY date
    )
    SELECT c.*, t.total_gex, t.total_dex, t.total_vex
    FROM curves c LEFT JOIN totals t ON c.date = t.date
    """
    df_curve = con.query(query).to_df()
    df_curve["date"] = pd.to_datetime(df_curve["date"], errors="coerce").dropna()

    # 6. INTERPOLATE FINAL VECTORS
    print("6. Generating Final Surface Dataset...")
    surface_rows = []

    for day, day_curve in df_curve.groupby("date"):
        day_str = day.strftime('%Y-%m-%d')
        vec = {"trade_date": day}

        vec["net_gamma_exposure"] = float(day_curve["total_gex"].max())
        vec["net_delta_exposure"] = float(day_curve["total_dex"].max())
        vec["net_vega_exposure"] = float(day_curve["total_vex"].max())

        vec["atm_iv_14d"] = interpolate_linear(day_curve, 14, "atm_iv")
        vec["skew_25d_14d"] = interpolate_linear(day_curve, 14, "skew_25d")

        iv30 = interpolate_linear(day_curve, 30, "atm_iv")
        iv90 = interpolate_linear(day_curve, 90, "atm_iv")
        vec["term_structure_30_90"] = iv30 - iv90

        vix_val = vix_map.get(day_str, np.nan)
        spy_val = spy_map.get(day_str, np.nan)
        tsla_val = price_map.get(day_str, np.nan)

        vec["vol_spread"] = (vec["atm_iv_14d"] - (vix_val / 100.0)) if pd.notna(vix_val) and vec[
            "atm_iv_14d"] > 0 else 0.0
        vec["spy_close"] = spy_val if pd.notna(spy_val) else 0.0
        vec["dxy_close"] = dxy_map.get(day_str, 0.0)
        vec["wti_close"] = wti_map.get(day_str, 0.0)
        vec["tsla_spy_ratio"] = (tsla_val / spy_val) if (pd.notna(spy_val) and spy_val > 0) else 0.0

        surface_rows.append(vec)

    df_out = pd.DataFrame(surface_rows)

    if not df_out.empty and "trade_date" in df_out.columns:
        df_out = df_out.sort_values("trade_date").fillna(0.0)
        if "atm_iv_14d" in df_out.columns:
            df_out['iv_change_1d'] = df_out['atm_iv_14d'].diff().fillna(0.0)
        else:
            df_out['iv_change_1d'] = 0.0
    else:
        df_out = df_out.fillna(0.0)
        df_out['iv_change_1d'] = 0.0

    df_out.to_csv(surf_out_path, index=False)
    print(f"SUCCESS. Saved Surface to: {surf_out_path}")
    print(f"SUCCESS. Saved Menu to:    {menu_out_path}")


def main():
    ap = argparse.ArgumentParser()
    base_dir = r"C:\My Documents\Mics\Logs"
    ap.add_argument("--daily", default=os.path.join(base_dir, "tsla_daily.csv"))
    ap.add_argument("--options", default=os.path.join(base_dir, "TSLA_Options_Chain_Historical_combined.csv"))
    ap.add_argument("--out-surface", default=os.path.join(base_dir, "TSLA_Surface_Vector_Merged.csv"))
    ap.add_argument("--out-menu", default=os.path.join(base_dir, "TSLA_Options_Contracts.csv"))

    args = ap.parse_args()
    build_surface_vectors(args.daily, args.options, args.out_surface, args.out_menu)


if __name__ == "__main__":
    main()