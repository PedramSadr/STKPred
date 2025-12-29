import argparse
import os
import numpy as np
import pandas as pd
import duckdb

# ================= CONFIG =================
CONTRACT_MULTIPLIER = 100.0
BASE_DIR = r"C:\My Documents\Mics\Logs"
# =========================================


def interpolate_linear(curve_df: pd.DataFrame, target_dte: int, col: str):
    """
    Linear interpolation across DTE.
    Returns NaN if data is unavailable (NO silent fabrication).
    """
    curve_df = curve_df.dropna(subset=["dte", col]).sort_values("dte")
    if curve_df.empty:
        return np.nan

    exact = curve_df[curve_df["dte"] == target_dte]
    if not exact.empty:
        return float(exact.iloc[0][col])

    near = curve_df[curve_df["dte"] < target_dte]
    far = curve_df[curve_df["dte"] > target_dte]

    if near.empty or far.empty:
        idx = (curve_df["dte"] - target_dte).abs().idxmin()
        return float(curve_df.loc[idx, col])

    r1, r2 = near.iloc[-1], far.iloc[0]
    d1, d2 = float(r1["dte"]), float(r2["dte"])
    v1, v2 = float(r1[col]), float(r2[col])

    if d1 == d2:
        return v1

    w = (target_dte - d1) / (d2 - d1)
    return v1 + w * (v2 - v1)


def _ensure_datetime(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def build_surface_vectors(daily_path: str, options_path: str, out_path: str):

    print("=== BUILDING INSTITUTIONAL OPTIONS SURFACE ===")

    # ---------- LOAD ----------
    df_daily = pd.read_csv(daily_path)
    df_opts = pd.read_csv(options_path, low_memory=False)

    df_daily.columns = df_daily.columns.str.lower().str.strip()
    df_opts.columns = df_opts.columns.str.lower().str.strip()

    _ensure_datetime(df_daily, "date")
    _ensure_datetime(df_opts, "date")
    _ensure_datetime(df_opts, "expiration")

    df_daily = df_daily.dropna(subset=["date"]).sort_values("date")
    df_opts = df_opts.dropna(subset=["date", "expiration"]).sort_values("date")

    if "close" not in df_daily.columns:
        raise ValueError("Daily file must contain 'close' column")

    # ---------- MAP DAILY CONTEXT ----------
    df_daily = df_daily.set_index("date")

    price_map = df_daily["close"].to_dict()
    vix_map = df_daily["vix"].to_dict() if "vix" in df_daily.columns else {}
    spy_map = df_daily["spy"].to_dict() if "spy" in df_daily.columns else {}
    dxy_map = df_daily["dxy"].to_dict() if "dxy" in df_daily.columns else {}
    wti_map = df_daily["wti"].to_dict() if "wti" in df_daily.columns else {}

    # ---------- OPTIONS CLEAN ----------
    df_opts["underlying_price"] = df_opts["date"].map(price_map)
    df_opts = df_opts.dropna(subset=["underlying_price"])

    df_opts["dte"] = (df_opts["expiration"] - df_opts["date"]).dt.days
    df_opts = df_opts[df_opts["dte"] >= 0]

    df_opts["type"] = df_opts.get("type", "call").astype(str).str.lower().str.strip()

    for c in ["delta", "gamma", "vega", "open_interest", "implied_volatility"]:
        df_opts[c] = pd.to_numeric(df_opts.get(c, 0.0), errors="coerce").fillna(0.0)

    # ---------- GREEKS EXPOSURES ----------
    df_opts["dex"] = (
        df_opts["delta"]
        * df_opts["open_interest"]
        * CONTRACT_MULTIPLIER
        * df_opts["underlying_price"]
    )

    df_opts["vex"] = (
        df_opts["vega"]
        * df_opts["open_interest"]
        * CONTRACT_MULTIPLIER
    )

    df_opts["gex"] = (
        df_opts["gamma"]
        * df_opts["open_interest"]
        * CONTRACT_MULTIPLIER
        * (df_opts["underlying_price"] ** 2)
    )

    # ---------- AGGREGATION (DUCKDB) ----------
    con = duckdb.connect()
    con.register("df_opts", df_opts)

    query = """
    WITH base AS (
        SELECT date, dte, type, delta, implied_volatility, dex, gex, vex
        FROM df_opts
    ),
    curves AS (
        SELECT
            date,
            dte,
            AVG(CASE WHEN ABS(delta) BETWEEN 0.40 AND 0.60 THEN implied_volatility END) AS atm_iv,
            AVG(CASE WHEN type='put'  AND ABS(delta) BETWEEN 0.20 AND 0.30 THEN implied_volatility END)
          - AVG(CASE WHEN type='call' AND ABS(delta) BETWEEN 0.20 AND 0.30 THEN implied_volatility END) AS skew_25d
        FROM base
        GROUP BY date, dte
    ),
    totals AS (
        SELECT
            date,
            SUM(CASE WHEN type='call' THEN gex ELSE 0 END) AS call_gex,
            SUM(CASE WHEN type='put'  THEN gex ELSE 0 END) AS put_gex,
            SUM(CASE WHEN type='call' THEN dex ELSE 0 END) AS call_dex,
            SUM(CASE WHEN type='put'  THEN dex ELSE 0 END) AS put_dex,
            SUM(vex) AS total_vex
        FROM base
        GROUP BY date
    )
    SELECT c.*, t.*
    FROM curves c
    LEFT JOIN totals t
    ON c.date = t.date
    """

    df_curve = con.execute(query).df()
    df_curve["date"] = pd.to_datetime(df_curve["date"])

    # ---------- FINAL DAILY VECTORS ----------
    surface_rows = []

    for day, day_curve in df_curve.groupby("date"):
        vec = {"trade_date": day}

        # Directional flows
        vec["call_gamma_exposure"] = float(day_curve["call_gex"].max())
        vec["put_gamma_exposure"] = float(day_curve["put_gex"].max())
        vec["net_gamma_exposure"] = vec["call_gamma_exposure"] - vec["put_gamma_exposure"]

        vec["call_delta_exposure"] = float(day_curve["call_dex"].max())
        vec["put_delta_exposure"] = float(day_curve["put_dex"].max())
        vec["net_delta_exposure"] = vec["call_delta_exposure"] - vec["put_delta_exposure"]

        vec["net_vega_exposure"] = float(day_curve["total_vex"].max())

        # Surface
        vec["atm_iv_14d"] = interpolate_linear(day_curve, 14, "atm_iv")
        vec["skew_25d_14d"] = interpolate_linear(day_curve, 14, "skew_25d")

        iv30 = interpolate_linear(day_curve, 30, "atm_iv")
        iv90 = interpolate_linear(day_curve, 90, "atm_iv")
        vec["term_structure_30_90"] = iv30 - iv90 if pd.notna(iv30) and pd.notna(iv90) else np.nan

        # Macro
        vix = vix_map.get(day)
        spy = spy_map.get(day)
        tsla = price_map.get(day)

        vec["has_vix"] = int(vix is not None)
        vec["has_spy"] = int(spy is not None)
        vec["has_surface"] = int(pd.notna(vec["atm_iv_14d"]))

        vec["vol_spread"] = (
            vec["atm_iv_14d"] - (vix / 100.0)
            if vec["has_vix"] and vec["has_surface"]
            else np.nan
        )

        vec["spy_close"] = spy
        vec["dxy_close"] = dxy_map.get(day)
        vec["wti_close"] = wti_map.get(day)
        vec["tsla_spy_ratio"] = tsla / spy if spy and spy > 0 else np.nan

        surface_rows.append(vec)

    df_out = pd.DataFrame(surface_rows).sort_values("trade_date")

    # ---------- LOG-STABILIZED FLOWS ----------
    for c in [
        "net_gamma_exposure",
        "net_delta_exposure",
        "net_vega_exposure",
    ]:
        df_out[f"log_{c}"] = np.sign(df_out[c]) * np.log1p(np.abs(df_out[c]))

    # ---------- DYNAMICS ----------
    df_out["iv_change_1d"] = df_out["atm_iv_14d"].diff()

    # ---------- FINALIZE ----------
    df_out = df_out.fillna(0.0)
    df_out.to_csv(out_path, index=False)

    print("✅ SUCCESS")
    print("Saved to:", out_path)
    print("Columns:", list(df_out.columns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", default=os.path.join(BASE_DIR, "tsla_daily.csv"))
    ap.add_argument("--options", default=os.path.join(BASE_DIR, "TSLA_Options_Chain_Historical_combined.csv"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "TSLA_Surface_Vector_Merged_v2.csv"))
    args = ap.parse_args()

    build_surface_vectors(args.daily, args.options, args.out)


if __name__ == "__main__":
    main()
