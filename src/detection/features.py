"""
Feature engineering for synthetic transactions.

Computes five derived fields that the rule engine works against:
amount_ratio, geo_delta_km, is_new_device, txn_velocity_2min, is_odd_hour.
"""

import json
from datetime import datetime, timedelta

import pandas as pd

from src.data.synthetic import haversine_distance_km


def extract_synthetic_features_single(
    txn: dict,
    recent_txns_history: list[dict] | None = None,
) -> dict:
    """Compute derived features for a single transaction dict."""
    amount = float(txn["amount"])
    hist_avg = float(txn.get("historical_avg_amount", amount))
    amount_ratio = round(amount / max(hist_avg, 1e-4), 2)

    loc = (float(txn["location_lat"]), float(txn["location_lon"]))
    hist_loc = (
        float(txn.get("historical_location_lat", loc[0])),
        float(txn.get("historical_location_lon", loc[1])),
    )
    geo_delta_km = haversine_distance_km(loc, hist_loc)

    raw_dev_ids = txn.get("historical_device_ids", "[]")
    if isinstance(raw_dev_ids, str):
        try:
            hist_device_ids = set(json.loads(raw_dev_ids))
        except json.JSONDecodeError:
            hist_device_ids = {raw_dev_ids}
    elif isinstance(raw_dev_ids, (list, set)):
        hist_device_ids = set(raw_dev_ids)
    else:
        hist_device_ids = set()

    is_new_device = bool(txn["device_id"] not in hist_device_ids)

    ts_str = txn["timestamp"]
    ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str

    user_id = txn["user_id"]
    velocity_2min = 1
    if recent_txns_history:
        two_min_ago = ts - timedelta(minutes=2)
        for h in recent_txns_history:
            if h.get("user_id") == user_id:
                h_ts = h["timestamp"]
                h_dt = datetime.fromisoformat(h_ts) if isinstance(h_ts, str) else h_ts
                if two_min_ago <= h_dt <= ts:
                    velocity_2min += 1

    start_hour = int(txn.get("typical_active_start_hour", 8))
    end_hour = int(txn.get("typical_active_end_hour", 22))
    is_odd_hour = bool(ts.hour < start_hour or ts.hour > end_hour)

    features = dict(txn)
    features.update({
        "amount_ratio": amount_ratio,
        "geo_delta_km": geo_delta_km,
        "is_new_device": is_new_device,
        "txn_velocity_2min": velocity_2min,
        "is_odd_hour": is_odd_hour,
    })
    return features


def extract_synthetic_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features for a batch DataFrame of synthetic transactions."""
    df = df.copy()
    if "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("dt").reset_index(drop=True)

    df["amount_ratio"] = (df["amount"] / df["historical_avg_amount"].clip(lower=1e-4)).round(2)

    def calc_geo(row):
        return haversine_distance_km(
            (row["location_lat"], row["location_lon"]),
            (row["historical_location_lat"], row["historical_location_lon"]),
        )

    df["geo_delta_km"] = df.apply(calc_geo, axis=1)

    def check_device(row):
        raw = row["historical_device_ids"]
        if isinstance(raw, str):
            try:
                known = set(json.loads(raw))
            except Exception:
                known = {raw}
        elif isinstance(raw, (list, set)):
            known = set(raw)
        else:
            known = set()
        return row["device_id"] not in known

    df["is_new_device"] = df.apply(check_device, axis=1)

    if "dt" in df.columns:
        velocities = []
        user_times = {uid: group["dt"].values for uid, group in df.groupby("user_id")}
        for _, row in df.iterrows():
            uid = row["user_id"]
            curr_dt = row["dt"]
            times = user_times[uid]
            two_min_ago = curr_dt - pd.Timedelta(minutes=2)
            count = int(((times >= two_min_ago) & (times <= curr_dt)).sum())
            velocities.append(count)
        df["txn_velocity_2min"] = velocities
    else:
        df["txn_velocity_2min"] = 1

    def check_odd_hour(row):
        hr = row["dt"].hour if "dt" in row else 12
        return hr < row.get("typical_active_start_hour", 8) or hr > row.get("typical_active_end_hour", 22)

    df["is_odd_hour"] = df.apply(check_odd_hour, axis=1)

    if "dt" in df.columns:
        df = df.drop(columns=["dt"])

    return df
