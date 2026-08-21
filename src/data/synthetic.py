"""
Synthetic transaction generator with documented ground truth labels.

Generates normal transactions and controlled anomalies across 5 categories:
high_velocity, geo_anomaly, amount_spike, new_device_high_value, odd_hour_spend.
Every row includes is_anomaly (0/1) and root_cause so the eval script has
unambiguous ground truth to grade against.
"""

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED,
    SYNTHETIC_ANOMALY_COUNT,
    SYNTHETIC_NORMAL_COUNT,
)

LOCATIONS = {
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "London": (51.5074, -0.1278),
    "New York": (40.7128, -74.0060),
    "Singapore": (1.3521, 103.8198),
}

MERCHANT_CATEGORIES = [
    "food_dining",
    "groceries",
    "travel_flight",
    "electronics",
    "apparel",
    "entertainment",
    "utilities",
    "health_wellness",
]


def haversine_distance_km(
    loc1: tuple[float, float], loc2: tuple[float, float]
) -> float:
    """Great-circle distance between two lat/lon points in kilometres."""
    lat1, lon1 = loc1
    lat2, lon2 = loc2
    r = 6371.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


@dataclass
class UserProfile:
    user_id: str
    home_city: str
    home_coords: tuple[float, float]
    known_devices: list[str]
    typical_avg_amount: float
    active_start_hour: int
    active_end_hour: int


def _generate_user_profiles(num_users: int = 200, rng: random.Random | None = None) -> list[UserProfile]:
    if rng is None:
        rng = random.Random(RANDOM_SEED)

    city_names = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Kolkata"]
    profiles = []

    for i in range(num_users):
        user_id = f"usr_{i+1:04d}"
        home_city = rng.choice(city_names)
        home_coords = LOCATIONS[home_city]
        num_devices = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        known_devices = [f"dev_{user_id}_{d+1}" for d in range(num_devices)]
        typical_avg_amount = round(rng.uniform(300.0, 4500.0), 2)
        active_start_hour = rng.choice([6, 7, 8, 9])
        active_end_hour = rng.choice([21, 22, 23])

        profiles.append(
            UserProfile(
                user_id=user_id,
                home_city=home_city,
                home_coords=home_coords,
                known_devices=known_devices,
                typical_avg_amount=typical_avg_amount,
                active_start_hour=active_start_hour,
                active_end_hour=active_end_hour,
            )
        )
    return profiles


def generate_synthetic_transactions(
    normal_count: int = SYNTHETIC_NORMAL_COUNT,
    anomaly_count: int = SYNTHETIC_ANOMALY_COUNT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic transactions with ground truth labels.

    Returns a DataFrame with is_anomaly (0/1) and root_cause for every row.
    """
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    profiles = _generate_user_profiles(num_users=250, rng=py_rng)
    base_time = datetime(2026, 8, 1, 8, 0, 0)

    rows: list[dict] = []
    txn_id_counter = 1

    # Normal transactions
    for _ in range(normal_count):
        profile = py_rng.choice(profiles)
        txn_id = f"txn_{txn_id_counter:05d}"
        txn_id_counter += 1

        amount_factor = float(np_rng.lognormal(mean=0.0, sigma=0.35))
        amount_factor = max(0.2, min(2.5, amount_factor))
        amount = round(profile.typical_avg_amount * amount_factor, 2)

        day_offset = py_rng.randint(0, 14)
        active_hours = list(range(profile.active_start_hour, profile.active_end_hour + 1))
        hour = py_rng.choice(active_hours)
        minute = py_rng.randint(0, 59)
        second = py_rng.randint(0, 59)
        timestamp = base_time + timedelta(days=day_offset, hours=hour - 8, minutes=minute, seconds=second)

        lat_jitter = py_rng.uniform(-0.15, 0.15)
        lon_jitter = py_rng.uniform(-0.15, 0.15)
        location = (round(profile.home_coords[0] + lat_jitter, 4), round(profile.home_coords[1] + lon_jitter, 4))

        rows.append({
            "transaction_id": txn_id,
            "user_id": profile.user_id,
            "amount": amount,
            "merchant_category": py_rng.choice(MERCHANT_CATEGORIES),
            "location_lat": location[0],
            "location_lon": location[1],
            "device_id": py_rng.choice(profile.known_devices),
            "timestamp": timestamp.isoformat(),
            "historical_avg_amount": profile.typical_avg_amount,
            "historical_location_lat": profile.home_coords[0],
            "historical_location_lon": profile.home_coords[1],
            "historical_device_ids": json.dumps(profile.known_devices),
            "typical_active_start_hour": profile.active_start_hour,
            "typical_active_end_hour": profile.active_end_hour,
            "is_anomaly": 0,
            "root_cause": "normal",
        })

    # Anomalies — 5 categories
    categories = [
        "high_velocity",
        "geo_anomaly",
        "amount_spike",
        "new_device_high_value",
        "odd_hour_spend",
    ]
    per_cat = anomaly_count // len(categories)
    remainder = anomaly_count % len(categories)

    for cat_idx, cat in enumerate(categories):
        count = per_cat + (1 if cat_idx < remainder else 0)

        if cat == "high_velocity":
            # Bursts of 3 transactions spaced 25 seconds apart.
            # The velocity rule fires at txn_velocity_2min >= 3, so:
            # txn 1 and 2 in the burst have velocity 1 and 2 — they don't trigger.
            # Only txn 3 reaches count=3 and gets labeled is_anomaly=1.
            bursts = max(1, count // 3)
            for _ in range(bursts):
                profile = py_rng.choice(profiles)
                burst_time = base_time + timedelta(
                    days=py_rng.randint(0, 14),
                    hours=py_rng.choice(range(profile.active_start_hour, profile.active_end_hour)),
                    minutes=py_rng.randint(0, 55),
                )
                device_id = py_rng.choice(profile.known_devices)
                for b_idx in range(3):
                    txn_id = f"txn_{txn_id_counter:05d}"
                    txn_id_counter += 1
                    t = burst_time + timedelta(seconds=b_idx * 25)
                    amount = round(profile.typical_avg_amount * py_rng.uniform(0.8, 1.5), 2)
                    rows.append({
                        "transaction_id": txn_id,
                        "user_id": profile.user_id,
                        "amount": amount,
                        "merchant_category": py_rng.choice(MERCHANT_CATEGORIES),
                        "location_lat": profile.home_coords[0],
                        "location_lon": profile.home_coords[1],
                        "device_id": device_id,
                        "timestamp": t.isoformat(),
                        "historical_avg_amount": profile.typical_avg_amount,
                        "historical_location_lat": profile.home_coords[0],
                        "historical_location_lon": profile.home_coords[1],
                        "historical_device_ids": json.dumps(profile.known_devices),
                        "typical_active_start_hour": profile.active_start_hour,
                        "typical_active_end_hour": profile.active_end_hour,
                        "is_anomaly": 1 if b_idx == 2 else 0,
                        "root_cause": "high_velocity" if b_idx == 2 else "normal",
                    })

        elif cat == "geo_anomaly":
            distant_cities = ["London", "New York", "Singapore", "Kolkata", "Delhi"]
            for _ in range(count):
                profile = py_rng.choice(profiles)
                txn_id = f"txn_{txn_id_counter:05d}"
                txn_id_counter += 1

                target_city = py_rng.choice(distant_cities)
                target_coords = LOCATIONS[target_city]
                while haversine_distance_km(profile.home_coords, target_coords) <= 500:
                    target_city = py_rng.choice(["London", "New York", "Singapore"])
                    target_coords = LOCATIONS[target_city]

                amount = round(profile.typical_avg_amount * py_rng.uniform(0.8, 2.0), 2)
                t = base_time + timedelta(
                    days=py_rng.randint(0, 14),
                    hours=py_rng.choice(range(profile.active_start_hour, profile.active_end_hour)),
                    minutes=py_rng.randint(0, 59),
                )

                rows.append({
                    "transaction_id": txn_id,
                    "user_id": profile.user_id,
                    "amount": amount,
                    "merchant_category": "travel_flight" if "York" in target_city or "London" in target_city else py_rng.choice(MERCHANT_CATEGORIES),
                    "location_lat": target_coords[0],
                    "location_lon": target_coords[1],
                    "device_id": py_rng.choice(profile.known_devices),
                    "timestamp": t.isoformat(),
                    "historical_avg_amount": profile.typical_avg_amount,
                    "historical_location_lat": profile.home_coords[0],
                    "historical_location_lon": profile.home_coords[1],
                    "historical_device_ids": json.dumps(profile.known_devices),
                    "typical_active_start_hour": profile.active_start_hour,
                    "typical_active_end_hour": profile.active_end_hour,
                    "is_anomaly": 1,
                    "root_cause": "geo_anomaly",
                })

        elif cat == "amount_spike":
            for _ in range(count):
                profile = py_rng.choice(profiles)
                txn_id = f"txn_{txn_id_counter:05d}"
                txn_id_counter += 1

                spike_ratio = py_rng.uniform(5.5, 12.0)
                amount = round(profile.typical_avg_amount * spike_ratio, 2)
                t = base_time + timedelta(
                    days=py_rng.randint(0, 14),
                    hours=py_rng.choice(range(profile.active_start_hour, profile.active_end_hour)),
                    minutes=py_rng.randint(0, 59),
                )

                rows.append({
                    "transaction_id": txn_id,
                    "user_id": profile.user_id,
                    "amount": amount,
                    "merchant_category": "electronics",
                    "location_lat": profile.home_coords[0],
                    "location_lon": profile.home_coords[1],
                    "device_id": py_rng.choice(profile.known_devices),
                    "timestamp": t.isoformat(),
                    "historical_avg_amount": profile.typical_avg_amount,
                    "historical_location_lat": profile.home_coords[0],
                    "historical_location_lon": profile.home_coords[1],
                    "historical_device_ids": json.dumps(profile.known_devices),
                    "typical_active_start_hour": profile.active_start_hour,
                    "typical_active_end_hour": profile.active_end_hour,
                    "is_anomaly": 1,
                    "root_cause": "amount_spike",
                })

        elif cat == "new_device_high_value":
            for _ in range(count):
                profile = py_rng.choice(profiles)
                txn_id = f"txn_{txn_id_counter:05d}"
                txn_id_counter += 1

                new_device = f"dev_unrecognized_{py_rng.randint(10000, 99999)}"
                ratio = py_rng.uniform(2.3, 4.5)
                amount = round(profile.typical_avg_amount * ratio, 2)
                t = base_time + timedelta(
                    days=py_rng.randint(0, 14),
                    hours=py_rng.choice(range(profile.active_start_hour, profile.active_end_hour)),
                    minutes=py_rng.randint(0, 59),
                )

                rows.append({
                    "transaction_id": txn_id,
                    "user_id": profile.user_id,
                    "amount": amount,
                    "merchant_category": "electronics",
                    "location_lat": profile.home_coords[0],
                    "location_lon": profile.home_coords[1],
                    "device_id": new_device,
                    "timestamp": t.isoformat(),
                    "historical_avg_amount": profile.typical_avg_amount,
                    "historical_location_lat": profile.home_coords[0],
                    "historical_location_lon": profile.home_coords[1],
                    "historical_device_ids": json.dumps(profile.known_devices),
                    "typical_active_start_hour": profile.active_start_hour,
                    "typical_active_end_hour": profile.active_end_hour,
                    "is_anomaly": 1,
                    "root_cause": "new_device_high_value",
                })

        elif cat == "odd_hour_spend":
            for _ in range(count):
                profile = py_rng.choice(profiles)
                txn_id = f"txn_{txn_id_counter:05d}"
                txn_id_counter += 1

                hour = py_rng.choice([1, 2, 3, 4])
                ratio = py_rng.uniform(3.2, 4.8)
                amount = round(profile.typical_avg_amount * ratio, 2)
                day_offset = py_rng.randint(1, 14)
                midnight = datetime(base_time.year, base_time.month, base_time.day) + timedelta(days=day_offset)
                t = midnight + timedelta(hours=hour, minutes=py_rng.randint(0, 59))

                rows.append({
                    "transaction_id": txn_id,
                    "user_id": profile.user_id,
                    "amount": amount,
                    "merchant_category": "entertainment",
                    "location_lat": profile.home_coords[0],
                    "location_lon": profile.home_coords[1],
                    "device_id": py_rng.choice(profile.known_devices),
                    "timestamp": t.isoformat(),
                    "historical_avg_amount": profile.typical_avg_amount,
                    "historical_location_lat": profile.home_coords[0],
                    "historical_location_lon": profile.home_coords[1],
                    "historical_device_ids": json.dumps(profile.known_devices),
                    "typical_active_start_hour": profile.active_start_hour,
                    "typical_active_end_hour": profile.active_end_hour,
                    "is_anomaly": 1,
                    "root_cause": "odd_hour_spend",
                })

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("dt").reset_index(drop=True)
    df = df.drop(columns=["dt"])
    return df
