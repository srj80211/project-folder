# NYC Taxi Demand Zone Model

Machine learning model for predicting NYC taxi demand zones and forecasting pickup demand. Built with HDBSCAN clustering on 1.4M+ taxi trips.

## Model Overview

| Property | Value |
|----------|-------|
| **Model Type** | HDBSCAN_demand_zones_optimized |
| **Version** | 2.0 |
| **Training Samples** | 1,447,895 |
| **Clustering Samples** | 100,000 |
| **Number of Zones** | 7 |
| **Clustering Method** | HDBSCAN |
| **Silhouette Score** | N/A (HDBSCAN) |
| **NYC Bounds** | lat: 40.5–41.2, lon: -74.5–-73.3 |

## Quick Start

```python
from demand_zone_model_v2 import DemandZoneModelV2

# Load model
model = DemandZoneModelV2.load("demand_zones_model_optimized.pkl")

# Predict zone for a coordinate
zone_id, distance_km = model.predict(40.7580, -73.9855)  # Times Square

# Get demand at specific hour
demand = model.get_demand(zone_id, hour=11)  # 11 AM demand

# Get normalized demand level (0-1)
level = model.get_demand_level(zone_id, hour=11)

# Get 24-hour forecast
forecast = model.get_demand_forecast(zone_id, hours_ahead=24)

# List all zones
zones = model.list_zones(sort_by="total_demand")
```

---

## Python API Reference

### `DemandZoneModelV2`

Main class for demand zone inference and forecasting.

#### `load(filepath: str) -> DemandZoneModelV2`
Load model from pickle artifact file.

```python
model = DemandZoneModelV2.load("demand_zones_model_optimized.pkl")
```

#### `predict(lat: float, lon: float, k: int = 1) -> tuple | list[tuple]`
Predict nearest demand zone(s) for a coordinate.

**Args:**
- `lat` (float): Pickup latitude (40.5–41.2)
- `lon` (float): Pickup longitude (-74.5–-73.3)
- `k` (int, optional): Number of nearest zones to return. Default: 1

**Returns:**
- If `k=1`: `(zone_id: int, distance_km: float)`
- If `k>1`: `[(zone_id: int, distance_km: float), ...]`

**Example:**
```python
zone_id, dist = model.predict(40.7580, -73.9855)
# Returns: (6, 0.234)

top3 = model.predict(40.7580, -73.9855, k=3)
# Returns: [(6, 0.234), (2, 1.123), (0, 2.456)]
```

#### `get_zone_info(zone_id: int) -> dict | None`
Get complete zone information.

**Returns:**
```python
{
    "zone_id": 6,
    "center_lat": 40.7589,
    "center_lon": -73.9852,
    "zone_size": 45231,
    "total_demand": 234567,
    "demand_by_hour": {0: 123, 1: 98, ..., 23: 456},
    "demand_by_day": {0: 34567, 1: 32123, ..., 6: 28901},
    "demand_by_month": {1: 189023, ..., 12: 201345},
    "demand_peak_normalized": [0.12, 0.09, ..., 0.98],
    "peak_hour": 18,
    "peak_day": 4,
    "avg_membership_prob": 0.823,
    "zone_spread_km": 1.4
}
```

#### `get_demand(zone_id: int, hour: int = None, dayofweek: int = None, month: int = None) -> int`
Get demand count for a zone at specific time granularity.

**Args:**
- `zone_id` (int): Zone identifier
- `hour` (int, optional): Hour of day (0–23)
- `dayofweek` (int, optional): Day of week (0=Monday, 6=Sunday)
- `month` (int, optional): Month (1–12)

**Returns:** Pickup count from training data for specified period.

**Examples:**
```python
model.get_demand(zone_id)           # Total demand across all time
model.get_demand(zone_id, hour=11)  # Demand at 11 AM
model.get_demand(zone_id, dayofweek=4)  # Demand on Friday
model.get_demand(zone_id, month=7)      # Demand in July
```

#### `get_demand_level(zone_id: int, hour: int = None) -> float`
Get normalized demand level (0.0–1.0) relative to zone's peak hour.

**Args:**
- `zone_id` (int): Zone identifier
- `hour` (int, optional): Hour of day (0–23). If None, returns overall normalized mean.

**Returns:** Float between 0.0 and 1.0

**Example:**
```python
model.get_demand_level(6, hour=11)  # 0.45 (45% of peak)
model.get_demand_level(6, hour=18)  # 1.0 (peak hour)
```

#### `get_demand_forecast(zone_id: int, hours_ahead: int = 24) -> list[dict]`
Get cyclic daily demand forecast for next N hours.

**Args:**
- `zone_id` (int): Zone identifier
- `hours_ahead` (int): Number of hours to forecast (default: 24)

**Returns:** List of forecast objects:
```python
[
    {"hour": 14, "demand": 456, "demand_level": 0.67},
    {"hour": 15, "demand": 523, "demand_level": 0.77},
    ...
]
```

#### `get_zone_quality(zone_id: int) -> dict | None`
Get zone quality metrics.

**Returns:**
```python
{
    "avg_membership_prob": 0.823,
    "zone_spread_km": 1.4,
    "total_demand": 234567,
    "silhouette_score": None
}
```

#### `list_zones(sort_by: str = "total_demand") -> list[dict]`
List all zones with summary info.

**Args:**
- `sort_by` (str): Sort key - `"total_demand"` (default) or `"zone_id"`

**Returns:**
```python
[
    {
        "zone_id": 6,
        "center_lat": 40.7589,
        "center_lon": -73.9852,
        "zone_size": 45231,
        "total_demand": 234567,
        "peak_hour": 18,
        "peak_day": 4,
        "avg_membership_prob": 0.823,
        "zone_spread_km": 1.4
    },
    ...
]
```

#### `is_in_bounds(lat: float, lon: float) -> bool`
Check if coordinate falls within NYC training bounds.

#### `get_model_info() -> dict`
Get model metadata and configuration.

**Returns:**
```python
{
    "model_type": "HDBSCAN_demand_zones_optimized",
    "version": "2.0",
    "n_zones": 7,
    "training_samples": 1447895,
    "silhouette_score": None,
    "feature_columns": ["pickup_latitude", "pickup_longitude"],
    "nyc_bounds": {"min_lat": 40.5, "max_lat": 41.2, "min_lon": -74.5, "max_lon": -73.3}
}
```

---

## REST API Endpoints (FastAPI Example)

The following endpoints are implemented in `test_integration_v2.py` as a FastAPI integration example. To deploy:

```bash
pip install fastapi uvicorn pydantic
uvicorn api:app --host 0.0.0.0 --port 8000
```

### `POST /predict`
Predict demand zone(s) for a pickup coordinate.

**Request Body:**
```json
{
  "pickup_latitude": 40.7580,
  "pickup_longitude": -73.9855,
  "hour": 11,
  "dayofweek": 4,
  "include_forecast": false,
  "top_k": 1
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pickup_latitude` | float | Yes | Latitude (40.5–41.2) |
| `pickup_longitude` | float | Yes | Longitude (-74.5–-73.3) |
| `hour` | int | No | Hour of day (0–23) |
| `dayofweek` | int | No | Day of week (0=Mon, 6=Sun) |
| `include_forecast` | bool | No | Include 24h forecast. Default: false |
| `top_k` | int | No | Number of nearest zones (1–5). Default: 1 |

**Response (200):**
```json
[
  {
    "zone_id": 6,
    "distance_km": 0.234,
    "center_lat": 40.7589,
    "center_lon": -73.9852,
    "demand_at_hour": 4567,
    "demand_level": 0.672,
    "demand_forecast": null,
    "zone_quality": {
      "avg_membership_prob": 0.823,
      "zone_spread_km": 1.4,
      "total_demand": 234567
    }
  }
]
```

**Error Responses:**
- `400`: Coordinate outside NYC bounds
- `422`: Invalid request parameters

---

### `GET /zones`
List all demand zones sorted by total demand (descending).

**Response (200):**
```json
[
  {
    "zone_id": 6,
    "center_lat": 40.7589,
    "center_lon": -73.9852,
    "zone_size": 45231,
    "total_demand": 234567,
    "peak_hour": 18,
    "peak_day": 4,
    "avg_membership_prob": 0.823,
    "zone_spread_km": 1.4
  },
  ...
]
```

---

### `GET /model/info`
Get model metadata and configuration.

**Response (200):**
```json
{
  "model_type": "HDBSCAN_demand_zones_optimized",
  "version": "2.0",
  "n_zones": 7,
  "training_samples": 1447895,
  "silhouette_score": null,
  "feature_columns": ["pickup_latitude", "pickup_longitude"],
  "nyc_bounds": {
    "min_lat": 40.5,
    "max_lat": 41.2,
    "min_lon": -74.5,
    "max_lon": -73.3
  }
}
```

---

## Zone Definitions

The model identifies 7 demand zones across NYC:

| Zone ID | Area | Center (lat, lon) | Total Demand | Peak Hour | Spread (km) |
|---------|------|-------------------|--------------|-----------|-------------|
| 0 | Upper Manhattan | 40.81, -73.95 | ~156K | 18:00 | 2.1 |
| 1 | Brooklyn | 40.67, -73.94 | ~134K | 19:00 | 3.2 |
| 2 | Queens | 40.73, -73.82 | ~189K | 17:00 | 4.1 |
| 3 | Bronx | 40.85, -73.88 | ~98K | 18:00 | 2.8 |
| 4 | JFK Airport | 40.64, -73.78 | ~87K | 14:00 | 1.9 |
| 5 | LaGuardia Airport | 40.78, -73.87 | ~76K | 12:00 | 1.5 |
| 6 | **Midtown Manhattan** | **40.76, -73.99** | **~235K** | **18:00** | **1.4** |

*Zone 6 (Midtown) has highest demand and tightest cluster*

---

## Input Features

The model expects exactly 2 features in this order:

1. **`pickup_latitude`** (float): Latitude coordinate
2. **`pickup_longitude`** (float): Longitude coordinate

Both must fall within NYC bounds: `lat ∈ [40.5, 41.2]`, `lon ∈ [-74.5, -73.3]`

---

## Output Specification

### Prediction Output
| Field | Type | Description |
|-------|------|-------------|
| `zone_id` | int | Cluster identifier (0–6) |
| `distance_km` | float | Haversine distance to zone center |
| `center_lat` | float | Zone center latitude |
| `center_lon` | float | Zone center longitude |
| `demand_at_hour` | int | Pickup count for requested hour |
| `demand_level` | float | Normalized demand 0–1 vs zone peak |
| `demand_forecast` | list | 24h forecast (if requested) |
| `zone_quality` | object | Zone cohesion metrics |

### Zone Quality Metrics
| Metric | Range | Description |
|--------|-------|-------------|
| `avg_membership_prob` | 0–1 | Cluster assignment confidence |
| `zone_spread_km` | >0 | Spatial extent (radius) |
| `total_demand` | ≥0 | Total pickups in training data |

---

## Example Usage

### Batch Prediction
```python
import pandas as pd
from demand_zone_model_v2 import DemandZoneModelV2

model = DemandZoneModelV2.load("demand_zones_model_optimized.pkl")

df = pd.read_csv("trip_requests.csv")  # columns: pickup_lat, pickup_lon
results = []

for _, row in df.iterrows():
    zone_id, dist = model.predict(row.pickup_lat, row.pickup_lon)
    z = model.get_zone_info(zone_id)
    results.append({
        "pickup_lat": row.pickup_lat,
        "pickup_lon": row.pickup_lon,
        "zone_id": zone_id,
        "distance_km": dist,
        "zone_center_lat": z["center_lat"],
        "zone_center_lon": z["center_lon"],
        "zone_demand": z["total_demand"]
    })

pd.DataFrame(results).to_csv("predictions.csv", index=False)
```

### Demand Heatmap by Hour
```python
zones = model.list_zones()
for hour in range(24):
    print(f"\n--- Hour {hour}:00 ---")
    for z in zones[:5]:  # Top 5 zones
        demand = model.get_demand(z["zone_id"], hour=hour)
        level = model.get_demand_level(z["zone_id"], hour=hour)
        print(f"  Zone {z['zone_id']}: {demand:,} pickups (level={level:.2f})")
```

---

## Files

| File | Description |
|------|-------------|
| `demand_zone_model_v2.py` | Main model class (v2 optimized) |
| `demand_zone_model.py` | Original model class (v1) |
| `demand_zones_model_optimized.pkl` | Serialized model artifact (v2) |
| `demand_zones_model.pkl` | Serialized model artifact (v1) |
| `model_metadata.json` | Model configuration |
| `zones.json` | Zone definitions |
| `zone_centers.json` | Zone center coordinates |
| `demand_zones_model.json` | Full model export |
| `test_integration_v2.py` | Integration tests + FastAPI example |
| `train_optimized.py` | Training script (v2) |
| `train_model.py` | Training script (v1) |

---

## Requirements

```
scikit-learn>=1.3
joblib>=1.3
numpy>=1.24
hdbscan>=0.8  # for training only
pandas>=2.0   # for training only
fastapi>=0.100  # for API deployment
uvicorn>=0.23   # for API deployment
pydantic>=2.0   # for API deployment
```

---

## License

Internal use only.