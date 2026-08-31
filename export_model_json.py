"""
Export model data to JSON format for company delivery
"""

import joblib
import json
import numpy as np

# Load the optimized model
artifact = joblib.load('demand_zones_model_optimized.pkl')
zone_lookup = artifact['zone_lookup']

# Convert numpy types to Python native types for JSON serialization
def convert(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert(v) for v in obj)
    return obj

# 1. Model metadata
model_metadata = {
    "model_type": artifact['model_type'],
    "version": artifact['version'],
    "feature_columns": artifact['feature_columns'],
    "nyc_bounds": artifact['nyc_bounds'],
    "training_samples": artifact['training_samples'],
    "clustering_samples": artifact.get('clustering_samples'),
    "silhouette_score": artifact.get('silhouette_score'),
    "clustering_params": artifact.get('clustering_params', {}),
    "n_zones": zone_lookup['n_zones'],
    "clustering_method": zone_lookup['clustering_method']
}

# 2. Zones data (serializable)
zones_json = []
for z in zone_lookup['zones']:
    zone_data = {
        "zone_id": int(z['zone_id']),
        "center": {
            "latitude": float(z['center_lat']),
            "longitude": float(z['center_lon'])
        },
        "zone_size": int(z['zone_size']),
        "total_demand": int(z['total_demand']),
        "avg_membership_prob": float(z['avg_membership_prob']),
        "zone_spread_km": float(z['zone_spread_km']),
        "peak_hour": int(z['peak_hour']),
        "peak_day": int(z['peak_day']),
        "demand_by_hour": {int(k): int(v) for k, v in z['demand_by_hour'].items()},
        "demand_by_day": {int(k): int(v) for k, v in z['demand_by_day'].items()},
        "demand_by_month": {int(k): int(v) for k, v in z['demand_by_month'].items()},
        "demand_peak_normalized": [float(v) for v in z['demand_peak_normalized']],
        "demand_mean": float(z['demand_mean']),
        "demand_std": float(z['demand_std']),
        "demand_percentile_90": float(z['demand_percentile_90'])
    }
    zones_json.append(zone_data)

# 3. Zone centers for spatial lookup (BallTree can't be serialized to JSON directly)
zone_centers = []
for z in zones_json:
    zone_centers.append({
        "zone_id": z['zone_id'],
        "latitude": z['center']['latitude'],
        "longitude": z['center']['longitude']
    })

# 4. Input/Output specification
io_spec = {
    "input": {
        "features": [
            {"name": "pickup_latitude", "type": "float", "min": 40.5, "max": 41.2, "required": True},
            {"name": "pickup_longitude", "type": "float", "min": -74.5, "max": -73.3, "required": True}
        ],
        "format": "JSON object with pickup_latitude and pickup_longitude"
    },
    "output": {
        "zone_id": {"type": "integer", "description": "Cluster/zone identifier (0-6)"},
        "distance_to_zone_center_km": {"type": "float", "description": "Haversine distance to zone center"},
        "zone_center": {
            "latitude": {"type": "float"},
            "longitude": {"type": "float"}
        },
        "demand_at_hour": {"type": "integer", "description": "Pickup count for requested hour (optional)"},
        "demand_level_normalized": {"type": "float", "description": "Normalized demand 0-1 vs zone peak (optional)"},
        "zone_quality": {
            "avg_membership_prob": {"type": "float", "description": "Cluster cohesion 0-1"},
            "zone_spread_km": {"type": "float", "description": "Spatial extent of zone"},
            "total_demand": {"type": "integer", "description": "Total pickups in training data"}
        }
    }
}

# 5. Sample test cases
sample_cases = [
    {
        "name": "Times Square",
        "input": {"pickup_latitude": 40.7580, "pickup_longitude": -73.9855},
        "expected_output": {
            "zone_id": 6,
            "distance_to_zone_center_km": 0.57,
            "zone_center": {"latitude": 40.7537, "longitude": -73.9818},
            "demand_at_hour_11": 13448,
            "demand_level_normalized_11": 0.76,
            "zone_quality": {"avg_membership_prob": 0.998, "zone_spread_km": 18.3, "total_demand": 266607}
        }
    },
    {
        "name": "JFK Airport",
        "input": {"pickup_latitude": 40.7282, "pickup_longitude": -73.7949},
        "expected_output": {
            "zone_id": 2,
            "distance_to_zone_center_km": 7.37,
            "zone_center": {"latitude": 40.7695, "longitude": -73.8633},
            "demand_at_hour_18": 2134,
            "demand_level_normalized_18": 0.99,
            "zone_quality": {"avg_membership_prob": 0.935, "zone_spread_km": 1.0, "total_demand": 32241}
        }
    },
    {
        "name": "LaGuardia Airport",
        "input": {"pickup_latitude": 40.7769, "pickup_longitude": -73.8740},
        "expected_output": {
            "zone_id": 4,
            "distance_to_zone_center_km": 0.35,
            "zone_center": {"latitude": 40.7739, "longitude": -73.8725},
            "demand_at_hour_14": 2043,
            "demand_level_normalized_14": 0.87,
            "zone_quality": {"avg_membership_prob": 0.989, "zone_spread_km": 1.1, "total_demand": 33918}
        }
    }
]

# 6. Dependencies
dependencies = {
    "python": ">=3.10",
    "numpy": ">=1.21",
    "scikit-learn": ">=1.0",
    "joblib": ">=1.1",
    "hdbscan": ">=0.8"
}

# Combine all
export_data = {
    "model_metadata": convert(model_metadata),
    "zones": convert(zones_json),
    "zone_centers": convert(zone_centers),
    "io_specification": convert(io_spec),
    "sample_test_cases": convert(sample_cases),
    "dependencies": dependencies,
    "usage_notes": {
        "distance_calculation": "Use haversine formula for distance to zone center",
        "nearest_zone_lookup": "Compute haversine distance from input point to all zone_centers, pick minimum",
        "bounds_check": "Verify input latitude in [40.5, 41.2] and longitude in [-74.5, -73.3]",
        "demand_query": "Use zones[zone_id]['demand_by_hour'][hour] for hourly demand",
        "demand_normalization": "Use zones[zone_id]['demand_peak_normalized'][hour] for 0-1 level"
    }
}

# Save main model JSON
with open('demand_zones_model.json', 'w') as f:
    json.dump(export_data, f, indent=2)

# Save zones only (for lighter weight)
with open('zones.json', 'w') as f:
    json.dump(convert(zones_json), f, indent=2)

# Save zone centers only (for spatial lookup)
with open('zone_centers.json', 'w') as f:
    json.dump(convert(zone_centers), f, indent=2)

# Save metadata only
with open('model_metadata.json', 'w') as f:
    json.dump(convert(model_metadata), f, indent=2)

print("Exported JSON files:")
print("  demand_zones_model.json  - Complete model data")
print("  zones.json               - Zone details only")
print("  zone_centers.json        - Centers for spatial lookup")
print("  model_metadata.json      - Model info only")
print(f"\nTotal zones: {len(zones_json)}")
print(f"Model size: {len(json.dumps(export_data)) / 1024:.1f} KB")