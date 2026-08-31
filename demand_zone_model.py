"""
Demand Zone Model - Deployable inference module for NYC taxi demand zones.

Usage:
    from demand_zone_model import DemandZoneModel
    
    model = DemandZoneModel.load('demand_zones_model.pkl')
    zone_id, distance_km = model.predict(40.7580, -73.9855)
    demand = model.get_demand(zone_id, hour=11)
"""

import joblib
import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371


class DemandZoneModel:
    """Deployable demand zone model for NYC taxi pickups."""
    
    def __init__(self, artifact):
        self.zone_lookup = artifact['zone_lookup']
        self.feature_columns = artifact['feature_columns']
        self.model_type = artifact['model_type']
        self.eps_km = artifact['eps_km']
        self.min_samples = artifact['min_samples']
        self.nyc_bounds = artifact['nyc_bounds']
        self.version = artifact['version']
        
        self.zone_tree = self.zone_lookup['zone_tree']
        self.zone_ids = self.zone_lookup['zone_ids']
        self.zones = {z['zone_id']: z for z in self.zone_lookup['zones']}
    
    @classmethod
    def load(cls, filepath):
        """Load model from pickle file."""
        artifact = joblib.load(filepath)
        return cls(artifact)
    
    def predict(self, lat, lon):
        """
        Predict the demand zone for a given coordinate.
        
        Args:
            lat: Pickup latitude
            lon: Pickup longitude
            
        Returns:
            tuple: (zone_id, distance_to_zone_center_km)
        """
        point_rad = np.radians([[lat, lon]])
        distances, indices = self.zone_tree.query(point_rad, k=1)
        zone_idx = indices[0, 0]
        zone_id = int(self.zone_ids[zone_idx])
        distance_km = float(distances[0, 0] * EARTH_RADIUS_KM)
        return zone_id, distance_km
    
    def get_zone_info(self, zone_id):
        """Get full zone information."""
        return self.zones.get(int(zone_id))
    
    def get_demand(self, zone_id, hour=None, dayofweek=None):
        """
        Get demand level for a zone at specific hour/day.
        
        Args:
            zone_id: Zone identifier
            hour: Hour of day (0-23), optional
            dayofweek: Day of week (0=Monday), optional
            
        Returns:
            int: Demand count (number of pickups in training data)
        """
        zone = self.zones.get(int(zone_id))
        if not zone:
            return 0
        
        if hour is not None:
            return zone['demand_by_hour'].get(hour, 0)
        if dayofweek is not None:
            return zone['demand_by_day'].get(dayofweek, 0)
        return sum(zone['demand_by_hour'].values())
    
    def get_demand_level(self, zone_id, hour=None):
        """
        Get normalized demand level (0-1) for a zone at specific hour.
        
        Args:
            zone_id: Zone identifier
            hour: Hour of day (0-23), optional
            
        Returns:
            float: Normalized demand level (0.0 to 1.0)
        """
        zone = self.zones.get(int(zone_id))
        if not zone:
            return 0.0
        
        max_demand = max(zone['demand_by_hour'].values()) if zone['demand_by_hour'] else 1
        if hour is not None:
            return zone['demand_by_hour'].get(hour, 0) / max_demand
        return sum(zone['demand_by_hour'].values()) / (max_demand * 24)
    
    def list_zones(self):
        """List all zones with basic info."""
        return [
            {
                'zone_id': z['zone_id'],
                'center_lat': z['center_lat'],
                'center_lon': z['center_lon'],
                'zone_size': z['zone_size'],
                'peak_hour': max(z['demand_by_hour'], key=z['demand_by_hour'].get) if z['demand_by_hour'] else None
            }
            for z in self.zone_lookup['zones']
        ]
    
    def is_in_bounds(self, lat, lon):
        """Check if coordinate is within NYC bounds."""
        return (self.nyc_bounds['min_lat'] <= lat <= self.nyc_bounds['max_lat'] and
                self.nyc_bounds['min_lon'] <= lon <= self.nyc_bounds['max_lon'])


def predict_zone(zone_lookup, lat, lon):
    """Standalone prediction function (for backward compatibility)."""
    point_rad = np.radians([[lat, lon]])
    distances, indices = zone_lookup['zone_tree'].query(point_rad, k=1)
    zone_idx = indices[0, 0]
    zone_id = int(zone_lookup['zone_ids'][zone_idx])
    distance_km = float(distances[0, 0] * EARTH_RADIUS_KM)
    return zone_id, distance_km


def get_zone_demand(zone_lookup, zone_id, hour=None, dayofweek=None):
    """Standalone demand function (for backward compatibility)."""
    zone = next((z for z in zone_lookup['zones'] if z['zone_id'] == zone_id), None)
    if not zone:
        return 0
    if hour is not None:
        return zone['demand_by_hour'].get(hour, 0)
    if dayofweek is not None:
        return zone['demand_by_day'].get(dayofweek, 0)
    return sum(zone['demand_by_hour'].values())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model = DemandZoneModel.load(sys.argv[1])
    else:
        model = DemandZoneModel.load('demand_zones_model.pkl')
    
    print(f"Model: {model.model_type} v{model.version}")
    print(f"Zones: {len(model.zones)}")
    print(f"Bounds: {model.nyc_bounds}")
    print()
    
    for z in model.list_zones():
        print(f"  Zone {z['zone_id']}: center=({z['center_lat']:.4f}, {z['center_lon']:.4f}), "
              f"size={z['zone_size']}, peak_hour={z['peak_hour']}")
    
    print("\nTest predictions:")
    test_points = [
        (40.7580, -73.9855, "Times Square"),
        (40.7484, -73.9857, "Empire State"),
        (40.7128, -74.0060, "City Hall"),
        (40.7812, -73.9665, "Central Park"),
    ]
    
    for lat, lon, name in test_points:
        zone_id, dist = model.predict(lat, lon)
        demand_11 = model.get_demand(zone_id, hour=11)
        demand_18 = model.get_demand(zone_id, hour=18)
        level_11 = model.get_demand_level(zone_id, hour=11)
        level_18 = model.get_demand_level(zone_id, hour=18)
        print(f"  {name}: ({lat:.4f}, {lon:.4f}) -> Zone {zone_id} (dist={dist:.3f}km)")
        print(f"    Demand: 11am={demand_11} (level={level_11:.2f}), 6pm={demand_18} (level={level_18:.2f})")