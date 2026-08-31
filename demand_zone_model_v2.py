"""
Demand Zone Model v2 - Optimized inference with demand forecasting
"""

import joblib
import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371


class DemandZoneModelV2:
    """Optimized deployable demand zone model with demand forecasting."""
    
    def __init__(self, artifact):
        self.zone_lookup = artifact['zone_lookup']
        self.feature_columns = artifact['feature_columns']
        self.model_type = artifact['model_type']
        self.version = artifact['version']
        self.nyc_bounds = artifact['nyc_bounds']
        self.training_samples = artifact.get('training_samples', 0)
        self.silhouette_score = artifact.get('silhouette_score')
        
        self.zone_tree = self.zone_lookup['zone_tree']
        self.zone_ids = self.zone_lookup['zone_ids']
        self.zones = {z['zone_id']: z for z in self.zone_lookup['zones']}
        self.n_zones = self.zone_lookup['n_zones']
    
    @classmethod
    def load(cls, filepath):
        artifact = joblib.load(filepath)
        return cls(artifact)
    
    def predict(self, lat, lon, k=1):
        """
        Predict nearest zone(s) for a coordinate.
        
        Args:
            lat: Pickup latitude
            lon: Pickup longitude
            k: Number of nearest zones to return
            
        Returns:
            If k=1: (zone_id, distance_km)
            If k>1: [(zone_id, distance_km), ...]
        """
        point_rad = np.radians([[lat, lon]])
        distances, indices = self.zone_tree.query(point_rad, k=k)
        
        if k == 1:
            zone_idx = indices[0, 0]
            zone_id = int(self.zone_ids[zone_idx])
            distance_km = float(distances[0, 0] * EARTH_RADIUS_KM)
            return zone_id, distance_km
        else:
            results = []
            for i in range(k):
                zone_idx = indices[0, i]
                zone_id = int(self.zone_ids[zone_idx])
                distance_km = float(distances[0, i] * EARTH_RADIUS_KM)
                results.append((zone_id, distance_km))
            return results
    
    def get_zone_info(self, zone_id):
        return self.zones.get(int(zone_id))
    
    def get_demand(self, zone_id, hour=None, dayofweek=None, month=None):
        zone = self.zones.get(int(zone_id))
        if not zone:
            return 0
        
        if hour is not None:
            return zone['demand_by_hour'].get(hour, 0)
        if dayofweek is not None:
            return zone['demand_by_day'].get(dayofweek, 0)
        if month is not None:
            return zone['demand_by_month'].get(month, 0)
        return zone['total_demand']
    
    def get_demand_level(self, zone_id, hour=None):
        """Normalized demand 0-1 relative to zone's peak hour"""
        zone = self.zones.get(int(zone_id))
        if not zone:
            return 0.0
        
        if hour is not None:
            return zone['demand_peak_normalized'][hour]
        return zone['demand_mean'] / zone['demand_percentile_90'] if zone['demand_percentile_90'] > 0 else 0.0
    
    def get_demand_forecast(self, zone_id, hours_ahead=24):
        """Get demand forecast for next N hours (cyclic daily pattern)"""
        zone = self.zones.get(int(zone_id))
        if not zone:
            return []
        
        current_hour = np.datetime64('now', 'h').astype(int) % 24
        forecast = []
        for h in range(hours_ahead):
            hour = (current_hour + h) % 24
            forecast.append({
                'hour': hour,
                'demand': zone['demand_by_hour'].get(hour, 0),
                'demand_level': zone['demand_peak_normalized'][hour]
            })
        return forecast
    
    def get_zone_quality(self, zone_id):
        """Get zone quality metrics"""
        zone = self.zones.get(int(zone_id))
        if not zone:
            return None
        return {
            'avg_membership_prob': zone['avg_membership_prob'],
            'zone_spread_km': zone['zone_spread_km'],
            'total_demand': zone['total_demand'],
            'silhouette_score': self.silhouette_score
        }
    
    def list_zones(self, sort_by='total_demand'):
        zones_list = list(self.zones.values())
        if sort_by == 'total_demand':
            zones_list.sort(key=lambda z: z['total_demand'], reverse=True)
        elif sort_by == 'zone_id':
            zones_list.sort(key=lambda z: z['zone_id'])
        return [
            {
                'zone_id': z['zone_id'],
                'center_lat': z['center_lat'],
                'center_lon': z['center_lon'],
                'zone_size': z['zone_size'],
                'total_demand': z['total_demand'],
                'peak_hour': z['peak_hour'],
                'peak_day': z['peak_day'],
                'avg_membership_prob': z['avg_membership_prob'],
                'zone_spread_km': z['zone_spread_km']
            }
            for z in zones_list
        ]
    
    def is_in_bounds(self, lat, lon):
        return (self.nyc_bounds['min_lat'] <= lat <= self.nyc_bounds['max_lat'] and
                self.nyc_bounds['min_lon'] <= lon <= self.nyc_bounds['max_lon'])
    
    def get_model_info(self):
        return {
            'model_type': self.model_type,
            'version': self.version,
            'n_zones': self.n_zones,
            'training_samples': self.training_samples,
            'silhouette_score': self.silhouette_score,
            'feature_columns': self.feature_columns,
            'nyc_bounds': self.nyc_bounds
        }


# Backward compatibility functions
def predict_zone(zone_lookup, lat, lon):
    point_rad = np.radians([[lat, lon]])
    distances, indices = zone_lookup['zone_tree'].query(point_rad, k=1)
    zone_idx = indices[0, 0]
    zone_id = int(zone_lookup['zone_ids'][zone_idx])
    distance_km = float(distances[0, 0] * EARTH_RADIUS_KM)
    return zone_id, distance_km


def get_zone_demand(zone_lookup, zone_id, hour=None, dayofweek=None, month=None):
    zone = next((z for z in zone_lookup['zones'] if z['zone_id'] == zone_id), None)
    if not zone:
        return 0
    if hour is not None:
        return zone['demand_by_hour'].get(hour, 0)
    if dayofweek is not None:
        return zone['demand_by_day'].get(dayofweek, 0)
    if month is not None:
        return zone['demand_by_month'].get(month, 0)
    return zone['total_demand']


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'demand_zones_model_optimized.pkl'
    model = DemandZoneModelV2.load(model_path)
    
    info = model.get_model_info()
    print(f"Model: {info['model_type']} v{info['version']}")
    print(f"Zones: {info['n_zones']}, Training samples: {info['training_samples']:,}")
    print(f"Silhouette: {info['silhouette_score']:.4f}" if info['silhouette_score'] else "Silhouette: N/A")
    print(f"Bounds: {info['nyc_bounds']}")
    print()
    
    for z in model.list_zones():
        print(f"  Zone {z['zone_id']}: center=({z['center_lat']:.4f}, {z['center_lon']:.4f}), "
              f"demand={z['total_demand']:,}, peak={z['peak_hour']}:00, "
              f"spread={z['zone_spread_km']:.1f}km, prob={z['avg_membership_prob']:.3f}")
    
    print("\nTest predictions:")
    test_points = [
        (40.7580, -73.9855, "Times Square"),
        (40.7484, -73.9857, "Empire State"),
        (40.7128, -74.0060, "City Hall"),
        (40.7812, -73.9665, "Central Park"),
        (40.7282, -73.7949, "JFK Airport"),
        (40.7769, -73.8740, "LaGuardia Airport"),
    ]
    
    for lat, lon, name in test_points:
        zone_id, dist = model.predict(lat, lon)
        z = model.get_zone_info(zone_id)
        print(f"  {name}: Zone {zone_id} (dist={dist:.3f}km, center=({z['center_lat']:.4f}, {z['center_lon']:.4f}))")
        print(f"    Demand 11am: {model.get_demand(zone_id, hour=11)} (level={model.get_demand_level(zone_id, hour=11):.2f})")
        print(f"    Demand 6pm:  {model.get_demand(zone_id, hour=18)} (level={model.get_demand_level(zone_id, hour=18):.2f})")
        print(f"    Quality: prob={z['avg_membership_prob']:.3f}, spread={z['zone_spread_km']:.1f}km")