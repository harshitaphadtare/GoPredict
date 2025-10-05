"""
Prediction Cache System for GoPredict
Implements caching for frequent route predictions to improve performance
"""

import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PredictionCache:
    """
    Cache system for storing and retrieving trip duration predictions.
    
    Features:
    - Hash-based key generation from route parameters
    - TTL (Time-to-Live) support for cache expiration
    - Disk-based persistence
    - Cache statistics tracking
    """
    
    def __init__(
        self, 
        cache_dir: str = "cache", 
        ttl_hours: int = 24,
        max_cache_size_mb: int = 100
    ):
        """
        Initialize the prediction cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cache entries in hours
            max_cache_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.ttl = timedelta(hours=ttl_hours)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
        self._load_stats()
        
    def _generate_cache_key(self, route_params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key from route parameters.
        
        Args:
            route_params: Dictionary containing route features
            
        Returns:
            SHA256 hash of the route parameters
        """
        # Sort keys for consistent hashing
        sorted_params = dict(sorted(route_params.items()))
        param_string = json.dumps(sorted_params, sort_keys=True)
        return hashlib.sha256(param_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, route_params: Dict[str, Any]) -> Optional[float]:
        """
        Retrieve prediction from cache if available and not expired.
        
        Args:
            route_params: Dictionary containing route features
            
        Returns:
            Cached prediction value or None if not found/expired
        """
        self.stats['total_requests'] += 1
        
        cache_key = self._generate_cache_key(route_params)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            self.stats['misses'] += 1
            logger.debug(f"Cache miss for key: {cache_key[:8]}...")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_entry = pickle.load(f)
            
            # Check if cache entry has expired
            if datetime.now() - cache_entry['timestamp'] > self.ttl:
                logger.debug(f"Cache expired for key: {cache_key[:8]}...")
                cache_path.unlink()  # Delete expired entry
                self.stats['misses'] += 1
                return None
            
            self.stats['hits'] += 1
            logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            return cache_entry['prediction']
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.stats['misses'] += 1
            return None
    
    def set(self, route_params: Dict[str, Any], prediction: float) -> None:
        """
        Store prediction in cache.
        
        Args:
            route_params: Dictionary containing route features
            prediction: Predicted trip duration
        """
        cache_key = self._generate_cache_key(route_params)
        cache_path = self._get_cache_path(cache_key)
        
        cache_entry = {
            'prediction': prediction,
            'timestamp': datetime.now(),
            'route_params': route_params
        }
        
        try:
            # Check cache size before adding
            self._manage_cache_size()
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_entry, f)
            
            logger.debug(f"Cached prediction for key: {cache_key[:8]}...")
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _manage_cache_size(self) -> None:
        """Remove oldest cache entries if size limit exceeded."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        # Calculate total cache size
        total_size = sum(f.stat().st_size for f in cache_files)
        
        if total_size > self.max_cache_size:
            # Sort by modification time and remove oldest
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            
            while total_size > self.max_cache_size * 0.8:  # Remove to 80% capacity
                if not cache_files:
                    break
                    
                oldest_file = cache_files.pop(0)
                file_size = oldest_file.stat().st_size
                oldest_file.unlink()
                total_size -= file_size
                
                logger.info(f"Removed old cache entry: {oldest_file.name}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        self.stats = {'hits': 0, 'misses': 0, 'total_requests': 0}
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing hit rate and other stats
        """
        hit_rate = (
            self.stats['hits'] / self.stats['total_requests'] 
            if self.stats['total_requests'] > 0 
            else 0
        )
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        cache_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'total_requests': self.stats['total_requests'],
            'hit_rate': f"{hit_rate:.2%}",
            'cache_entries': len(cache_files),
            'cache_size_mb': f"{cache_size_mb:.2f}"
        }
    
    def _load_stats(self) -> None:
        """Load statistics from disk."""
        stats_path = self.cache_dir / "stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load stats: {e}")
    
    def _save_stats(self) -> None:
        """Save statistics to disk."""
        stats_path = self.cache_dir / "stats.json"
        try:
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f)
        except Exception as e:
            logger.warning(f"Could not save stats: {e}")
    
    def __del__(self):
        """Save stats on object destruction."""
        self._save_stats()


def predict_with_cache(
    model: Any,
    route_params: Dict[str, Any],
    cache: PredictionCache,
    features_df: Optional[pd.DataFrame] = None
) -> float:
    """
    Predict trip duration with caching support.
    
    Args:
        model: Trained prediction model
        route_params: Dictionary of route parameters
        cache: PredictionCache instance
        features_df: Pre-computed features DataFrame (optional)
        
    Returns:
        Predicted trip duration
    """
    # Try to get from cache first
    cached_prediction = cache.get(route_params)
    if cached_prediction is not None:
        return cached_prediction
    
    # If not in cache, make prediction
    if features_df is not None:
        prediction = model.predict(features_df)[0]
    else:
        # Convert route_params to DataFrame for prediction
        df = pd.DataFrame([route_params])
        prediction = model.predict(df)[0]
    
    # Store in cache
    cache.set(route_params, float(prediction))
    
    return float(prediction)


# Example usage
if __name__ == "__main__":
    # Initialize cache
    cache = PredictionCache(
        cache_dir="cache/predictions",
        ttl_hours=24,
        max_cache_size_mb=50
    )
    
    # Example route parameters
    route = {
        'pickup_latitude': 40.7589,
        'pickup_longitude': -73.9851,
        'dropoff_latitude': 40.7614,
        'dropoff_longitude': -73.9776,
        'passenger_count': 1,
        'hour': 14,
        'day_of_week': 3
    }
    
    # Simulate prediction
    prediction = 15.5  # minutes
    cache.set(route, prediction)
    
    # Retrieve from cache
    cached = cache.get(route)
    print(f"Cached prediction: {cached}")
    
    # Display stats
    print(f"Cache statistics: {cache.get_stats()}")