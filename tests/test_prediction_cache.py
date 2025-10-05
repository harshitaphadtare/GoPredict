import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import time

from src.cache.prediction_cache import PredictionCache


class TestPredictionCache(unittest.TestCase):
    
    def setUp(self):
        """Set up test cache in temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = PredictionCache(
            cache_dir=self.temp_dir,
            ttl_hours=1,
            max_cache_size_mb=1
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_key_generation(self):
        """Test that same params generate same key."""
        params1 = {'lat': 40.7, 'lng': -73.9, 'hour': 14}
        params2 = {'hour': 14, 'lng': -73.9, 'lat': 40.7}  # Different order
        
        key1 = self.cache._generate_cache_key(params1)
        key2 = self.cache._generate_cache_key(params2)
        
        self.assertEqual(key1, key2, "Same params should generate same key")
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        route_params = {
            'pickup_lat': 40.7589,
            'pickup_lng': -73.9851,
            'dropoff_lat': 40.7614,
            'dropoff_lng': -73.9776
        }
        prediction = 15.5
        
        # Set cache
        self.cache.set(route_params, prediction)
        
        # Get from cache
        cached_value = self.cache.get(route_params)
        
        self.assertEqual(cached_value, prediction, "Cached value should match")
    
    def test_cache_miss(self):
        """Test cache miss for non-existent key."""
        route_params = {'lat': 40.7, 'lng': -73.9}
        
        cached_value = self.cache.get(route_params)
        
        self.assertIsNone(cached_value, "Should return None for cache miss")
    
    def test_cache_expiration(self):
        """Test that cache entries expire after TTL."""
        # Use very short TTL for testing
        cache = PredictionCache(
            cache_dir=self.temp_dir,
            ttl_hours=0.0001,  # ~0.36 seconds
            max_cache_size_mb=1
        )
        
        route_params = {'lat': 40.7, 'lng': -73.9}
        prediction = 15.5
        
        cache.set(route_params, prediction)
        
        # Should be in cache immediately
        self.assertIsNotNone(cache.get(route_params))
        
        # Wait for expiration
        time.sleep(0.5)
        
        # Should be expired now
        self.assertIsNone(cache.get(route_params))
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        route1 = {'lat': 40.7, 'lng': -73.9}
        route2 = {'lat': 40.8, 'lng': -73.8}
        
        # First request - miss
        self.cache.get(route1)
        
        # Set and get - hit
        self.cache.set(route1, 15.5)
        self.cache.get(route1)
        
        # Another miss
        self.cache.get(route2)
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['hits'], 1, "Should have 1 hit")
        self.assertEqual(stats['misses'], 2, "Should have 2 misses")
        self.assertEqual(stats['total_requests'], 3, "Should have 3 total requests")
    
    def test_cache_clear(self):
        """Test clearing cache."""
        route_params = {'lat': 40.7, 'lng': -73.9}
        
        self.cache.set(route_params, 15.5)
        self.assertIsNotNone(self.cache.get(route_params))
        
        # Clear cache
        self.cache.clear()
        
        # Should be gone
        self.assertIsNone(self.cache.get(route_params))
        
        # Stats should be reset
        stats = self.cache.get_stats()
        self.assertEqual(stats['total_requests'], 0)
    
    def test_max_cache_size(self):
        """Test that cache respects size limits."""
        # Create many cache entries
        for i in range(100):
            route = {'route_id': i, 'lat': 40.7 + i*0.001}
            self.cache.set(route, float(i))
        
        stats = self.cache.get_stats()
        cache_size_mb = float(stats['cache_size_mb'])
        
        # Should not exceed max size significantly
        self.assertLessEqual(
            cache_size_mb, 
            1.5,  # 1MB limit + some buffer
            "Cache should respect size limits"
        )


if __name__ == '__main__':
    unittest.main()