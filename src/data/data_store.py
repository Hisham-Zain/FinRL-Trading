"""
Data Store Module
================

Manages data persistence and caching:
- Local file storage
- Database operations
- Cache management
- Data versioning
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import json
import sqlite3

import pandas as pd

logger = logging.getLogger(__name__)


class DataStore:
    """Data store for managing persistent data storage."""

    def __init__(self, base_dir: str = "./data"):
        """
        Initialize data store.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / "cache"
        self.processed_dir = self.base_dir / "processed"
        self.db_path = self.base_dir / "finrl_trading.db"

        # Create directories
        for dir_path in [self.base_dir, self.cache_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT,
                    metadata TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    cache_key TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    file_path TEXT,
                    metadata TEXT
                )
            ''')

            conn.commit()

    def save_dataframe(self, df: pd.DataFrame, name: str,
                      version: str = None, metadata: Dict = None) -> str:
        """
        Save DataFrame to storage with versioning.

        Args:
            df: DataFrame to save
            name: Name identifier for the data
            version: Version string (auto-generated if None)
            metadata: Additional metadata

        Returns:
            Path to saved file
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{name}_{version}.csv"
        file_path = self.processed_dir / filename

        # Save DataFrame
        df.to_csv(file_path, index=False)
        logger.info(f"Saved DataFrame to {file_path}")

        # Save version info to database
        self._save_version_info(name, version, str(file_path), metadata)

        return str(file_path)

    def load_dataframe(self, name: str, version: str = None) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from storage.

        Args:
            name: Name identifier for the data
            version: Specific version to load (latest if None)

        Returns:
            Loaded DataFrame or None if not found
        """
        file_path = self._get_file_path(name, version)
        if file_path and file_path.exists():
            logger.info(f"Loading DataFrame from {file_path}")
            return pd.read_csv(file_path)
        return None

    def _save_version_info(self, data_type: str, version: str,
                          file_path: str, metadata: Dict = None):
        """Save version information to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO data_versions (data_type, version, file_path, metadata)
                VALUES (?, ?, ?, ?)
            ''', (data_type, version, file_path, json.dumps(metadata) if metadata else None))
            conn.commit()

    def _get_file_path(self, name: str, version: str = None) -> Optional[Path]:
        """Get file path for given name and version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if version:
                cursor.execute('''
                    SELECT file_path FROM data_versions
                    WHERE data_type = ? AND version = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (name, version))
            else:
                cursor.execute('''
                    SELECT file_path FROM data_versions
                    WHERE data_type = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (name,))

            result = cursor.fetchone()
            return Path(result[0]) if result else None

    def list_versions(self, data_type: str) -> List[Dict]:
        """List all versions for a data type."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT version, created_at, metadata FROM data_versions
                WHERE data_type = ?
                ORDER BY created_at DESC
            ''', (data_type,))

            versions = []
            for row in cursor.fetchall():
                versions.append({
                    'version': row[0],
                    'created_at': row[1],
                    'metadata': json.loads(row[2]) if row[2] else None
                })

            return versions

    def cache_data(self, key: str, data: Any, ttl_hours: int = 24) -> str:
        """
        Cache data with time-to-live.

        Args:
            key: Cache key
            data: Data to cache (DataFrame, dict, etc.)
            ttl_hours: Time-to-live in hours

        Returns:
            Path to cached file
        """
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        filename = f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        file_path = self.cache_dir / filename

        # Save data based on type
        if isinstance(data, pd.DataFrame):
            data.to_pickle(file_path)
        elif isinstance(data, dict):
            with open(file_path, 'w') as f:
                json.dump(data, f)
        else:
            pd.to_pickle(data, file_path)

        # Save cache metadata
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO cache_metadata
                (cache_key, expires_at, file_path)
                VALUES (?, ?, ?)
            ''', (key, expires_at.isoformat(), str(file_path)))
            conn.commit()

        logger.info(f"Cached data with key '{key}' to {file_path}")
        return str(file_path)

    def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Retrieve cached data if not expired.

        Args:
            key: Cache key

        Returns:
            Cached data or None if expired/not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT file_path, expires_at FROM cache_metadata
                WHERE cache_key = ?
            ''', (key,))

            result = cursor.fetchone()
            if not result:
                return None

            file_path, expires_at = result
            if datetime.now() > datetime.fromisoformat(expires_at):
                logger.info(f"Cache expired for key '{key}'")
                return None

            file_path = Path(file_path)
            if not file_path.exists():
                return None

            # Load data based on file extension
            if file_path.suffix == '.pkl':
                return pd.read_pickle(file_path)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                return pd.read_csv(file_path)

    def cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get expired cache entries
            cursor.execute('''
                SELECT cache_key, file_path FROM cache_metadata
                WHERE expires_at < ?
            ''', (datetime.now().isoformat(),))

            expired_entries = cursor.fetchall()

            # Delete files and database entries
            for key, file_path in expired_entries:
                try:
                    Path(file_path).unlink(missing_ok=True)
                    logger.info(f"Deleted expired cache file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {file_path}: {e}")

            # Remove from database
            cursor.execute('''
                DELETE FROM cache_metadata
                WHERE expires_at < ?
            ''', (datetime.now().isoformat(),))

            conn.commit()

            logger.info(f"Cleaned up {len(expired_entries)} expired cache entries")

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        total_size = 0
        file_count = 0

        for dir_path in [self.cache_dir, self.processed_dir]:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

        # Database stats
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM data_versions")
            version_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM cache_metadata")
            cache_count = cursor.fetchone()[0]

        return {
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'data_versions': version_count,
            'cache_entries': cache_count
        }


# Global data store instance
_data_store = None

def get_data_store(base_dir: str = "./data") -> DataStore:
    """Get global data store instance."""
    global _data_store
    if _data_store is None:
        _data_store = DataStore(base_dir)
    return _data_store


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    store = get_data_store()

    # Create sample data
    sample_df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'price': [150.0, 280.0, 2500.0],
        'sector': ['Technology', 'Technology', 'Technology']
    })

    # Save and load data
    file_path = store.save_dataframe(sample_df, "sample_stocks")
    print(f"Saved data to: {file_path}")

    loaded_df = store.load_dataframe("sample_stocks")
    print(f"Loaded data shape: {loaded_df.shape}")

    # Cache data
    store.cache_data("test_cache", {"key": "value"})

    # Get storage stats
    stats = store.get_storage_stats()
    print(f"Storage stats: {stats}")
