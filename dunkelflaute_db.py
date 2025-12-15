#!/usr/bin/env python3
"""
Database module for storing Dunkelflaute analysis results.

Supports:
- Multiple regions (Germany, Europe, etc.)
- Multiple model resolutions (ERA5, TCo1279, TCo319, etc.)
- Multiple scenarios (1950C, 2080C, historical, etc.)

Tables:
- runs: Metadata about each analysis run
- events: Individual dunkelflaute events
- annual_stats: Annual statistics per run
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import json


class DunkelflauteDB:
    """Database interface for Dunkelflaute results."""
    
    def __init__(self, db_path="dunkelflaute_results.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Runs table - metadata about each analysis run
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT NOT NULL,
                model TEXT NOT NULL,
                resolution TEXT,
                scenario TEXT,
                start_year INTEGER NOT NULL,
                end_year INTEGER NOT NULL,
                threshold REAL DEFAULT 0.06,
                moving_avg_hours INTEGER DEFAULT 48,
                min_duration_hours INTEGER DEFAULT 24,
                capacity_weights TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                UNIQUE(region, model, scenario, start_year, end_year)
            )
        """)
        
        # Events table - individual dunkelflaute events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                year INTEGER NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL,
                duration_hours REAL NOT NULL,
                mean_cf REAL NOT NULL,
                min_cf REAL NOT NULL,
                severity REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        # Annual statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annual_stats (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                year INTEGER NOT NULL,
                mean_cf REAL,
                std_cf REAL,
                min_cf REAL,
                max_cf REAL,
                hours_below_threshold INTEGER,
                n_events INTEGER,
                total_dunkelflaute_hours INTEGER,
                max_duration_hours REAL,
                max_severity REAL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id),
                UNIQUE(run_id, year)
            )
        """)
        
        # Create indices for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_year ON events(year)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_run ON annual_stats(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_region ON runs(region)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model)")
        
        self.conn.commit()
    
    def add_run(self, region, model, scenario, start_year, end_year,
                resolution=None, threshold=0.06, moving_avg_hours=48,
                min_duration_hours=24, capacity_weights=None, notes=None):
        """
        Add a new analysis run to the database.
        
        Returns:
            run_id: ID of the created/existing run
        """
        cursor = self.conn.cursor()
        
        # Check if run already exists
        cursor.execute("""
            SELECT run_id FROM runs 
            WHERE region=? AND model=? AND scenario=? AND start_year=? AND end_year=?
        """, (region, model, scenario, start_year, end_year))
        
        existing = cursor.fetchone()
        if existing:
            return existing['run_id']
        
        # Insert new run
        weights_json = json.dumps(capacity_weights) if capacity_weights else None
        cursor.execute("""
            INSERT INTO runs (region, model, resolution, scenario, start_year, end_year,
                            threshold, moving_avg_hours, min_duration_hours, 
                            capacity_weights, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (region, model, resolution, scenario, start_year, end_year,
              threshold, moving_avg_hours, min_duration_hours, weights_json, notes))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def add_events(self, run_id, events):
        """
        Add dunkelflaute events for a run.
        
        Args:
            run_id: ID of the run
            events: List of event dicts with keys: start, end, duration_hours, 
                   mean_cf, min_cf, severity
        """
        cursor = self.conn.cursor()
        
        for evt in events:
            year = evt['start'].year if hasattr(evt['start'], 'year') else evt.get('year')
            cursor.execute("""
                INSERT INTO events (run_id, year, start_time, end_time, duration_hours,
                                   mean_cf, min_cf, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, year, str(evt['start']), str(evt['end']), 
                  evt['duration_hours'], evt['mean_cf'], evt['min_cf'], evt['severity']))
        
        self.conn.commit()
    
    def add_annual_stats(self, run_id, year, stats):
        """
        Add annual statistics for a run/year.
        
        Args:
            run_id: ID of the run
            year: Year
            stats: Dict with keys: mean_cf, std_cf, min_cf, max_cf, 
                   hours_below_threshold, n_events, total_hours, 
                   max_duration_hours, max_severity
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO annual_stats 
            (run_id, year, mean_cf, std_cf, min_cf, max_cf, hours_below_threshold,
             n_events, total_dunkelflaute_hours, max_duration_hours, max_severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, year, stats.get('mean_cf'), stats.get('std_cf'),
              stats.get('min_cf'), stats.get('max_cf'), stats.get('hours_below_threshold'),
              stats.get('n_events'), stats.get('total_hours'), 
              stats.get('max_duration_hours'), stats.get('max_severity')))
        
        self.conn.commit()
    
    def get_events(self, region=None, model=None, scenario=None, 
                   start_year=None, end_year=None):
        """Query events with optional filters. Returns DataFrame."""
        query = """
            SELECT e.*, r.region, r.model, r.scenario, r.resolution
            FROM events e
            JOIN runs r ON e.run_id = r.run_id
            WHERE 1=1
        """
        params = []
        
        if region:
            query += " AND r.region = ?"
            params.append(region)
        if model:
            query += " AND r.model = ?"
            params.append(model)
        if scenario:
            query += " AND r.scenario = ?"
            params.append(scenario)
        if start_year:
            query += " AND e.year >= ?"
            params.append(start_year)
        if end_year:
            query += " AND e.year <= ?"
            params.append(end_year)
        
        query += " ORDER BY e.start_time"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_annual_stats(self, region=None, model=None, scenario=None):
        """Query annual statistics with optional filters. Returns DataFrame."""
        query = """
            SELECT s.*, r.region, r.model, r.scenario, r.resolution
            FROM annual_stats s
            JOIN runs r ON s.run_id = r.run_id
            WHERE 1=1
        """
        params = []
        
        if region:
            query += " AND r.region = ?"
            params.append(region)
        if model:
            query += " AND r.model = ?"
            params.append(model)
        if scenario:
            query += " AND r.scenario = ?"
            params.append(scenario)
        
        query += " ORDER BY r.region, r.model, r.scenario, s.year"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_runs(self):
        """Get all runs. Returns DataFrame."""
        return pd.read_sql_query("SELECT * FROM runs ORDER BY created_at DESC", self.conn)
    
    def compare_scenarios(self, region, model, scenario1, scenario2):
        """
        Compare statistics between two scenarios.
        
        Returns:
            DataFrame with comparison statistics
        """
        stats1 = self.get_annual_stats(region=region, model=model, scenario=scenario1)
        stats2 = self.get_annual_stats(region=region, model=model, scenario=scenario2)
        
        if stats1.empty or stats2.empty:
            return pd.DataFrame()
        
        comparison = pd.DataFrame({
            'metric': ['mean_cf', 'n_events_per_year', 'total_hours_per_year', 
                      'max_duration', 'max_severity'],
            f'{scenario1}_mean': [
                stats1['mean_cf'].mean(),
                stats1['n_events'].mean(),
                stats1['total_dunkelflaute_hours'].mean(),
                stats1['max_duration_hours'].max(),
                stats1['max_severity'].max()
            ],
            f'{scenario2}_mean': [
                stats2['mean_cf'].mean(),
                stats2['n_events'].mean(),
                stats2['total_dunkelflaute_hours'].mean(),
                stats2['max_duration_hours'].max(),
                stats2['max_severity'].max()
            ]
        })
        
        comparison['difference'] = comparison[f'{scenario2}_mean'] - comparison[f'{scenario1}_mean']
        comparison['pct_change'] = (comparison['difference'] / comparison[f'{scenario1}_mean'] * 100).round(1)
        
        return comparison
    
    def export_to_csv(self, output_dir):
        """Export all tables to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        self.get_runs().to_csv(output_dir / "runs.csv", index=False)
        
        events = pd.read_sql_query("""
            SELECT e.*, r.region, r.model, r.scenario
            FROM events e JOIN runs r ON e.run_id = r.run_id
        """, self.conn)
        events.to_csv(output_dir / "events.csv", index=False)
        
        stats = pd.read_sql_query("""
            SELECT s.*, r.region, r.model, r.scenario
            FROM annual_stats s JOIN runs r ON s.run_id = r.run_id
        """, self.conn)
        stats.to_csv(output_dir / "annual_stats.csv", index=False)
        
        print(f"Exported to {output_dir}")
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def save_results_to_db(db_path, region, model, scenario, start_year, end_year,
                       events_by_year, stats_by_year, resolution=None,
                       threshold=0.06, capacity_weights=None):
    """
    Convenience function to save analysis results to database.
    
    Args:
        db_path: Path to SQLite database
        region: Region name (e.g., 'Germany', 'Europe')
        model: Model name (e.g., 'ERA5', 'TCo1279-DART')
        scenario: Scenario (e.g., '1950C', '2080C', 'historical')
        start_year, end_year: Year range
        events_by_year: Dict {year: [list of events]}
        stats_by_year: Dict {year: stats_dict}
        resolution: Optional resolution string
        threshold: CF threshold used
        capacity_weights: Dict of capacity weights used
    """
    db = DunkelflauteDB(db_path)
    
    run_id = db.add_run(
        region=region,
        model=model,
        scenario=scenario,
        start_year=start_year,
        end_year=end_year,
        resolution=resolution,
        threshold=threshold,
        capacity_weights=capacity_weights
    )
    
    # Add events
    all_events = []
    for year, events in events_by_year.items():
        for evt in events:
            evt['year'] = year
            all_events.append(evt)
    
    if all_events:
        db.add_events(run_id, all_events)
    
    # Add annual stats
    for year, stats in stats_by_year.items():
        db.add_annual_stats(run_id, year, stats)
    
    db.close()
    print(f"Saved results to {db_path} (run_id={run_id})")
    return run_id


if __name__ == "__main__":
    # Example usage
    db = DunkelflauteDB("dunkelflaute_results.db")
    
    print("Runs:")
    print(db.get_runs())
    
    print("\nEvents:")
    print(db.get_events())
    
    print("\nAnnual Stats:")
    print(db.get_annual_stats())
    
    db.close()
