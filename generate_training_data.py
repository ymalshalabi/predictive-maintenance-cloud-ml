# Copyright (c) 2026 Yasmin Mazen AlShalabi
# SPDX-License-Identifier: MIT
"""
Synthetic Telemetry Generator for Cloud Predictive Maintenance
This script generates synthetic hardware metrics data for ML training and saves it to a CSV file.
The script runs continuously until interrupted by the user (Ctrl+C).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import signal
import sys
import os

class TrainingDataGenerator:
    def __init__(self, output_file='training_data.csv'):
        """
        Initialize the data generator
        
        Args:
            output_file: Path to save the CSV file
        """
        self.output_file = output_file
        self.running = True
        self.row_count = 0
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Define columns for the CSV
        self.columns = [
            'timestamp',
            'server_id',
            'server_type',
            'min_cpu',
            'max_cpu',
            'avg_cpu',
            'memory_usage',
            'disk_io',
            'network_latency',
            'temperature',
            'failure_within_1hr'
        ]
        
        # Server types and their characteristics
        self.server_types = {
            'WEB': {'cpu_base': 1500000, 'memory_base': 75, 'fail_rate': 0.1},
            'DB': {'cpu_base': 1800000, 'memory_base': 85, 'fail_rate': 0.15},
            'APP': {'cpu_base': 1200000, 'memory_base': 70, 'fail_rate': 0.08},
            'CACHE': {'cpu_base': 1000000, 'memory_base': 65, 'fail_rate': 0.05}
        }
        
        print("="*60)
        print("HARDWARE TRAINING DATA GENERATOR")
        print("="*60)
        print(f"Output file: {output_file}")
        print("Columns:", ", ".join(self.columns))
        print("\nGenerating data... Press Ctrl+C to stop")
        print("="*60 + "\n")
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C interrupt"""
        print(f"\n\n{'='*60}")
        print(f"INTERRUPTED BY USER")
        print(f"Total rows generated: {self.row_count:,}")
        print(f"Data saved to: {self.output_file}")
        print(f"{'='*60}")
        self.running = False
    
    def generate_row(self, timestamp, server_id):
        """Generate a single row of synthetic hardware data"""
        
        # Randomly select server type
        server_type = np.random.choice(list(self.server_types.keys()))
        characteristics = self.server_types[server_type]
        
        # Generate base metrics with some randomness
        hour = timestamp.hour
        is_business_hours = 9 <= hour <= 17
        
        # CPU metrics (higher during business hours)
        cpu_multiplier = 1.2 if is_business_hours else 0.8
        cpu_base = characteristics['cpu_base'] * cpu_multiplier
        
        min_cpu = max(500000, cpu_base * np.random.uniform(0.7, 0.9))
        max_cpu = max(min_cpu * 1.5, cpu_base * np.random.uniform(1.1, 1.3))
        avg_cpu = (min_cpu + max_cpu) / 2 * np.random.uniform(0.9, 1.1)
        
        # Other metrics
        memory_usage = characteristics['memory_base'] + np.random.uniform(-15, 15)
        disk_io = 500 + np.random.uniform(-200, 200)
        network_latency = 50 + (20 if is_business_hours else 0) + np.random.uniform(-15, 15)
        
        # Temperature correlated with CPU usage and time of day
        temperature = 60 + (avg_cpu / 50000) + (5 if is_business_hours else 0) + np.random.uniform(-3, 3)
        
        # Determine if failure will occur in next hour
        # Higher chance during high load, high temperature, or high memory usage
        failure_risk = characteristics['fail_rate']
        failure_risk += 0.1 if avg_cpu > 1800000 else 0
        failure_risk += 0.1 if temperature > 75 else 0
        failure_risk += 0.1 if memory_usage > 85 else 0
        
        failure_within_1hr = 1 if np.random.random() < failure_risk else 0
        
        # Create the row
        row = {
            'timestamp': timestamp,
            'server_id': server_id,
            'server_type': server_type,
            'min_cpu': round(min_cpu, 2),
            'max_cpu': round(max_cpu, 2),
            'avg_cpu': round(avg_cpu, 2),
            'memory_usage': round(max(10, min(100, memory_usage)), 2),
            'disk_io': round(max(100, min(1000, disk_io)), 2),
            'network_latency': round(max(10, min(200, network_latency)), 2),
            'temperature': round(max(30, min(100, temperature)), 2),
            'failure_within_1hr': failure_within_1hr
        }
        
        return row
    
    def print_row(self, row, row_number):
        """Print a formatted row to console"""
        status = "⚠️ FAILURE PREDICTED" if row['failure_within_1hr'] else "✅ Normal"
        print(f"Row {row_number:6,d} | {row['timestamp']} | "
              f"Server: {row['server_type']}-{row['server_id']:03d} | "
              f"CPU: {row['avg_cpu']/10000:6.1f}k | "
              f"Mem: {row['memory_usage']:5.1f}% | "
              f"Temp: {row['temperature']:5.1f}°C | "
              f"{status}")
    
    def save_batch(self, data_batch):
        """Save a batch of data to CSV"""
        df = pd.DataFrame(data_batch)
        
        # If file doesn't exist, write with header, otherwise append
        if not os.path.exists(self.output_file):
            df.to_csv(self.output_file, index=False)
        else:
            df.to_csv(self.output_file, mode='a', header=False, index=False)
    
    def run(self, servers=50, batch_size=100):
        """
        Main data generation loop
        
        Args:
            servers: Number of unique servers to simulate
            batch_size: Number of rows to accumulate before saving to disk
        """
        start_time = time.time()
        data_batch = []
        
        # Initial timestamp
        current_time = datetime.now() - timedelta(days=30)
        
        try:
            while self.running:
                # Generate data for each server
                for server_id in range(servers):
                    if not self.running:
                        break
                    
                    # Generate row
                    row = self.generate_row(current_time, server_id)
                    self.row_count += 1
                    
                    # Print to console
                    self.print_row(row, self.row_count)
                    
                    # Add to batch
                    data_batch.append(row)
                    
                    # Save batch when it reaches batch_size
                    if len(data_batch) >= batch_size:
                        self.save_batch(data_batch)
                        data_batch = []
                        print(f"  ↪ Saved batch to {self.output_file}")
                    
                    # Small delay to make output readable
                    time.sleep(0.01)
                
                # Advance time by 5 minutes for next cycle
                current_time += timedelta(minutes=5)
                
        except Exception as e:
            print(f"\nError: {e}")
            self.running = False
        
        finally:
            # Save any remaining data
            if data_batch:
                self.save_batch(data_batch)
                print(f"  ↪ Final batch saved to {self.output_file}")
            
            # Calculate statistics
            elapsed_time = time.time() - start_time
            rows_per_second = self.row_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n{'='*60}")
            print("DATA GENERATION COMPLETE")
            print(f"{'='*60}")
            print(f"Total rows:        {self.row_count:,}")
            print(f"Total servers:     {servers}")
            print(f"Output file:       {self.output_file}")
            print(f"File size:         {os.path.getsize(self.output_file) / 1024 / 1024:.2f} MB")
            print(f"Generation time:   {elapsed_time:.2f} seconds")
            print(f"Rows per second:   {rows_per_second:.2f}")
            
            if self.row_count > 0:
                # Load and show some statistics
                try:
                    df = pd.read_csv(self.output_file)
                    print(f"\nDATA SUMMARY:")
                    print(f"  Failures:        {df['failure_within_1hr'].sum():,} "
                          f"({df['failure_within_1hr'].mean()*100:.1f}%)")
                    print(f"  Time range:      {df['timestamp'].min()} to {df['timestamp'].max()}")
                    print(f"  Server types:    {df['server_type'].unique()}")
                    
                    # Show some sample rows
                    print(f"\nSAMPLE ROWS:")
                    print(df.head(3).to_string(index=False))
                    
                except Exception as e:
                    print(f"Could not read file for statistics: {e}")

def main():
    """Main function"""
    
    # Configuration
    OUTPUT_FILE = "training_data.csv"
    NUM_SERVERS = 50      # Number of unique servers to simulate
    BATCH_SIZE = 100      # Save to disk every X rows
    
    # Create generator and run
    generator = TrainingDataGenerator(output_file=OUTPUT_FILE)
    generator.run(servers=NUM_SERVERS, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()
