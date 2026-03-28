#!/usr/bin/env python3
"""Benchmark end-to-end inference latency."""

import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def benchmark_mstcn(model, input_shape, device, n_warmup=10, n_runs=100):
    """Benchmark MS-TCN inference latency."""
    # Create dummy input
    x = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    return {
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
        'mean': np.mean(times),
    }

def main():
    output_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/thesis_figures/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nBenchmarking MS-TCN inference latency...")
    
    # Load MS-TCN model
    sys.path.insert(0, '/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/MS-TCN')
    try:
        from model import MS_TCN2
    except ImportError:
        print("ERROR: Could not import MS_TCN2. Make sure the MS-TCN module is available.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Model config (adjust to match your setup)
    D = 384  # V-JEPA feature dim
    num_classes = 7  # 7-class setup
    model = MS_TCN2(
        num_layers_PG=10, num_layers_R=10, num_R=3,
        num_f_maps=64, dim=D, num_classes=num_classes
    ).to(device)
    model.eval()
    
    print("  Model loaded successfully")
    
    results = []
    
    # Benchmark for different sequence lengths
    seq_lengths = [100, 500, 1000, 3000, 6000]  # 10s, 50s, 100s, 5min, 10min at 10Hz
    
    for T in seq_lengths:
        print(f"\n  Benchmarking T={T} frames ({T/10:.0f}s)...")
        input_shape = (1, D, T)
        try:
            latency = benchmark_mstcn(model, input_shape, device, n_warmup=5, n_runs=50)
            
            results.append({
                'Sequence Length': T,
                'Duration (s)': T / 10,
                'P50 (ms)': latency['p50'],
                'P95 (ms)': latency['p95'],
                'P99 (ms)': latency['p99'],
                'Mean (ms)': latency['mean'],
                'Throughput (fps)': T / (latency['mean'] / 1000),
                'Per-frame (ms)': latency['mean'] / T,
            })
            print(f"    P50: {latency['p50']:.2f}ms, P95: {latency['p95']:.2f}ms, Throughput: {T / (latency['mean'] / 1000):.1f} fps")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    
    if results:
        df = pd.DataFrame(results)
        
        # Format for display
        df_display = df.copy()
        df_display['P50 (ms)'] = df_display['P50 (ms)'].apply(lambda x: f"{x:.2f}")
        df_display['P95 (ms)'] = df_display['P95 (ms)'].apply(lambda x: f"{x:.2f}")
        df_display['P99 (ms)'] = df_display['P99 (ms)'].apply(lambda x: f"{x:.2f}")
        df_display['Mean (ms)'] = df_display['Mean (ms)'].apply(lambda x: f"{x:.2f}")
        df_display['Throughput (fps)'] = df_display['Throughput (fps)'].apply(lambda x: f"{x:.1f}")
        df_display['Per-frame (ms)'] = df_display['Per-frame (ms)'].apply(lambda x: f"{x:.3f}")
        
        df_display.to_csv(output_dir / 'table7_latency.csv', index=False)
        
        print("\n" + "="*100)
        print("TABLE 7: End-to-End Latency Benchmarks")
        print("="*100)
        print(df_display.to_string(index=False))
        print("\n✓ Saved to:")
        print(f"  - {output_dir / 'table7_latency.csv'}")
        print(f"\nDevice: {device}")
        print(f"Target update rate: 100ms (10 Hz)")
        
        # Check if real-time capable
        per_frame_ms = df['Per-frame (ms)'].iloc[0]
        if per_frame_ms < 100:
            print(f"✓ System is real-time capable: {per_frame_ms:.3f}ms per frame < 100ms target")
        else:
            print(f"✗ System exceeds real-time budget: {per_frame_ms:.3f}ms per frame > 100ms target")
    else:
        print("\nERROR: No benchmark results generated!")

if __name__ == '__main__':
    main()


