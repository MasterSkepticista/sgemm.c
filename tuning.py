import subprocess
import re
import csv
import itertools
import os
import random

def compile_and_run(kc, mc, nc, matrix_size=3840):
    """
    Compiles gemm.c with given KC, MC, NC and runs it with the specified matrix size.
    
    Args:
        kc (int): KC tile size.
        mc (int): MC tile size.
        nc (int): NC tile size.
        matrix_size (int): Argument to pass to ./gemm (default 1920).
    
    Returns:
        float: Extracted GFLOP/s value, or None if compilation or execution fails.
    """
    # Define compile command
    compile_cmd = [
        'clang', '-DKC={}'.format(kc), '-DMC={}'.format(mc), '-DNC={}'.format(nc),
        '-O2', '-march=native', 'gemm.c', '-o', './gemm'
    ]
    
    # Compile
    try:
        subprocess.run(compile_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed for KC={kc}, MC={mc}, NC={nc}: {e}")
        return None
    
    # Run executable
    run_cmd = ['./gemm', str(matrix_size)]
    try:
        result = subprocess.run(run_cmd, check=True, capture_output=True, text=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Execution failed for KC={kc}, MC={mc}, NC={nc}: {e}")
        return None
    
    # Parse GFLOP/s from output (e.g., "[M = 3840, K = 3840, N = 3840] GFLOP/s: 55.09")
    gflop_pattern = re.search(r'\[M = \d+, K = \d+, N = \d+\] GFLOP/s: ([\d.]+)', output)
    if gflop_pattern:
        return float(gflop_pattern.group(1))
    else:
        print(f"No GFLOP/s found in output for KC={kc}, MC={mc}, NC={nc}")
        return None

def grid_search_tuning(kc_range, mc_range, nc_range, output_file='gemm_tuning_results.csv'):
    """
    Performs grid search over KC, MC, NC ranges to find the configuration maximizing GFLOP/s.
    
    Args:
        kc_range (list): List of KC values to try.
        mc_range (list): List of MC values to try.
        nc_range (list): List of NC values to try.
        output_file (str): CSV file to save results.
    
    Returns:
        tuple: (best_kc, best_mc, best_nc, max_gflops)
    """
    results = []
    max_gflops = 0.0
    best_params = (0, 0, 0)
    
    print(f"Starting grid search: {len(kc_range)} x {len(mc_range)} x {len(nc_range)} = {len(kc_range)*len(mc_range)*len(nc_range)} trials")
    
    for kc, mc, nc in random.sample(list(itertools.product(kc_range, mc_range, nc_range)), k=len(kc_range)*len(mc_range)*len(nc_range)):
        print(f"Trying KC={kc}, MC={mc}, NC={nc}. Best so far: {max_gflops:.2f} GFLOP/s at {best_params}")
        gflops = compile_and_run(kc, mc, nc)
        if gflops is not None:
            results.append((kc, mc, nc, gflops))
            if gflops > max_gflops:
                max_gflops = gflops
                best_params = (kc, mc, nc)
            print(f"  GFLOP/s: {gflops:.2f}")
        else:
            results.append((kc, mc, nc, None))
    
    # Save results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['KC', 'MC', 'NC', 'GFLOP/s'])
        writer.writerows(results)
    
    print(f"\nGrid search complete. Results saved to {output_file}")
    print(f"Best configuration: KC={best_params[0]}, MC={best_params[1]}, NC={best_params[2]} with {max_gflops:.2f} GFLOP/s")
    
    return best_params + (max_gflops,)

# Example usage: Define ranges (adjust based on your hardware/experiments)
if __name__ == "__main__":
    sweep_range = [64, 128, 256, 512, 768, 960, 1024, 1536, 1920, 2048]
    grid_search_tuning(sweep_range, sweep_range, sweep_range)