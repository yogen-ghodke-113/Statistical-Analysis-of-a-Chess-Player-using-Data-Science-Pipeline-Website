import concurrent.futures
import os
import subprocess
from typing import List, Tuple
import sys
import time

def run_visualization_module(args: Tuple[str, str, str, str]) -> Tuple[str, float, str, str]:
    """Run a single visualization module and return results"""
    python_exe, module_file, username, description = args
    module_start = time.time()
    
    try:
        result = subprocess.run(
            [python_exe, module_file, username],
            check=True,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        duration = time.time() - module_start
        return (description, duration, result.stdout, "")
    except subprocess.TimeoutExpired:
        return (description, time.time() - module_start, "", "Process timed out after 60 seconds")
    except Exception as e:
        return (description, time.time() - module_start, "", str(e))

def visualize_data(username: str) -> None:
    """Visualize chess data using parallel processing"""
    try:
        print("\nStarting visualization process...")
        start_time = time.time()
        
        modules: List[Tuple[str, str]] = [
            ("unt.py", "game analysis"),
            ("heatmap1.py", "heatmap analysis 1"),
            ("heatmap2.py", "heatmap analysis 2"),
            ("heatmap3.py", "correlation analysis")
        ]
        
        python_exe = sys.executable
        print(f"Using Python executable: {python_exe}")
        
        # Prepare arguments for parallel processing
        args = [(python_exe, module[0], username, module[1]) for module in modules]
        
        # Run visualizations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_visualization_module, arg) for arg in args]
            
            for future in concurrent.futures.as_completed(futures):
                description, duration, stdout, error = future.result()
                if error:
                    print(f"✗ Error in {description} after {duration:.2f} seconds:")
                    print(f"Error: {error}")
                else:
                    print(f"✓ Completed {description} in {duration:.2f} seconds")
                    if stdout:
                        print(f"Output:\n{stdout}")
        
        total_duration = time.time() - start_time
        print(f"\nVisualization process completed in {total_duration:.2f} seconds")
        
    except Exception as e:
        print(f"\n✗ Fatal error in visualization process: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_data(sys.argv[1])
    else:
        print("Please provide a username as argument")
