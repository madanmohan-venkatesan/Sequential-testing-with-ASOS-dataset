#!/usr/bin/env python3
"""
Script to kill processes running on ports 8000 and 8501.
"""

import subprocess
import sys

def kill_port(port):
    """Kill process running on specified port."""
    try:
        # Find process ID using netstat
        result = subprocess.run(
            f'netstat -ano | findstr ":{port}"',
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            pids = set()
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid.isdigit():
                        pids.add(pid)
            
            # Kill each process
            for pid in pids:
                try:
                    subprocess.run(f'taskkill /PID {pid} /F', shell=True, check=True)
                    print(f"Killed process {pid} on port {port}")
                except subprocess.CalledProcessError:
                    print(f"Failed to kill process {pid} on port {port}")
        else:
            print(f"No processes found on port {port}")
            
    except Exception as e:
        print(f"Error killing processes on port {port}: {e}")

if __name__ == "__main__":
    print("Killing processes on ports 8000 and 8501...")
    kill_port(8000)
    kill_port(8501)
    print("Done!")
