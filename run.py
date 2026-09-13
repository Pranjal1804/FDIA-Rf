import sys
import os

def run():
    sys.stdout = open('/home/pranjal/Desktop/fdia/install.log', 'w')
    sys.stderr = sys.stdout

    import subprocess
    print("Starting raw pip install...", flush=True)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "grid2op", "pandapower", "numpy", "scipy"], check=True)
        print("Install success.", flush=True)
    except Exception as e:
        print(f"Install failed: {e}", flush=True)
        return

    print("Starting generation...", flush=True)
    try:
        from src.simulation import generate_data
        generate_data.main()
        print("Generation success.", flush=True)
    except Exception as e:
        print(f"Generation failed: {e}", flush=True)

if __name__ == "__main__":
    run()
