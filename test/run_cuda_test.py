import subprocess
import sys


# Check hardware before loading an executable linked against the CUDA driver.
try:
    subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    sys.exit(77)

sys.exit(subprocess.call(sys.argv[1:]))
