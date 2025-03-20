import subprocess
import sys

# List of required packages
packages = [
    "Flask",
    "scikit-learn",
    "pandas",
    "joblib",
    "numpy"
]

# Install each package
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
print("All required libraries have been installed.")
