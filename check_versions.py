import sys
import pkg_resources
import importlib

# List of libraries to check
libraries = [
    "streamlit",
    "tensorflow",
    "numpy",
    "opencv-python-headless",  # imported as cv2
    "pillow"  # imported as PIL
]

print(f"Python version: {sys.version}")
print("\nInstalled package versions:")
print("-" * 50)

for lib in libraries:
    try:
        # Get the installed version from pkg_resources
        version = pkg_resources.get_distribution(lib).version
        print(f"{lib}: {version}")
        
        # For libraries with different import names
        if lib == "opencv-python-headless":
            import cv2
            print(f"cv2: {cv2.__version__}")
        elif lib == "pillow":
            from PIL import __version__ as pil_version
            print(f"PIL: {pil_version}")
    except pkg_resources.DistributionNotFound:
        print(f"{lib}: Not installed")
    except ImportError:
        print(f"{lib}: Installed but import failed")

print("\nDetailed module versions:")
print("-" * 50)

# Check tensorflow components
try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {tf.keras.__version__}")
except ImportError:
    print("TensorFlow: Import failed")

# Check numpy
try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except ImportError:
    print("NumPy: Import failed")

# Check streamlit
try:
    import streamlit as st
    print(f"Streamlit: {st.__version__}")
except ImportError:
    print("Streamlit: Import failed")
