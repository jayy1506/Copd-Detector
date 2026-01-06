print("Test script running")

# Simple test to check if libraries can be imported
try:
    import os
    import numpy as np
    import cv2
    print("All imports successful")
except Exception as e:
    print(f"Import error: {e}")

print("Test script completed")