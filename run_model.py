#!/usr/bin/env python3
"""
Script to run the COPD detection model
"""

print("Starting COPD detection model...")

try:
    import copd_detection_model
    print("Successfully imported copd_detection_model")
    
    # Run the main function in test mode
    print("Running main function in test mode...")
    copd_detection_model.main(test_mode=True)
    print("Main function completed")
    
except Exception as e:
    print(f"Error running model: {e}")
    import traceback
    traceback.print_exc()

print("Script completed")