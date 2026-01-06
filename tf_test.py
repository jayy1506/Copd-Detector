#!/usr/bin/env python3
"""
Minimal TensorFlow test
"""

print("Testing TensorFlow installation...")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow imported successfully!")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Simple test
    print("Running simple TensorFlow operation...")
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])
    c = tf.add(a, b)
    print(f"✓ Simple operation successful: {a.numpy()} + {b.numpy()} = {c.numpy()}")
    
except Exception as e:
    print(f"✗ Error with TensorFlow: {e}")

print("TensorFlow test completed.")