import sys

print("EXECUTABLE:", sys.executable)
print("PATH:")
for p in sys.path:
    print("  ", p)

import tensorflow as tf

print("TensorFlow:", tf.__version__)