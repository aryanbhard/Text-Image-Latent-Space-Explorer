"""
Pytest configuration and environment shims for reproducible execution.
"""
import sys

# Ensure regex C-extension issues on restricted OS environments fall back to built-in re
try:
    import regex
except ImportError:
    import re
    sys.modules["regex"] = re
