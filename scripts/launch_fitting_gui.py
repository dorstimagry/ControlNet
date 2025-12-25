#!/usr/bin/env python3
"""Launch the vehicle parameter fitting GUI."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fitting.gui_fitting import main

if __name__ == "__main__":
    main()

