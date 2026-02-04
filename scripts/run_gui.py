#!/usr/bin/env python
"""
Launch the AggreQuant GUI application.

Usage:
    python scripts/run_gui.py

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gui import main

if __name__ == "__main__":
    main()
