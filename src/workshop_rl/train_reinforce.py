"""Backward-compatibility shim — delegates to car_racing.train.

The implementation has moved to src/car_racing/; run directly with:
    cd src && python -m car_racing.train [options]

This file is kept so that existing scripts that call
    python -m workshop_rl.train_reinforce
continue to work unchanged.
"""

import sys
import os

# Allow `python -m workshop_rl.train_reinforce` to find `car_racing` and `rl_core`
# even when the caller hasn't set PYTHONPATH.
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from car_racing.train import main  # noqa: E402

if __name__ == "__main__":
    main()
