from typing import Final

# Backtest protocol (ADR to write)
HORIZON_DAYS: Final = 14
INITIAL_TRAIN_DAYS: Final = 180

# Series scope policy (step 2.5b)
MAX_FILLABLE_GAP_DAYS: Final = 7
MAX_DAYS_MISSING_AT_END: Final = 14
LONG_ZERO_RUN_DAYS: Final = 7
MIN_USEFUL_DAYS: Final = INITIAL_TRAIN_DAYS + HORIZON_DAYS  # 194
