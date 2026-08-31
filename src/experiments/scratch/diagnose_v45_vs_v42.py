import numpy as np
import pandas as pd

# Let's inspect the exact difference between v42 and v45 predictions
# and examine the calibration drift.

# v42 score: 1032.137582 (Brier skill)
# Base Brier for r=0.4861: 0.4861 * (1 - 0.4861) = 0.24980679
# Score = 100000 * (1 - Brier / 0.24980679)
# For v42: Brier = 0.24980679 * (1 - 1032.137582 / 100000) = 0.24722838
# For v45: Brier = 0.24980679 * (1 - 993.023757 / 100000) = 0.24732610
# The Brier difference is only 0.0000977 (less than 0.0001!), but in Brier Skill Score that's 39.11 points!

print(f"Brier diff: {(0.24732610 - 0.24722838):.7f}")
