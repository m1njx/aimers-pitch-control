import os
import torch
import numpy as np

# Let's inspect the difference in weights / predictions between v42 and v48
# v42 MLP vs v48 MLP

# In v42:
# 5 SimpleMLP models trained with AdamW, lr=?, epochs=?, NO SWA!
# SimpleMLP dropout = 0.12 in v42
# In v48: SWA averaged models over epochs 2-5 with AdamW lr=3e-3.

# Let's check how v42 models were trained vs v48!
