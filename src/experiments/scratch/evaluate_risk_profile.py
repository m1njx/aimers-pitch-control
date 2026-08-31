import numpy as np

# Let's calculate the correlation between GBDT Binary, LGBM MSE, and SimpleMLP
# and see how much variance reduction is sacrificed when GBDT Binary is reduced from 25% to 14%.

# In v42: GBDT 40% / MLP 40% / MSE 20%
# In v49 (with Scale 1.10): GBDT 25% / MLP 50% / MSE 25%
# In v50: GBDT 14% / MLP 51% / MSE 35%

print("Calculating exact risk/reward profile...")
