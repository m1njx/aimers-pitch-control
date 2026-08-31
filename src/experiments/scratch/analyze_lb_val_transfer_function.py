import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge

# Data points: (Version, Description, Val_2024, LB_Score)
# Let's map known versions
data = [
    {"ver": "v16", "desc": "Initial Ensemble (Scale 1.00)", "val_2024": 1520.0, "lb": 1003.0255},
    {"ver": "v23", "desc": "Scale 1.10", "val_2024": 1580.0, "lb": 1016.1281},
    {"ver": "v29", "desc": "Scale 1.15", "val_2024": 1550.0, "lb": 1008.7229},
    {"ver": "v36", "desc": "GBDT 45 + MLP 35 + MSE 20", "val_2024": 1610.0, "lb": 1010.5100},
    {"ver": "v40", "desc": "133 Feats + 3D Tunneling", "val_2024": 1690.0, "lb": 1029.8800},
    {"ver": "v42", "desc": "SOTA BCE MLP 40 + GBDT 40 + MSE 20", "val_2024": 1716.91, "lb": 1032.1376},
    {"ver": "v45", "desc": "Overfitted Quad Neural", "val_2024": 1450.0, "lb": 993.0200},
    {"ver": "v48", "desc": "SWA MSE Loss Retrain", "val_2024": 1650.0, "lb": 1020.4300},
    {"ver": "v50", "desc": "Safe Balanced (MLP 50, GBDT 25, MSE 25, Shift -0.0035)", "val_2024": 1816.17, "lb": 1032.8239},
]

df = pd.DataFrame(data)
print("=== Historical Submission Matrix ===")
print(df.to_string(index=False))

# Why did v50 only gain +0.686 despite higher val?
# Notice: In v50, the base models were NOT retrained. It was a pure post-hoc weight & shift adjustment on the same predictions.
# When changing weights among collinear models (r > 0.96), local val can overfit to the validation distribution's specific density mode!
# True structural improvements (adding new features like v40 +19.37 pts, or calibrating scale v23 +13.10 pts) show real LB jumps.

# Let's compute the transfer function for:
# 1. Structural changes (New features / Better loss functions / Model architectures)
# 2. Re-weighting / Calibration tweaks

print("\n--- Correlation Analysis ---")
corr = np.corrcoef(df['val_2024'], df['lb'])[0, 1]
print(f"Pearson Correlation (Val 2024 vs Public LB): r = {corr:.4f}")

# Fit regression model
X = df[['val_2024']].values
y = df['lb'].values

reg = HuberRegressor().fit(X, y)
slope = reg.coef_[0]
intercept = reg.intercept_

print(f"\nRobust Linear Transfer Function:")
print(f"  Predicted Public LB = {slope:.4f} * (2024 Val Score) + ({intercept:.2f})")

# Test on key anchor points
print("\nModel In-Sample Fit & Residuals:")
df['predicted_lb'] = reg.predict(X)
df['residual'] = df['lb'] - df['predicted_lb']
print(df[['ver', 'val_2024', 'lb', 'predicted_lb', 'residual']].to_string(index=False))
