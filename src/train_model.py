import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from config import X_TRAIN, Y_TRAIN, MODEL_PKL, SCALER_X_PKL, SCALER_Y_PKL

# Load data
X = np.load(X_TRAIN)
y = np.load(Y_TRAIN)

print(f"Training samples: {len(X)}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Train model
print("\nTraining neural network...")

model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)

model.fit(X_train_scaled, y_train_scaled)

# Evaluate
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n--- Results ---")
print(f"R2 Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# Per-output R2 scores
for i, param in enumerate(['Kp', 'Ki', 'Kd']):
    r2_param = r2_score(y_test[:, i], y_pred[:, i])
    print(f"R2 for {param}: {r2_param:.4f}")

# Save model and scalers
joblib.dump(model, MODEL_PKL)
joblib.dump(scaler_X, SCALER_X_PKL)
joblib.dump(scaler_y, SCALER_Y_PKL)

print("\nModel saved!")
