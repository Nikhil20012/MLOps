import logging
import json
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage()
        })

logger = logging.getLogger("ridge_training")
logger.setLevel(logging.INFO)
for handler in [logging.FileHandler("training.log"), logging.StreamHandler()]:
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

logger.info("Loading California Housing dataset")
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alpha = 1.0
logger.info(f"Starting Ridge Regression training with alpha={alpha}")
model = Ridge(alpha=alpha)
model.fit(X_train_scaled, y_train)
logger.info("Model training complete")

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
logger.info(f"MSE: {round(mse, 4)}")
logger.info(f"RMSE: {round(float(np.sqrt(mse)), 4)}")
logger.info(f"R2 Score: {round(r2, 4)}")
logger.info(f"Model coefficients: {model.coef_.tolist()}")
logger.info(f"Model intercept: {round(float(model.intercept_), 4)}")
logger.info(f"Alpha (regularization): {alpha}")
logger.info(f"Features: {list(housing.feature_names)}")