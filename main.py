import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("train.csv")

# Select useful features
df = df[['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'SalePrice']].dropna()

# Split into input (X) and target (y)
X = df[['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageCars']]
y = df['SalePrice']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Predict for a new house
new_data = pd.DataFrame([[2000, 7, 800, 2]], columns=X.columns)
predicted_price = model.predict(new_data)
print("Predicted Price:", predicted_price[0])

# Optional: Visualize predictions
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

