# PRODIGY_ML_01
🏠 House Price Prediction using Linear Regression
This project builds a machine learning model using linear regression to predict house prices based on key features such as living area, basement size, garage capacity, and overall quality.

📁 Dataset
Source: train.csv (must be present in the same directory as main.py)

Features Used:

GrLivArea – Above grade (ground) living area in square feet

OverallQual – Overall material and finish quality

TotalBsmtSF – Total square feet of basement area

GarageCars – Size of garage in car capacity

SalePrice – Target variable (price of the house)

📌 Requirements
Install the required libraries using pip:

bash
Copy
Edit
pip install pandas scikit-learn matplotlib
🚀 How to Run
Ensure train.csv is in the same folder as main.py.

Run the script:

bash
Copy
Edit
python main.py
This will:

Load and preprocess the data

Train a linear regression model

Print model evaluation metrics (MSE and R² Score)

Predict the price of a sample new house

Plot actual vs predicted prices

📈 Output
Metrics:

Mean Squared Error

R² Score (coefficient of determination)

Predicted Price for a sample house with:

2000 sq ft living area

Quality rating of 7

800 sq ft basement

2-car garage

Visualization:

Scatter plot comparing actual vs predicted sale prices

📊 Sample Prediction
python
Copy
Edit
new_data = pd.DataFrame([[2000, 7, 800, 2]], columns=X.columns)
predicted_price = model.predict(new_data)
📌 Notes
The model is a basic implementation of linear regression and can be improved with more features, feature engineering, or using advanced models.

The script uses train_test_split to evaluate generalization performance.

🧠 Author
Sanju Reddy — Machine Learning Enthusiast
(Created as part of AI/ML learning journey)
