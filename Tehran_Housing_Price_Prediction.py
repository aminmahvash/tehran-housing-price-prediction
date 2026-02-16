import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Load Dataset ---
df = pd.read_csv("data\\housePrice.csv")


# --- Data Cleaning & Preprocessing ---
def cleanData(csv):
    # Remove rows with missing Address
    csv.dropna(subset=['Address'], inplace=True)

    # Convert Area to numeric, coerce errors to NaN and drop them
    csv['Area'] = pd.to_numeric(csv['Area'], errors='coerce')
    csv.dropna(subset=['Area'], inplace=True)

    # Filter out properties with Area >= 1000 sqm
    csv = csv[csv['Area'] < 1000]

    # Convert boolean features to integers
    bool_to_int = ['Parking', 'Warehouse', 'Elevator']
    for col in bool_to_int:
        csv[col] = csv[col].astype(int)

    # Normalize Address using Label Encoding
    csv['Address'] = LabelEncoder().fit_transform(csv['Address'])

    return csv


df = cleanData(df)

# --- Exploratory Data Analysis (EDA) ---
# Visualizing features impact on Price
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(x='Parking', y='Price', data=df, ax=axes[0])
axes[0].set_title('Parking vs Price')

sns.boxplot(x='Warehouse', y='Price', data=df, ax=axes[1])
axes[1].set_title('Warehouse vs Price')

sns.boxplot(x='Elevator', y='Price', data=df, ax=axes[2])
axes[2].set_title('Elevator vs Price')
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
plt.title('Feature Correlation Matrix')
plt.show()

# --- Data Splitting ---
# Define Independent (X) and Dependent (y) variables
X = df[['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address']]
y = df['Price']

# Split data: 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: Simple Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print('--- Linear Regression Results ---')
print('Coefficients:', lr_model.coef_)
print('Intercept:', lr_model.intercept_)
print(f"R2-score: {r2_score(y_test, y_pred_lr):.2f}\n")

# --- Model 2: Polynomial Regression (Degree 2) ---
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)

clf2 = LinearRegression()
clf2.fit(X_train_poly2, y_train)
y_pred_poly2 = clf2.predict(X_test_poly2)

print('--- Polynomial Regression (Degree 2) Results ---')
print('Coefficients:', clf2.coef_)
print('Intercept:', clf2.intercept_)
print(f"R2-score: {r2_score(y_test, y_pred_poly2):.2f}\n")

# --- Model 3: Polynomial Regression (Degree 3) ---
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)

clf3 = LinearRegression()
clf3.fit(X_train_poly3, y_train)
y_pred_poly3 = clf3.predict(X_test_poly3)

print('--- Polynomial Regression (Degree 3) Results ---')
print(f"R2-score: {r2_score(y_test, y_pred_poly3):.2f}")