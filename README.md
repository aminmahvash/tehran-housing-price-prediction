# Tehran Housing Price Prediction

A Machine Learning project using **Regression** algorithms to estimate real estate prices in Tehran.

## Model Performance Results
After testing different regression models, here are the accuracy results based on the **R2-score**:

| Model | R2 Score | Status |
| :--- | :--- | :--- |
| **Simple Linear Regression** | **0.53** | Baseline |
| **Polynomial Regression (Degree 2)** | **0.72** | Best Balance |
| **Polynomial Regression (Degree 3)** | **0.73** | Marginal Improvement |

## Key Findings
* The relationship between **Area** and **Price** in Tehran is non-linear.
* Adding **Polynomial Features** (Interaction between variables like Area and Address) improved the model accuracy by nearly **20%**.
* Higher degrees (Degree 3) did not significantly improve performance compared to Degree 2, suggesting a risk of overfitting.

## Features Used
- **Area**: Square meters of the property.
- **Room**: Number of bedrooms.
- **Parking/Warehouse/Elevator**: Boolean features (0 or 1).
- **Address**: Encoded location data.

## How to Use
1. Clone the repository.
2. Ensure you have `scikit-learn`, `pandas`, `numpy`, and `seaborn` installed.
3. Open `Tehran_Housing_Prediction.ipynb` to see the full analysis and visualizations.
