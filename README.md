# House Price Prediction

## Intern ID
CITS9341

## Project Overview
A Machine Learning project that predicts house prices based on features like area, number of bedrooms/bathrooms, age of the property, and location score. Built using Python, Pandas, and Scikit-learn (Linear Regression).

## Features
- Loads and explores housing dataset (`data/house_data.csv`)
- Trains a Linear Regression model on 80% of the data
- Evaluates model performance using MAE, RMSE, and R² score
- Predicts price for a new sample house
- Visualizes Actual vs Predicted prices as a scatter plot

## Dataset
`data/house_data.csv` — 30 sample house records with:
- `area_sqft` — area of the house in square feet
- `bedrooms` — number of bedrooms
- `bathrooms` — number of bathrooms
- `age_years` — age of the property in years
- `location_score` — location rating (1–10)
- `price_lakhs` — price in lakhs (target variable)

## Files
- `house_price_prediction.py` — main script (load data, train model, evaluate, predict, visualize)
- `data/house_data.csv` — dataset
- `requirements.txt` — Python dependencies
- `actual_vs_predicted.png` — output chart generated after running the script

## How to Run
1. Make sure Python is installed
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the script:
   ```
   python house_price_prediction.py
   ```
4. Output will show model evaluation metrics in the terminal, and a chart (`actual_vs_predicted.png`) will be saved in the project folder.

## Tech Used
- Python
- Pandas, NumPy
- Scikit-learn (Linear Regression)
- Matplotlib

## Author
Submitted as part of CodTech IT Solutions Internship Program.
