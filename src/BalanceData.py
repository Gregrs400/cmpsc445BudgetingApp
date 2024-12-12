import pandas as pd
import numpy as np

# Load your dataset
data = pd.read_csv('budgeting_dataset.csv')

# Calculate expense ratios for training data
data['TotalExpenses'] = data[['HousingExpense', 'TransportationExpense', 'FoodExpense',
                              'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)
data['ExpenseIncomeRatio'] = data['TotalExpenses'] / data['Income']

# Filter rows by classification
frugal = data[data['ExpenseIncomeRatio'] <= 0.5]
spender = data[data['ExpenseIncomeRatio'] > 0.75]

# Define the desired number of rows
desired_frugal = 1056  # Adjust as needed
desired_spender = 550  # Adjust as needed

# Generate synthetic rows for frugal
synthetic_frugal = frugal.sample(n=desired_frugal, replace=True).copy()
synthetic_spender = spender.sample(n=desired_spender, replace=True).copy()

# Add variability to multiple columns
for col in ['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']:
    synthetic_frugal[col] += np.random.uniform(-0.025, 0.025, size=len(synthetic_frugal)) * synthetic_frugal[col].mean()
    synthetic_spender[col] += np.random.uniform(-0.025, 0.025, size=len(synthetic_spender))
synthetic_frugal['ExpenseIncomeRatio'] += np.random.uniform(-0.025, 0.025, size=len(synthetic_frugal))
synthetic_spender['ExpenseIncomeRatio'] += np.random.uniform(-0.025, 0.025, size=len(synthetic_spender))

# Combine original data with synthetic rows
balanced_data = pd.concat([data, synthetic_frugal, synthetic_spender], ignore_index=True)

# Drop duplicates
balanced_data = balanced_data.drop_duplicates()

# Save the new dataset
balanced_data.to_csv('balanced_dataset.csv', index=False)
