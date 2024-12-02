import pandas as pd

from src.GUI import GUI
from src.Visualization import Visualization


def main():

    df = pd.read_csv("budgeting_dataset.csv")  # Create a dataframe of the budgeting_dataset.csv file
    print(df.head())  # Print first 5 rows
    print(df.info())  # Print additional info on data

    # Change the budgeting col so instead of a percentage they are the actual amount in relation to their income
    df['HousingBudgetAmount'] = df['Income'] * (df['HousingBudget'] / 100)
    df['TransportationBudgetAmount'] = df['Income'] * (df['TransportationBudget'] / 100)
    df['FoodBudgetAmount'] = df['Income'] * (df['FoodBudget'] / 100)
    df['UtilitiesBudgetAmount'] = df['Income'] * (df['UtilitiesBudget'] / 100)
    df['EntertainmentBudgetAmount'] = df['Income'] * (df['EntertainmentBudget'] / 100)

    print(df.head())

    # Display data visualizations
    vis = Visualization(df)
    # Pair plot
    vis.display_pair_plot()

    # Average expenses bar chart
    vis.display_avg_exp_chart()

    # Pie chart
    vis.display_pie_chart()

    # Line plot of Total Monthly Expenses
    vis.display_line_plot()

    # Box plot
    vis.display_box_plot()

    # Stacked bar chart
    vis.display_stacked_chart()

    # Histogram of income distribution
    vis.display_histogram()

    # Scatter plot of Income vs Total Expenses
    vis.display_scatter_plot("Income", "Total Expenses")

    # Scatter plot of Total Expenses vs Savings
    vis.display_scatter_plot("Total Expenses", "Savings")

    app = GUI().root
    app.mainloop()


main()
