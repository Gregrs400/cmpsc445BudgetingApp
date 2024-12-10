import pandas as pd

from GUI import GUI
from Visualization import Visualization


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
    # # Pair plot
    # vis.display_pair_plot()
    #
    # # Average expenses bar chart
    # vis.display_avg_exp_chart()
    #
    # # Pie chart
    # vis.display_pie_chart()
    #
    # # Line plot of Total Monthly Expenses
    # vis.display_line_plot()
    #
    # # Box plot
    # vis.display_box_plot()
    #
    # # Stacked bar chart
    # vis.display_stacked_chart()
    #
    # # Histogram of income distribution
    # vis.display_histogram("Income", 10, "Income Distribution", "Income", "Frequency")
    #
    # # Scatter plot of Income vs Total Expenses
    # vis.display_scatter_plot("Income", "TotalExpenses")
    #
    # # Scatter plot of Total Expenses vs Savings
    # vis.display_scatter_plot("TotalExpenses", "Savings")
    #
    # # Heatmap of Correlation Between Expenses and Budget
    # vis.display_heatmap()
    #
    # # Bar Chart of Average Expenses by Category with Budget Comparison
    # vis.display_bar_chart()
    #
    # # Line Plot of Total Monthly Expenses vs. Total Budget
    # vis.display_adv_line_plt()
    #
    # # Scatter Plot of Actual vs. Budgeted
    # vis.display_adv_scatter_plt()
    #
    # # Histogram of housing expenses
    # vis.display_histogram("HousingExpense", 15, "Housing Expense", "Amount", "Frequency")

    app = GUI(df).root
    app.mainloop()


main()
