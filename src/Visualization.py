import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Visualization:
    def __init__(self, df):
        self.df = df
        self.average_expenses = df[['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']].mean()
        self.total_monthly_expenses = df[['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)


    def create_pie_chart(self):
        pass

    def display_heatmap(self):
        # Calculate mean values for plotting
        budgeted_expenses = self.df[
            ['HousingBudgetAmount', 'TransportationBudgetAmount', 'FoodBudgetAmount', 'UtilitiesBudgetAmount',
             'EntertainmentBudgetAmount']].mean()
        actual_expenses = self.df[['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense',
                              'EntertainmentExpense']].mean()

        # Note: I don't think comparison_df is used at all, maybe it will be used in the future?
        # Combine into a single DataFrame for plotting
        comparison_df = pd.DataFrame({
            'Budgeted': budgeted_expenses,
            'Actual': actual_expenses
        })

        # Heatmap of Correlation Between Expenses and Budget
        # Prepare correlation data
        correlation_data = self.df[
            ['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense',
             'HousingBudgetAmount', 'TransportationBudgetAmount', 'FoodBudgetAmount',
             'UtilitiesBudgetAmount', 'EntertainmentBudgetAmount']].corr()

        # Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Heatmap of Correlation Between Expenses and Budgets")
        plt.show()

    def create_bar_chart(self):
        pass

    def create_line_graph(self):
        pass

    def display_pair_plot(self):
        sns.pairplot(self.df[['Income', 'HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']])
        plt.title("Pair Plot of Expenses")
        plt.show()

    def display_avg_exp_chart(self):
        self.average_expenses.plot(kind='bar', color='skyblue')
        plt.title("Average Expenses by Category")
        plt.ylabel("Average Amount")
        plt.xlabel("Expense Category")
        plt.xticks(rotation=45)
        plt.show()

    def display_pie_chart(self):
        explode = (0, 0, 0, 0, 0)  # explode 1st slice
        self.average_expenses.plot(kind='pie', autopct='%1.1f%%', startangle=90, explode=explode)
        plt.title("Total Expenses by Category")
        plt.ylabel("")  # Hide the y-label
        plt.show()

    def display_line_plot(self):
        plt.plot(self.df['UserID'], self.total_monthly_expenses, marker='o')
        plt.title("Total Monthly Expenses")
        plt.xlabel("User ID")
        plt.ylabel("Total Expenses")
        plt.grid()
        plt.show()

    def display_box_plot(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df[
            ['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']])
        plt.title("Box Plots of Expense Categories")
        plt.ylabel("Amount")
        plt.xlabel("Expense Category")
        plt.xticks(rotation=45)
        plt.show()

    def display_stacked_chart(self):
        self.df[['UserID', 'HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense',
            'EntertainmentExpense']].set_index('UserID').plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title("Stacked Bar Chart of Expenses by User")
        plt.ylabel("Amount")
        plt.xlabel("User ID")
        plt.legend(title="Expense Category")
        plt.show()

    def display_histogram(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['Income'], bins=10, kde=True)
        plt.title("Income Distribution")
        plt.xlabel("Income")
        plt.ylabel("Frequency")
        plt.show()

    def display_scatter_plot(self, x_val, y_val):
        self.df['TotalExpenses'] = self.df[
            ['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']].sum(
            axis=1)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_val, y=y_val, data=self.df)
        plt.title(x_val + " vs " + y_val)
        plt.xlabel(x_val)
        plt.ylabel(y_val)
        plt.grid()
        plt.show()