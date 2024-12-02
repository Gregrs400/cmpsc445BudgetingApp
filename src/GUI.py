import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# GUI
class GUI:
    def __init__(self, df):
        # Store the training dataframe
        self.df = df

        # App setup
        self.root = tk.Tk()
        self.root.title("Budgeting App")
        self.root.geometry("600x900")
        self.root.configure(bg='#f0f0f0')  # Light grey background

        # Create a frame for better organization
        self.frame = tk.Frame(self.root, bg='#ffffff', padx=20, pady=20)
        self.frame.pack(padx=10, pady=10, fill='both', expand=True)

        self.create_header()
        self.create_input_fields()
        self.create_visualize_button()

    def create_header(self):
        # Header label
        header_label = tk.Label(self.frame, text="Budgeting Tool", font=('Helvetica', 16, 'bold'), bg='#ffffff')
        header_label.pack(pady=(0, 20))

    def create_input_fields(self):
        # Input fields
        tk.Label(self.frame, text="Monthly Income:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
        self.entry_income = tk.Entry(self.frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
        self.entry_income.pack(pady=(0, 10), fill='x')

        tk.Label(self.frame, text="Housing Expense:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
        self.entry_housing = tk.Entry(self.frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
        self.entry_housing.pack(pady=(0, 10), fill='x')

        tk.Label(self.frame, text="Food Expense:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
        self.entry_food = tk.Entry(self.frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
        self.entry_food.pack(pady=(0, 10), fill='x')

        tk.Label(self.frame, text="Transportation Expense:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
        self.entry_transportation = tk.Entry(self.frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
        self.entry_transportation.pack(pady=(0, 10), fill='x')

        tk.Label(self.frame, text="Utilities Expense:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
        self.entry_utilities = tk.Entry(self.frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
        self.entry_utilities.pack(pady=(0, 10), fill='x')

        tk.Label(self.frame, text="Entertainment Expense:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
        self.entry_entertainment = tk.Entry(self.frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
        self.entry_entertainment.pack(pady=(0, 30), fill='x')

    def create_visualize_button(self):
        # Visualize button
        visualize_button = tk.Button(self.frame, text="Visualize Budget", command=self.visualize_budget, font=('Helvetica', 12),
                                     bg='#4CAF50', fg='white',
                                     relief='raised', padx=10, pady=5)
        visualize_button.pack()

    def classify_spending_category(self, total_expenses):
        # Ensure the dataframe has TotalExpenses column
        self.df['TotalExpenses'] = self.df[['HousingExpense', 'TransportationExpense', 'FoodExpense',
                                            'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)

        scaler = StandardScaler()
        scaled_expenses = scaler.fit_transform(self.df[['TotalExpenses']])

        kmeans = KMeans(n_clusters=3, random_state=42)
        self.df['SpendingCategory'] = kmeans.fit_predict(scaled_expenses)

        category_mapping = {0: 'Frugal', 1: 'Average', 2: 'Spender'}
        self.df['SpendingCategory'] = self.df['SpendingCategory'].map(category_mapping)

        # Scale the user input for prediction
        user_input_scaled = scaler.transform([[total_expenses]])

        # Predict spending category for the user input
        category_prediction = kmeans.predict(user_input_scaled)
        return category_mapping[category_prediction[0]]

    def predict_expenses(self, user_input):
        expenses = {}

        # Prepare the features for prediction
        X = self.df[['TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']]
        
        # Predict each expense category
        expense_categories = ['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']
        
        for expense_category in expense_categories:
            y_target = self.df[expense_category]

            X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            user_input_scaled = scaler.transform(user_input)

            knn_model = KNeighborsRegressor(n_neighbors=3)
            knn_model.fit(X_train_scaled, y_train)

            predicted_expense = knn_model.predict(user_input_scaled)
            expenses[expense_category] = predicted_expense[0]

        return expenses

    def plot_budget_pie_chart(self, income, housing, food, transportation, utilities, entertainment):
        # Calculate total expenses and savings
        total_expenses = housing + food + transportation + utilities + entertainment
        savings = income - total_expenses

        # Create a pie chart
        labels = ['Savings', 'Housing', 'Food', 'Transportation', 'Utilities', 'Entertainment']
        sizes = [savings, housing, food, transportation, utilities, entertainment]
        colors = ['#ffcc99', '#ff9999', '#66b3ff', '#99ff99', '#800080', '#c2c2f0']  # Custom colors for each category

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Function to format the percentage display
        def func(pct, allvalues):
            absolute = int(pct / 100. * sum(allvalues))  # Calculate absolute value
            return f'{absolute} ({pct:.1f}%)'

        wedges, texts, autotexts = ax.pie(sizes, colors=colors, autopct=lambda pct: func(pct, sizes), startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.

        # Create a legend to the left of the pie chart
        ax.legend(wedges, labels, title="Budget Categories", loc="center left", bbox_to_anchor=(-0.45, 0.5),
                  fontsize='small')

        # Adjust the position of the pie chart
        ax.set_position([0.3, 0.1, 0.6, 0.8])  # Adjust left margin to move pie chart to the right

        # Create a canvas to display the chart
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def visualize_budget(self):
        try:
            # Retrieve inputs
            income = float(self.entry_income.get())
            housing = float(self.entry_housing.get())
            food = float(self.entry_food.get())
            transportation = float(self.entry_transportation.get())
            utilities = float(self.entry_utilities.get())
            entertainment = float(self.entry_entertainment.get())

            total_expenses = housing + food + transportation + utilities + entertainment
            savings = income - total_expenses

            # Create a pie chart of the budget
            self.plot_budget_pie_chart(income, housing, food, transportation, utilities, entertainment)

            messagebox.showinfo("Savings", f"Total Savings: ${savings:.2f}")

            # Convert user input to DataFrame for scaling
            user_input = pd.DataFrame([[transportation, food, utilities, entertainment]],
                                      columns=['TransportationExpense', 'FoodExpense', 'UtilitiesExpense',
                                               'EntertainmentExpense'])

            # Predict future expenses
            predicted_expenses = self.predict_expenses(user_input)

            predicted_messages = "\n".join([f"Predicted Housing Expense: ${predicted_expenses['HousingExpense']:.2f}"] +
                                           [f"Predicted {cat}: ${amount:.2f}"
                                            for cat, amount in predicted_expenses.items() if cat != 'HousingExpense'])
            messagebox.showinfo("Predicted Expenses", predicted_messages)

            # Classify spending category
            spending_category = self.classify_spending_category(total_expenses)
            messagebox.showinfo("Spending Category", f"You belong to Spending Category: {spending_category}")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers.")
