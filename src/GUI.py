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


# GUI
class GUI:
    def __init__(self):
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

    def classify_spending_category(self, user_input, df):
        df['TotalExpenses'] = df[['HousingExpense', 'TransportationExpense', 'FoodExpense',
                                  'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)

        scaler = StandardScaler()
        scaled_expenses = scaler.fit_transform(df[['TotalExpenses']])

        kmeans = KMeans(n_clusters=3, random_state=42)
        df['SpendingCategory'] = kmeans.fit_predict(scaled_expenses)

        category_mapping = {0: 'Frugal', 1: 'Average', 2: 'Spender'}
        df['SpendingCategory'] = df['SpendingCategory'].map(category_mapping)

        # Scale the user input for prediction
        user_input_scaled = scaler.transform(user_input)

        # Predict spending category for the user input
        category_prediction = kmeans.predict(user_input_scaled)
        return category_mapping[category_prediction[0]]


    def predict_expenses(self, user_input, df):
        expenses = {}

        # Predict housing expense
        y_housing = df['HousingExpense']
        X = df[['TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']]

        # Using actual expenses for KNN
        X_train, X_test, y_train_housing, y_test_housing = train_test_split(X, y_housing, test_size=0.2,
                                                                            random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        knn_model = KNeighborsRegressor(n_neighbors=3)
        knn_model.fit(X_train_scaled, y_train_housing)

        user_input_scaled = scaler.transform(user_input)
        predicted_housing = knn_model.predict(user_input_scaled)[0]
        expenses['HousingExpense'] = predicted_housing

        # Predict other expenses
        for expense_category in ['TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']:
            y_target = df[expense_category]

            X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

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
            spending_category = self.classify_spending_category(
                [[total_expenses]])  # Pass total expenses as input for classification
            messagebox.showinfo("Spending Category", f"You belong to Spending Category: {spending_category}")



        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers.")

# # Function to classify spending category
# def classify_spending_category(user_input, df):
#     df['TotalExpenses'] = df[['HousingExpense', 'TransportationExpense', 'FoodExpense',
#                               'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)
#
#     scaler = StandardScaler()
#     scaled_expenses = scaler.fit_transform(df[['TotalExpenses']])
#
#     kmeans = KMeans(n_clusters=3, random_state=42)
#     df['SpendingCategory'] = kmeans.fit_predict(scaled_expenses)
#
#     category_mapping = {0: 'Frugal', 1: 'Average', 2: 'Spender'}
#     df['SpendingCategory'] = df['SpendingCategory'].map(category_mapping)
#
#     # Scale the user input for prediction
#     user_input_scaled = scaler.transform(user_input)
#
#     # Predict spending category for the user input
#     category_prediction = kmeans.predict(user_input_scaled)
#     return category_mapping[category_prediction[0]]
#
#
# # Function to predict future expenses
# def predict_expenses(user_input, df):
#     expenses = {}
#
#     # Predict housing expense
#     y_housing = df['HousingExpense']
#     X = df[['TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']]
#
#     # Using actual expenses for KNN
#     X_train, X_test, y_train_housing, y_test_housing = train_test_split(X, y_housing, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#
#     knn_model = KNeighborsRegressor(n_neighbors=3)
#     knn_model.fit(X_train_scaled, y_train_housing)
#
#     user_input_scaled = scaler.transform(user_input)
#     predicted_housing = knn_model.predict(user_input_scaled)[0]
#     expenses['HousingExpense'] = predicted_housing
#
#     # Predict other expenses
#     for expense_category in ['TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']:
#         y_target = df[expense_category]
#
#         X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#
#         knn_model = KNeighborsRegressor(n_neighbors=3)
#         knn_model.fit(X_train_scaled, y_train)
#
#         predicted_expense = knn_model.predict(user_input_scaled)
#         expenses[expense_category] = predicted_expense[0]
#
#     return expenses
#
#
# def plot_budget_pie_chart(income, housing, food, transportation, utilities, entertainment):
#     # Calculate total expenses and savings
#     total_expenses = housing + food + transportation + utilities + entertainment
#     savings = income - total_expenses
#
#     # Create a pie chart
#     labels = ['Savings', 'Housing', 'Food', 'Transportation', 'Utilities', 'Entertainment']
#     sizes = [savings, housing, food, transportation, utilities, entertainment]
#     colors = ['#ffcc99', '#ff9999', '#66b3ff', '#99ff99', '#800080', '#c2c2f0']  # Custom colors for each category
#
#     fig = Figure(figsize=(5, 4), dpi=100)
#     ax = fig.add_subplot(111)
#
#     # Function to format the percentage display
#     def func(pct, allvalues):
#         absolute = int(pct / 100. * sum(allvalues))  # Calculate absolute value
#         return f'{absolute} ({pct:.1f}%)'
#
#     wedges, texts, autotexts = ax.pie(sizes, colors=colors, autopct=lambda pct: func(pct, sizes), startangle=90)
#     ax.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
#
#     # Create a legend to the left of the pie chart
#     ax.legend(wedges, labels, title="Budget Categories", loc="center left", bbox_to_anchor=(-0.45, 0.5),
#               fontsize='small')
#
#     # Adjust the position of the pie chart
#     ax.set_position([0.3, 0.1, 0.6, 0.8])  # Adjust left margin to move pie chart to the right
#
#     # Create a canvas to display the chart
#     canvas = FigureCanvasTkAgg(fig, master=frame)
#     canvas.draw()
#     canvas.get_tk_widget().pack()
#
#
# # Function to visualize budget
# def visualize_budget():
#     try:
#         # Retrieve inputs
#         income = float(entry_income.get())
#         housing = float(entry_housing.get())
#         food = float(entry_food.get())
#         transportation = float(entry_transportation.get())
#         utilities = float(entry_utilities.get())
#         entertainment = float(entry_entertainment.get())
#
#         total_expenses = housing + food + transportation + utilities + entertainment
#         savings = income - total_expenses
#
#         # Create a pie chart of the budget
#         plot_budget_pie_chart(income, housing, food, transportation, utilities, entertainment)
#
#         messagebox.showinfo("Savings", f"Total Savings: ${savings:.2f}")
#
#         # Convert user input to DataFrame for scaling
#         user_input = pd.DataFrame([[transportation, food, utilities, entertainment]],
#                                   columns=['TransportationExpense', 'FoodExpense', 'UtilitiesExpense',
#                                            'EntertainmentExpense'])
#
#         # Predict future expenses
#         predicted_expenses = predict_expenses(user_input)
#
#         predicted_messages = "\n".join([f"Predicted Housing Expense: ${predicted_expenses['HousingExpense']:.2f}"] +
#                                        [f"Predicted {cat}: ${amount:.2f}"
#                                         for cat, amount in predicted_expenses.items() if cat != 'HousingExpense'])
#         messagebox.showinfo("Predicted Expenses", predicted_messages)
#
#         # Classify spending category
#         spending_category = classify_spending_category(
#             [[total_expenses]])  # Pass total expenses as input for classification
#         messagebox.showinfo("Spending Category", f"You belong to Spending Category: {spending_category}")
#
#
#
#     except ValueError:
#         messagebox.showerror("Input Error", "Please enter valid numbers.")
#
# # Set up the main window
# root = tk.Tk()
# root.title("Budgeting App")
#
# # Set the window size (width x height)
# root.geometry("600x900")
# root.configure(bg='#f0f0f0')  # Light grey background
#
# # Create a frame for better organization
# frame = tk.Frame(root, bg='#ffffff', padx=20, pady=20)
# frame.pack(padx=10, pady=10, fill='both', expand=True)
#
# # Header label
# header_label = tk.Label(frame, text="Budgeting Tool", font=('Helvetica', 16, 'bold'), bg='#ffffff')
# header_label.pack(pady=(0, 20))
#
# # Input fields
# tk.Label(frame, text="Monthly Income:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
# entry_income = tk.Entry(frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
# entry_income.pack(pady=(0, 10), fill='x')
#
# tk.Label(frame, text="Housing Expense:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
# entry_housing = tk.Entry(frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
# entry_housing.pack(pady=(0, 10), fill='x')
#
# tk.Label(frame, text="Food Expense:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
# entry_food = tk.Entry(frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
# entry_food.pack(pady=(0, 10), fill='x')
#
# tk.Label(frame, text="Transportation Expense:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
# entry_transportation = tk.Entry(frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
# entry_transportation.pack(pady=(0, 10), fill='x')
#
# tk.Label(frame, text="Utilities Expense:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
# entry_utilities = tk.Entry(frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
# entry_utilities.pack(pady=(0, 10), fill='x')
#
# tk.Label(frame, text="Entertainment Expense:", font=('Helvetica', 12), bg='#ffffff').pack(anchor='w')
# entry_entertainment = tk.Entry(frame, font=('Helvetica', 12), bg='#e0e0e0', bd=2)
# entry_entertainment.pack(pady=(0, 30), fill='x')
#
# # Visualize button
# visualize_button = tk.Button(frame, text="Visualize Budget", command=visualize_budget, font=('Helvetica', 12), bg='#4CAF50', fg='white',
#                              relief='raised', padx=10, pady=5)
# visualize_button.pack()
#
#
#
# # Start the GUI event loop
# root.mainloop()