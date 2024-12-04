import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 9999)


class GUI:
    def __init__(self, df):
        # Store the training dataframe
        self.df = df

        # App setup
        self.root = tk.Tk()
        self.root.title("Budgeting App")
        self.root.geometry("800x1000")
        self.root.configure(bg='#f0f4f9')  # Soft blue-grey background

        # Create a main container frame
        self.main_frame = tk.Frame(self.root, bg='#ffffff', padx=20, pady=20)
        self.main_frame.pack(padx=20, pady=20, fill='both', expand=True, side=tk.TOP)

        # Create sections
        self.create_header()
        self.create_input_section()
        self.create_result_section()
        self.create_visualize_button()

    def create_header(self):
        # Header with improved styling
        header_frame = tk.Frame(self.main_frame, bg='#ffffff')
        header_frame.pack(fill='x', pady=(0, 20))

        header_label = tk.Label(header_frame, text="Smart Budget Analyzer", 
                                font=('Helvetica', 18, 'bold'), 
                                fg='#2c3e50', 
                                bg='#ffffff')
        header_label.pack()

        subtitle = tk.Label(header_frame, 
                            text="Understand Your Spending Habits", 
                            font=('Helvetica', 10), 
                            fg='#7f8c8d', 
                            bg='#ffffff')
        subtitle.pack()

    def create_input_section(self):
        # Input frame with improved layout
        input_frame = tk.Frame(self.main_frame, bg='#ffffff')
        input_frame.pack(fill='x', pady=(0, 20))

        # Configure grid
        input_frame.grid_columnconfigure(1, weight=1)

        # List of expense categories
        categories = [
            "Monthly Income", 
            "Housing Expense", 
            "Food Expense", 
            "Transportation Expense", 
            "Utilities Expense", 
            "Entertainment Expense"
        ]

        # Store entry references
        self.entries = {}

        # Create styled input fields
        for i, category in enumerate(categories):
            label = tk.Label(input_frame, text=f"{category}:", 
                             font=('Helvetica', 10), 
                             bg='#ffffff', 
                             fg='#2c3e50')
            label.grid(row=i, column=0, sticky='w', padx=(0, 10), pady=5)

            entry = tk.Entry(input_frame, 
                             font=('Helvetica', 10), 
                             bg='#ecf0f1', 
                             relief='flat', 
                             width=30)
            entry.grid(row=i, column=1, sticky='ew', pady=5)
            
            # Store entry reference
            self.entries[category.replace(" ", "")] = entry

    def create_result_section(self):
        # Result frame for displaying analysis
        self.result_frame = tk.Frame(self.main_frame, bg='#ffffff')
        self.result_frame.pack(fill='both', expand=True, pady=(20, 0))

        # Labels for different results
        self.savings_label = tk.Label(self.result_frame, 
                                      text="Total Savings: $0", 
                                      font=('Helvetica', 12, 'bold'), 
                                      bg='#ffffff', 
                                      fg='#27ae60')
        self.savings_label.pack(anchor='w', pady=5)

        self.spending_category_label = tk.Label(self.result_frame, 
                                                text="Spending Category: N/A", 
                                                font=('Helvetica', 12), 
                                                bg='#ffffff', 
                                                fg='#2980b9')
        self.spending_category_label.pack(anchor='w', pady=5)

        # Predictions label
        self.predictions_label = tk.Label(self.result_frame, 
                                          text="Predicted Expenses:\n", 
                                          font=('Helvetica', 12), 
                                          bg='#ffffff', 
                                          fg='#8e44ad', 
                                          justify=tk.LEFT)
        self.predictions_label.pack(anchor='w', pady=5)

        # Placeholder for pie chart
        self.chart_frame = tk.Frame(self.result_frame, bg='#ffffff')
        self.chart_frame.pack(fill='both', expand=True)

    def create_visualize_button(self):
        # Styled button
        visualize_button = tk.Button(
            self.main_frame, 
            text="Analyze Budget", 
            command=self.visualize_budget, 
            font=('Helvetica', 12, 'bold'),
            bg='#3498db', 
            fg='white', 
            relief='flat',
            padx=20, 
            pady=10
        )
        visualize_button.pack(pady=20)

    def plot_budget_pie_chart(self, income, housing, food, transportation, utilities, entertainment):
        # Clear previous chart if exists
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Calculate total expenses and savings
        total_expenses = housing + food + transportation + utilities + entertainment
        savings = income - total_expenses

        # Create a pie chart

        if savings <= 0:
            labels = ['Housing', 'Food', 'Transportation', 'Utilities', 'Entertainment']
            sizes = [housing, food, transportation, utilities, entertainment]
            colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
        else:
            labels = ['Savings', 'Housing', 'Food', 'Transportation', 'Utilities', 'Entertainment']
            sizes = [savings, housing, food, transportation, utilities, entertainment]
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Function to format the percentage display
        def func(pct, allvalues):
            absolute = int(pct / 100. * sum(allvalues))
            return f'${absolute} ({pct:.1f}%)'

        wedges, texts, autotexts = ax.pie(
            sizes, 
            colors=colors, 
            autopct=lambda pct: func(pct, sizes), 
            startangle=90,
            textprops={'fontsize': 8}
        )
        ax.axis('equal')

        # Create a legend
        ax.legend(
            wedges, 
            labels, 
            title="Budget Breakdown", 
            loc="center left", 
            bbox_to_anchor=(1, 0.5),
            fontsize='small'
        )

        # Embed the chart in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def visualize_budget(self):
        try:
            # Retrieve inputs dynamically
            income = float(self.entries['MonthlyIncome'].get())
            housing = float(self.entries['HousingExpense'].get())
            food = float(self.entries['FoodExpense'].get())
            transportation = float(self.entries['TransportationExpense'].get())
            utilities = float(self.entries['UtilitiesExpense'].get())
            entertainment = float(self.entries['EntertainmentExpense'].get())

            total_expenses = housing + food + transportation + utilities + entertainment
            savings = income - total_expenses

            # Update savings label
            self.savings_label.config(text=f"Total Savings: ${savings:.2f}")

            # Create pie chart
            self.plot_budget_pie_chart(income, housing, food, transportation, utilities, entertainment)

            # Prepare user input for predictions
            user_input = pd.DataFrame([[transportation, food, utilities, entertainment]],
                                      columns=['TransportationExpense', 'FoodExpense', 'UtilitiesExpense',
                                               'EntertainmentExpense'])

            # Predict expenses
            predicted_expenses = self.predict_expenses(user_input)

            # Update predictions label
            predictions_text = "Predicted Expenses:\n" + "\n".join([
                f"{cat.replace('Expense', '').strip()}: ${amount:.2f}"
                for cat, amount in predicted_expenses.items()
            ])
            self.predictions_label.config(text=predictions_text)

            # Classify spending category
            spending_category = self.classify_spending_category(income, total_expenses)
            self.spending_category_label.config(text=f"Spending Category: {spending_category}")

        except ValueError:
            # Update labels with error messages
            self.savings_label.config(text="Error: Invalid Input")
            self.predictions_label.config(text="Please enter valid numbers")
            self.spending_category_label.config(text="Unable to analyze")

    # Keep the predict_expenses and classify_spending_category methods from the previous implementation

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

    def classify_spending_category(self, user_income, user_total_expenses):

        # Ensure the dataframe has TotalExpenses column
        self.df['TotalExpenses'] = self.df[['HousingExpense', 'TransportationExpense', 'FoodExpense',
                                            'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)

        self.df['ExpenseIncomeRatio'] = self.df['TotalExpenses'] / self.df['Income']
        # Ensure the dataframe has TotalExpenses column
        self.df['TotalExpenses'] = self.df[['HousingExpense', 'TransportationExpense', 'FoodExpense',
                                            'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)

        self.df['ExpenseIncomeRatio'] = self.df['TotalExpenses'] / self.df['Income']

        scaler = StandardScaler()
        scaled_expense_income_ratio = scaler.fit_transform(self.df[['ExpenseIncomeRatio']])

        kmeans = KMeans(n_clusters=3, random_state=42, max_iter=500, n_init="auto")

        self.df['SpendingCategory'] = kmeans.fit_predict(scaled_expense_income_ratio)
        print(self.df[['TotalExpenses', 'Income', 'ExpenseIncomeRatio', 'SpendingCategory']])

        category_mapping = {0: 'Average', 1: 'Frugal', 2: 'Spender'}
        self.df['SpendingCategory'] = self.df['SpendingCategory'].map(category_mapping)

        user_expense_income_ratio = float(user_total_expenses / user_income)
        user_df = pd.DataFrame([[user_expense_income_ratio]], columns=['UserExpenseIncomeRatio'])
        print(f'user_expense_income_ratio: {user_expense_income_ratio}')

        # Scale the user input for prediction
        # scaled_user_expense_income_ratio = scaler.fit_transform(user_expense_income_ratio)
        # print(f'scaled_user_expense_income_ratio: {scaled_user_expense_income_ratio}')

        # Predict spending category for the user input
        print(kmeans.labels_)
        print(kmeans.cluster_centers_)
        category_prediction = kmeans.predict(user_df[['UserExpenseIncomeRatio']])
        print(f'category_prediction: {category_prediction}')
        return category_mapping[category_prediction[0]]
