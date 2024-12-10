import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import tkinter as tk
import numpy as np
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
        self.main_frame.pack(padx=20, pady=20, fill='both', expand=True)

        # Create sections
        self.create_header()
        self.create_input_section()
        self.create_buttons()  # Create buttons before result section
        self.create_result_section()

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

    def create_buttons(self):
        # Button frame
        self.button_frame = tk.Frame(self.main_frame, bg='#ffffff')
        self.button_frame.pack(fill='x', pady=20)  # Pack with fill='x'

        # Analyze button
        analyze_button = tk.Button(
            self.button_frame, 
            text="Analyze Budget", 
            command=self.visualize_budget, 
            font=('Helvetica', 12, 'bold'),
            bg='#3498db', 
            fg='white', 
            relief='flat',
            padx=20, 
            pady=10
        )
        analyze_button.pack(side=tk.LEFT, padx=10)

        # Reset button
        reset_button = tk.Button(
            self.button_frame,
            text="Reset", 
            command=self.reset_gui,
            font=('Helvetica', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            relief='flat',
            padx=20,
            pady=10
        )
        reset_button.pack(side=tk.LEFT, padx=10)

    def create_result_section(self):
        # Result frame for displaying analysis
        self.result_frame = tk.Frame(self.main_frame, bg='#ffffff')
        self.result_frame.pack(fill='both', expand=True)

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

        # Create separate container for chart
        self.chart_container = tk.Frame(self.result_frame, bg='#ffffff')
        self.chart_container.pack(fill='both', expand=True)

        # Placeholder for pie chart
        self.chart_frame = tk.Frame(self.chart_container, bg='#ffffff')
        self.chart_frame.pack(fill='both', expand=True)

    def plot_budget_pie_chart(self, income, housing, food, transportation, utilities, entertainment):
        # Clear previous chart if exists
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Calculate total expenses and savings
        total_expenses = housing + food + transportation + utilities + entertainment
        savings = income - total_expenses

        # Create a pie chart
        pie_chart_items = [['Savings', savings, '#2ecc71', ], ['Housing', housing, '#e74c3c'], ['Food', food, '#3498db'], ['Transportation', transportation, '#f39c12'], ['Utilities', utilities, '#9b59b6'], ['Entertainment', entertainment, '#1abc9c']]
        labels = []
        sizes = []
        colors = []
        for item in pie_chart_items:
            if item[1] > 0:
                labels.append(item[0])
                sizes.append(item[1])
                colors.append(item[2])

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

    def reset_gui(self):
        # Clear all entry fields
        for entry in self.entries.values():
            entry.delete(0, tk.END)

        # Reset labels to default values
        self.savings_label.config(text="Total Savings: $0")
        self.spending_category_label.config(text="Spending Category: N/A")
        self.predictions_label.config(text="Predicted Expenses:\n")

        # Clear the chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()


    def predict_expenses(self, user_input):
        # Define input and output features
        input_features = ['Income', 'TransportationExpense', 'FoodExpense', 
                        'UtilitiesExpense', 'EntertainmentExpense']
        expense_categories = ['HousingExpense', 'TransportationExpense', 
                            'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']
        
        # Initialize dictionary for predictions
        predictions = {}
        
        # Scale all features once
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[input_features])
        
        # Get user's data and scale it using the same scaler
        user_income = float(self.entries['MonthlyIncome'].get())
        user_data = pd.DataFrame([[
            user_income,
            float(self.entries['TransportationExpense'].get()),
            float(self.entries['FoodExpense'].get()),
            float(self.entries['UtilitiesExpense'].get()),
            float(self.entries['EntertainmentExpense'].get())
        ]], columns=input_features)
        user_scaled = scaler.transform(user_data)
        
        # Predict each expense category
        for category in expense_categories:
            # For each category, we'll use income and the other expenses as features
            # (excluding the current category if it's one we already have)
            feature_mask = [col != category for col in input_features]
            prediction_features = X_scaled[:, feature_mask]
            user_prediction_features = user_scaled[:, feature_mask]
            
            # Get the target values for this category
            y = self.df[category]
            
            # Create and train the model
            knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
            knn_model.fit(prediction_features, y)
            
            # Make prediction
            prediction = knn_model.predict(user_prediction_features)[0]
            
            # Store the prediction, ensuring it's not negative
            predictions[category] = max(0, prediction)
        
        # Adjust predictions to maintain reasonable proportions
        total_predicted = sum(predictions.values())
        total_actual = float(user_income) * 0.9  # Assuming people typically spend 90% of income
        
        if total_predicted > total_actual:
            # Scale down predictions to match typical spending
            scale_factor = total_actual / total_predicted
            predictions = {k: v * scale_factor for k, v in predictions.items()}
        
        return predictions

    def visualize_budget(self):
        try:
            # Retrieve inputs
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

            # Get predictions
            predicted_expenses = self.predict_expenses(None)  # 'None' since we now get values from entries directly

            # Update predictions label with formatted currency values
            predictions_text = "Predicted Expenses:\n" + "\n".join([
                f"{cat.replace('Expense', '').strip()}: ${amount:,.2f}"
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

    def classify_spending_category(self, user_income, user_total_expenses):
        # Calculate expense ratios for training data
        self.df['TotalExpenses'] = self.df[['HousingExpense', 'TransportationExpense', 'FoodExpense',
                                            'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)
        self.df['ExpenseIncomeRatio'] = self.df['TotalExpenses'] / self.df['Income']

        # Scale the training data
        scaler = StandardScaler()
        scaled_ratios = scaler.fit_transform(self.df[['ExpenseIncomeRatio']])

        # Train KMeans model
        kmeans = KMeans(n_clusters=3, random_state=42, max_iter=500, n_init="auto")
        kmeans.fit(scaled_ratios)

        # Get cluster centers and sort them to determine categories
        centers = kmeans.cluster_centers_.flatten()
        sorted_center_indices = np.argsort(centers)
        
        # Map cluster indices to spending categories based on sorted centers
        category_mapping = {
            sorted_center_indices[0]: 'Frugal',    # Lowest center
            sorted_center_indices[1]: 'Average',    # Middle center
            sorted_center_indices[2]: 'Spender'     # Highest center
        }

        # Calculate and scale user's expense ratio
        user_expense_ratio = user_total_expenses / user_income
        user_scaled_ratio = scaler.transform([[user_expense_ratio]])

        # Predict category for user
        user_cluster = kmeans.predict(user_scaled_ratio)[0]
        return category_mapping[user_cluster]