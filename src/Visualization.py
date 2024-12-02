import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self, df):
        self.df = df
        self.average_expenses = df[['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']].mean()
        self.total_monthly_expenses = df[['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)
        self.budgeted_expenses = budgeted_expenses = self.df[
            ['HousingBudgetAmount', 'TransportationBudgetAmount', 'FoodBudgetAmount', 'UtilitiesBudgetAmount',
             'EntertainmentBudgetAmount']].mean()
        self.actual_expenses = actual_expenses = self.df[['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense',
                              'EntertainmentExpense']].mean()


    def display_adv_pie_chart(self):
        pass

    def display_heatmap(self):
        # Note: I don't think comparison_df is used at all, maybe it will be used in the future?
        # Combine into a single DataFrame for plotting
        comparison_df = pd.DataFrame({
            'Budgeted': self.budgeted_expenses,
            'Actual': self.actual_expenses
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

    def display_bar_chart(self):
        # Combine averages into a single DataFrame
        avg_comparison_df = pd.DataFrame({
            'Actual Expenses': self.average_expenses,
            'Budgeted Amounts': self.budgeted_expenses
        })

        # Bar Chart
        avg_comparison_df.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'orange'])
        plt.title("Average Expenses vs. Budgeted Amounts")
        plt.ylabel("Amount")
        plt.xlabel("Expense Categories")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.show()

    def display_adv_line_plt(self):
        # Assuming 'Month' is a column in your DataFrame for monthly data
        # Calculate total monthly expenses and budgets
        self.df['TotalExpenses'] = self.df[
            ['HousingExpense', 'TransportationExpense', 'FoodExpense', 'UtilitiesExpense', 'EntertainmentExpense']].sum(
            axis=1)
        self.df['TotalBudget'] = self.df[
            ['HousingBudgetAmount', 'TransportationBudgetAmount', 'FoodBudgetAmount', 'UtilitiesBudgetAmount',
             'EntertainmentBudgetAmount']].sum(axis=1)

        # Line Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.df['UserID'], self.df['TotalExpenses'], label='Total Expenses', marker='o')
        plt.plot(self.df['UserID'], self.df['TotalBudget'], label='Total Budget', marker='x')
        plt.title("Total Monthly Expenses vs. Total Budget")
        plt.xlabel("User ID")
        plt.ylabel("Amount")
        plt.legend()
        plt.grid()
        plt.show()

    def display_adv_scatter_plt(self):
        # Scatter Plot of Actual vs. Budgeted
        plt.figure(figsize=(10, 6))

        # Set transparency for better visibility of overlapping points
        plt.scatter(self.df['HousingBudgetAmount'], self.df['HousingExpense'], color='blue', alpha=0.6, label='Housing')
        plt.scatter(self.df['TransportationBudgetAmount'], self.df['TransportationExpense'], color='orange', alpha=0.6,
                    label='Transportation')
        plt.scatter(self.df['FoodBudgetAmount'], self.df['FoodExpense'], color='green', alpha=0.6, label='Food')
        plt.scatter(self.df['UtilitiesBudgetAmount'], self.df['UtilitiesExpense'], color='red', alpha=0.6, label='Utilities')
        plt.scatter(self.df['EntertainmentBudgetAmount'], self.df['EntertainmentExpense'], color='purple', alpha=0.6,
                    label='Entertainment')

        # Diagonal line for budget vs. actual comparison
        plt.plot([0, 10000], [0, 6000], 'k--', label='Budget = Actual')  # Diagonal line with specified limits
        plt.xlim(0, 10000)  # Limit x-axis to 10,000
        plt.ylim(0, 6000)  # Limit y-axis to 6,000

        # Title and labels
        plt.title("Actual Expenses vs. Budgeted Amounts (Budget Max: 10,000, Actual Max: 6,000)")
        plt.xlabel("Budgeted Amount")
        plt.ylabel("Actual Expense")

        # Enhanced legend and grid
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

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

    def display_histogram(self, category, num_of_bins, title, x_lab, y_lab):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['Income'], bins=10, kde=True, color='blue')
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