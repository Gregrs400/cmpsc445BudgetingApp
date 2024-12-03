import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansModel:
    category_mapping = {
        0: 'Frugal',  # Example mapping
        1: 'Average',  # Example mapping
        2: 'Spender'  # Example mapping
    }

    def __init__(self, num_of_clusters):
        self.num_of_clusters = num_of_clusters
        self.model = KMeans(n_clusters=num_of_clusters, random_state=42)

    @staticmethod
    def load_dataset():
        return pd.read_csv("budgeting_dataset.csv")

    def calculate_total_expenses(self, data_frame):
        # Returns sum of all expenses for each user
        return data_frame[['HousingExpense', 'TransportationExpense', 'FoodExpense',
                           'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)

    def calculate_expense_ratio(self, data_frame):
        # Calculates the % of income spent by each user
        return data_frame['TotalExpenses'] / data_frame['Income']

    def train(self, data_frame):
        scaler = StandardScaler()
        return scaler.fit_transform(data_frame[['TotalExpenses']])


    def predict(self, kmeans, scaled_expenses):
        return kmeans.fit_predict(scaled_expenses)

    def get_cluster_analysis(self, data_frame):
        # Create criteria for each category
        # Example: <40% of income spent = frugal
        #           >40% but <60% of income spent = average
        #           >60% of income spent = spender
        data_frame['SpendingCategory'] = data_frame['SpendingCategory'].map(self.category_mapping)

        print(data_frame[['UserID', 'TotalExpenses', 'SpendingCategory']].head())

# # K Means Clustering
#
# # Step 1: Load your dataset
# df = pd.read_csv("budgeting_dataset.csv")
#
# # Step 2: Calculate Total Expenses
# df['TotalExpenses'] = df[['HousingExpense', 'TransportationExpense', 'FoodExpense',
#                             'UtilitiesExpense', 'EntertainmentExpense']].sum(axis=1)
#
# # Step 3: Prepare data for clustering (you can scale if necessary)
# scaler = StandardScaler()
# scaled_expenses = scaler.fit_transform(df[['TotalExpenses']])
#
# # Step 4: Train KMeans model
# kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
# df['SpendingCategory'] = kmeans.fit_predict(scaled_expenses)
## # Step 5: Check unique labels
# unique_labels = set(kmeans.labels_)
# print(f"Unique labels: {unique_labels}")
#
# # Step 6: Create a mapping for categories
# category_mapping = {
#     0: 'Frugal',   # Example mapping
#     1: 'Average',  # Example mapping
#     2: 'Spender'   # Example mapping
# }
#
# # Step 7: Map spending categories
# df['SpendingCategory'] = df['SpendingCategory'].map(category_mapping)
#
# # Step 8: Display results
# print(df[['UserID', 'TotalExpenses', 'SpendingCategory']].head())
