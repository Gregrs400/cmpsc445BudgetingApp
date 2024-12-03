
class User:
    def __init__(self, user_ID):
        self.user_ID = user_ID
        self.expenses = {}

    def add_expense(self, category, expense):
        self.expenses[category] = expense

    def get_expense(self, category):
        return self.expenses[category]

    def visualize_expenses(self):
        pass

    def get_recommendations(self):
        pass