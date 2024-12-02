from GUI import GUI
import pandas as pd

def main():
    df = pd.read_csv("budgeting_dataset.csv") 
    app = GUI(df).root
    app.mainloop()


main()
