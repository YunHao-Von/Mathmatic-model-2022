import pandas as pd


def excel_to_csv(file_path):
    excel_data = pd.read_excel(file_path)
    save_path = file_path[:-4]+'csv'
    excel_data = excel_data.fillna(0)
    excel_data.to_csv(save_path, index=False)