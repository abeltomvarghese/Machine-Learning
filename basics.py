import pandas as pd
filePath = "dataset/house-prices/train.csv"
house_data = pd.read_csv(filePath)
print(house_data.columns)
price_column = house_data.SalePrice
print(price_column.head())
columns_of_interest = ["GarageCond","Electrical"]
columns_of_data = house_data[columns_of_interest]
print(columns_of_data.describe())