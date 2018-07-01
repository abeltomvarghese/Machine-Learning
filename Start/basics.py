import pandas as pd
filePath = "dataset/house-prices/train.csv"
house_data = pd.read_csv(filePath)
print(house_data.columns)  #prints a list of columns
price_column = house_data.SalePrice
print(price_column)
columns_of_interest = ["GarageCond","Electrical"]
columns_of_data = house_data[columns_of_interest]
pool = house_data.PoolQC
pool = pool.dropna()
print(pool)
print(columns_of_data.describe())
