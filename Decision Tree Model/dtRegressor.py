from sklearn.tree import DecisionTreeRegressor
import pandas as pd
filePath = "dataset/train.csv"
iowaData = pd.read_csv(filePath)
y = iowaData.SalePrice
price_predictors = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = iowaData[price_predictors]
iowa_model = DecisionTreeRegressor()
iowa_model.fit(x,y)
print(x.head())
print("predictions are: ")
print(iowa_model.predict(x.head()))
