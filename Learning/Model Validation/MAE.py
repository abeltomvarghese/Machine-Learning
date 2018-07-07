from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
filePath = "dataset/train.csv"
iowaData = pd.read_csv(filePath)
y = iowaData.SalePrice
price_predictors = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = iowaData[price_predictors]
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0)
iowa_model = DecisionTreeRegressor()
iowa_model.fit(train_x,train_y)
val_predict = iowa_model.predict(test_x)
print(mean_absolute_error(test_y, val_predict))