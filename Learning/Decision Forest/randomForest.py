from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

filePath = "dataset/train.csv"
iowaData = pd.read_csv(filePath)
y = iowaData.SalePrice
pricePred = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = iowaData[pricePred]
train_x, val_x, train_y, val_y = train_test_split(x,y,random_state=0)
data_model = RandomForestRegressor()
data_model.fit(train_x,train_y)
iowa_predict = data_model.predict(val_x)
print(mean_absolute_error(val_y,iowa_predict))