from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def get_mae(x_train, x_test, y_train,y_test):
    model = RandomForestRegressor()
    model.fit(x_train,y_train)
    predict = model.predict(x_test)
    return mean_absolute_error(y_test,predict)


filePath = "dataset/train.csv"
iowaData = pd.read_csv(filePath)
y = iowaData.SalePrice
price_predictors = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = iowaData[price_predictors]
train_x, test_x, train_y, test_y = train_test_split(x,y,random_state=0)
x_train = train_x.copy()
x_test = test_x.copy()
missing_cols = (col for col in train_x.columns if train_x[col].isnull().any())
for col in missing_cols:
    x_train[col + "_was_missing"] = x_train[col].isnull()
    x_test[col + "_was_missing"] = x_test[col].isnull()
my_imputer = Imputer()
x_train = my_imputer.fit_transform(x_train)
x_test = my_imputer.fit_transform(x_test)
print("Mean absolute error from imputation while track what was imputed:")
print(get_mae(x_train,x_test,train_y,test_y))