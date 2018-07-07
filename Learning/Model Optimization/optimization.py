from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
def get_mae(max_leaf_nodes, predictors_train, predictors_test, target_train, target_test):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)
    model.fit(predictors_train, target_train)
    preds_test = model.predict(predictors_test)
    mae = mean_absolute_error(target_test,preds_test)
    return mae

file_path = "dataset/train.csv"
iowaData = pd.read_csv(file_path)
y = iowaData.SalePrice
price_predictors = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = iowaData[price_predictors]
train_x, val_x, train_y, val_y = train_test_split(x,y,random_state=0)
for max_leaf_nodes in [4,5,40,50,400,500,5000]:
    my_mae = get_mae(max_leaf_nodes,train_x,val_x,train_y,val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes,my_mae))
