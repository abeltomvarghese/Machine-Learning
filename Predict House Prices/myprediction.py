from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

train = pd.read_csv("dataset/train.csv")
y = train.SalePrice
cols = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
train_x = train[cols]
my_model = RandomForestRegressor()
my_model.fit(train_x, y)
test = pd.read_csv("dataset/test.csv")
test_x = test[cols]
predict_price = my_model.predict(test_x)
print(predict_price)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predict_price})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)