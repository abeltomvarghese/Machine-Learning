from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd

def getMae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(40),
                                X, y,
                                scoring = 'neg_mean_absolute_error').mean()


train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")
train.dropna(axis=0,subset=["SalePrice"],inplace=True)
target = train.SalePrice
cols_with_missing = [col for col in train.columns if train[col].isnull().any()]
can_train = train.drop(["Id", "SalePrice"] + cols_with_missing, axis = 1)
can_test = test.drop(["Id"] + cols_with_missing, axis = 1)
low_cardinal_cols = [cName for cName in can_train.columns if can_train[cName].nunique() <10 and can_train[cName].dtype == "object"]
numeric_cols = [cname for cname in can_train.columns if can_train[cname].dtype in ["int64", "float64"]]
my_cols = low_cardinal_cols + numeric_cols
train_pred = can_train[my_cols]
test_pred = can_test[my_cols]
encoded = pd.get_dummies(train_pred)
test_encode = pd.get_dummies(test_pred)
final_train, final_test = encoded.align(test_encode, join="left", axis=1)
mae = getMae(final_train,target)
print(mae)
