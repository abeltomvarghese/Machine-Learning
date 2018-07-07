from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
df_train = pd.read_csv("dataset/train.csv")
df_test = pd.read_csv("dataset/test.csv")

obj_df = df_train.select_dtypes(include=["object", "int64", "float64", "float32"]).copy()
new_df = df_test.select_dtypes(include=["object", "int64", "float64", "float32"]).copy()
obj_df["Neighborhood"] = obj_df["Neighborhood"].astype("category")
obj_df["NhCodes"] = obj_df["Neighborhood"].cat.codes
obj_df["GarageType"] = obj_df["GarageType"].astype("category")
obj_df["GT"] = obj_df["GarageType"].cat.codes
obj_df["SaleType"] = obj_df["SaleType"].astype("category")
obj_df["ST"] = obj_df["SaleType"].cat.codes
obj_df["HeatingQC"] = obj_df["HeatingQC"].astype("category")
obj_df["HQC"] = obj_df["HeatingQC"].cat.codes
new_df["Neighborhood"] = new_df["Neighborhood"].astype("category")
new_df["NhCodes"] = new_df["Neighborhood"].cat.codes
new_df["GarageType"] = new_df["GarageType"].astype("category")
new_df["GT"] = new_df["GarageType"].cat.codes
new_df["SaleType"] = new_df["SaleType"].astype("category")
new_df["ST"] = new_df["SaleType"].cat.codes
new_df["HeatingQC"] = new_df["HeatingQC"].astype("category")
new_df["HQC"] = new_df["HeatingQC"].cat.codes
#garage car sale type heating quality
cols = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd", "GarageCars", "HQC", "ST", "GT", "NhCodes"]

print(new_df["LotArea"].value_counts())
new_df = new_df.fillna({"GarageCars":2})
obj_df = obj_df.fillna({"GarageCars":2})
new_df = new_df.fillna({"HQC":0})
obj_df = obj_df.fillna({"HQC":0})
new_df = new_df.fillna({"ST":8})
obj_df = obj_df.fillna({"ST":8})
new_df = new_df.fillna({"GT":1})
obj_df = obj_df.fillna({"GT":1})
new_df = new_df.fillna({"NhCodes":12})
obj_df = obj_df.fillna({"NhCodes":12})
new_df = new_df.fillna({"TotRmsAbvGrd":6})
obj_df = obj_df.fillna({"TotRmsAbvGrd":6})
new_df = new_df.fillna({"BedroomAbvGr":3})
obj_df = obj_df.fillna({"BedroomAbvGr":3})
new_df = new_df.fillna({"FullBath":2})
obj_df = obj_df.fillna({"FullBath":2})
new_df = new_df.fillna({"2ndFlrSF":0})
obj_df = obj_df.fillna({"2ndFlrSF":0})
new_df = new_df.fillna({"1stFlrSF":864})
obj_df = obj_df.fillna({"1stFlrSF":864})
new_df = new_df.fillna({"YearBuilt":2005})
obj_df = obj_df.fillna({"YearBuilt":2005})
new_df = new_df.fillna({"LotArea":9600})
obj_df = obj_df.fillna({"LotArea":9600})
y = df_train.SalePrice
train_x = obj_df[cols]
my_model = RandomForestRegressor()
my_model.fit(train_x,y)
test_x = new_df[cols]
predict_price = my_model.predict(test_x)
print(predict_price)
my_submission = pd.DataFrame({"Id": df_test.Id, "SalePrice": predict_price })
my_submission.to_csv("new.csv", index=False)
