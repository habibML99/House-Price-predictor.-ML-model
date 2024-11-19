import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metric import mean_squared_error
fron sklearn.model_selection import train_test_split

Dataset = pd.read_csv("your_housing_data.csv")
X_data = Dataset.drop("price", axis = 1)
y_data = Dataset(["price"])

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2785, random_state = 42)

model = Pipeline({
    "regressor": LinearRegression(),
    "features" : StandardScaler()
})

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("mean_error :" ,mean_squared_error(y_pred,y_test)
print("difference : ", y_pred, y_test)


def predict_personal_data() : -> np.array 
 pass


'''
umeed he jis khushi ki is Dunia ko,
us ko dainey Wala tou har jagha he , magar  leney Wala he kon... 

na Mili insaaf o idalat tou kia hoa, 
beghair mushkil us ko yaad karta he kon...

'''