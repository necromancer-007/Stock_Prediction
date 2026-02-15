import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/silver_prices_historical.csv')
data = data.drop(columns=["Date"])
data['Daily_Return'] = data['Close'].pct_change()
data['returns'] = data['Daily_Return'].shift(-1)
data = data.dropna()

x = data[['High','Low','Open','Close','Volume']]
y = data['returns']

model = LinearRegression()
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=20)

model.fit(xtrain,ytrain)
predict = model.predict(xtest)

error = np.sqrt(mean_squared_error(ytest,predict))
print ('YTest : ',ytest,'\npredict : \n',predict[:10])
print ('\nRMSE : ',error,'\nStandard Deviation : ',y.std())

base_pred = np.full_like(ytest, ytrain.mean())
base_rmse = np.sqrt(mean_squared_error(ytest, base_pred))

print("Base RMSE:", base_rmse)
print("Predicted RMSE:", error)
