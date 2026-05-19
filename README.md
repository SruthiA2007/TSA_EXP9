# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 19/05/2026

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 1. Load your uploaded dataset
data = pd.read_csv("/content/salary_prediction_data.csv")

# 2. Prepare data for ARIMA (Simulating a time series by sorting by Age/Experience)
# Since ARIMA requires a sequence, we'll sort by Age and reset the index
data = data.sort_values(by='Age').reset_index(drop=True)

# Create a mock date index so your date-based plotting code still functions
data['date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
data.set_index('date', inplace=True)

# 3. Arima Model Function
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    
    # Fit ARIMA
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()
    
    # Forecast
    forecast = fitted_model.forecast(steps=len(test_data))
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
    
    # Plotting (Fixed the truncated string lines from your snippet)
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date (Mock Sequence)')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()
    
    print("Root Mean Squared Error (RMSE):", rmse)

# 4. Run the model on your target variable 'Salary'
arima_model(data, 'Salary', order=(5,1,0))
```

### OUTPUT:

<img width="818" height="531" alt="image" src="https://github.com/user-attachments/assets/96e7216a-1266-469d-94b3-bd446e9842a0" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
