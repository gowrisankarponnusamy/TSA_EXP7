### DEVELOPED BY:GOWRISANKAR P
### REG NO:212222230041
# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM :
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load dataset
file_path = 'FINAL_USO.csv'
data = pd.read_csv(file_path)


# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Extract 'USO_Close' prices
close_prices = data['USO_Close']
# Plot the USO_Close prices over time
plt.figure(figsize=(10, 6))
plt.plot(close_prices, label='USO Close Price')
plt.title('USO Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('USO Close Price')
plt.legend()
plt.grid(True)
plt.show()
# Resample to weekly frequency
weekly_close_prices = close_prices.resample('W').mean()

# Perform ADF test for stationarity
result = adfuller(weekly_close_prices.dropna())
adf_statistic, p_value = result[0], result[1]
print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')
if p_value < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")

# Split data into training and testing sets (80% train, 20% test)
train_size = int(len(weekly_close_prices) * 0.8)
train, test = weekly_close_prices[:train_size], weekly_close_prices[train_size:]

# Plot ACF and PACF of the training data
fig, ax = plt.subplots(2, figsize=(8, 6))
plot_acf(train.dropna(), ax=ax[0], title='Autocorrelation Function (ACF)')
plot_pacf(train.dropna(), ax=ax[1], title='Partial Autocorrelation Function (PACF)')
plt.show()

# Fit an AutoRegressive model (AR) on the training data
ar_model = AutoReg(train.dropna(), lags=13).fit()

# Predict on the test data
ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot the predicted vs actual values for the test set
plt.figure(figsize=(10, 4))
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('AR Model Prediction vs Test Data')
plt.xlabel('Time')
plt.ylabel('USO Close Price')
plt.legend()
plt.show()

# Calculate and display the mean squared error (MSE)
mse = mean_squared_error(test, ar_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Plot train, test, and prediction for comparison
plt.figure(figsize=(10, 4))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('Train, Test, and AR Model Prediction')
plt.xlabel('Time')
plt.ylabel('USO Close Price')
plt.legend()
plt.show()

```
### OUTPUT:

GIVEN DATA:
![image](https://github.com/user-attachments/assets/77e09a18-3292-4471-b7c6-8a8eea669f65)

PACF - ACF:
![image](https://github.com/user-attachments/assets/01eebb76-63f3-45cc-8a17-d6e48f6fe238)

PREDICTION:
![image](https://github.com/user-attachments/assets/123e19d9-74a7-4d61-bdca-0586bf0eb91d)

FINIAL PREDICTION:
![image](https://github.com/user-attachments/assets/de5babe0-c7cb-4383-898c-a65b7137b1d6)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
