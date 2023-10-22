# Required Libraries:
# - Pandas: For data manipulation and analysis.
# - NumPy: For numerical operations.
# - Matplotlib: For plotting and visualizations.
# - Sklearn: For data preprocessing and metrics.
# - Keras: For deep learning model construction and training.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# --- Step 1: Data Loading and Visualization ---

# Load the dataset from the given URL.
# The 'parse_dates' argument is used to specify that the 'Month' column should be parsed as a date.
# The 'index_col' argument is used to set the 'Month' column as the DataFrame index.

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# Plot the loaded data for visualization. 
# This helps in understanding the trend and seasonality in the data.

data.plot()
plt.title('Monthly Airline Passengers (1949-1960)')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.show()

# --- Step 2: Data Preprocessing ---

# We normalize the dataset to a range between 0 and 1.
# Neural networks, especially LSTMs, perform better with normalized data.

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Function to structure time series data for supervised learning.
# For each entry, the function will organize data to predict the next time point's value based on the current value.

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
    # Aggregate data in a single dataframe
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # Remove rows with NaN values, if specified
    if dropnan:
        agg.dropna(inplace=True)
        
    return agg.values

# Apply the function to transform data for supervised learning

values = series_to_supervised(scaled_data, 1, 1)

# Split the dataset into training (80%) and testing (20%)

n_train = int(len(values) * 0.8)
train = values[:n_train, :]
test = values[n_train:, :]

# Split datasets into input (X) and output (y) variables

X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

# Reshape input data into the 3D format required by LSTM: [samples, timesteps, features]

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# --- Step 3: Model Training ---

# Construct an LSTM model.
# We use a single LSTM layer followed by a dense layer to produce a prediction.

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))  # LSTM layer with 50 neurons
model.add(Dense(1))  # Output layer
model.compile(loss='mae', optimizer='adam')

# Train the LSTM model using the training data

model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)

# --- Step 4: Prediction & Evaluation ---

# Use the trained model to make predictions on the test dataset

yhat = model.predict(X_test)

# Convert the normalized predictions and actual values back to their original scale

inv_yhat = scaler.inverse_transform(yhat)
y_test_reshaped = y_test.reshape(-1, 1)
inv_y = scaler.inverse_transform(y_test_reshaped)

# Calculate the Root Mean Square Error (RMSE) to quantify the model's performance

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE:', rmse)

# --- Step 5: Visualization of Predicted vs Actual Values ---

# Plot the actual vs. predicted values to visually inspect the model's performance

plt.plot(inv_y, label="Actual")
plt.plot(inv_yhat, label="Predicted")
plt.legend()
plt.show()
