# Time-Series-Prediction-with-LSTM-

Time-Series Prediction with LSTM
This repository contains a practical exercise that demonstrates how to predict a time series using the Long Short-Term Memory (LSTM) network, a type of Recurrent Neural Network (RNN). Specifically, the script predicts the number of airline passengers based on historical data.

## Dataset
The dataset used in this project is the "Monthly Airline Passengers" dataset that captures the monthly count of airline passengers from 1949 to 1960. The dataset can be accessed here.

## Requirements
Python Libraries: Pandas, NumPy, Matplotlib, scikit-learn, Keras


## Dataset: "Monthly Airline Passengers"

## Implementation Steps
Data Loading and Visualization: The dataset is loaded and visualized to understand its trend and seasonality.
Data Preprocessing: The data is normalized to range between 0 and 1 to aid in LSTM performance. It's then structured for supervised learning to predict the next time point's value based on the current value.
Model Training: An LSTM model is constructed and trained using the training dataset.
Prediction & Evaluation: The trained model is used to make predictions on the test set, and its performance is evaluated using the Root Mean Square Error (RMSE).
Visualization: The actual vs. predicted values are plotted to visually inspect the model's performance.
Results
After training, the model's performance is evaluated using the RMSE metric on the test dataset. The script then visualizes the model's predictions against the actual values, providing a clear view of how well the model has learned the underlying patterns in the data.

## How to Run
Ensure you have all the required libraries installed.
Clone this repository.
Run the script using Python:
python time_series_lstm.py
Examine the generated plots and RMSE score to assess the model's performance.


