## MLP4timeSeries
Simple example of deep neural network for financial time series forecasting, including no sql database store and data retrieval.
Implemented a deep neural network and a convolutional neural network.

# Example
import fxdata

r = fxdata.Reader()

df = r.getFxData(startDate= '01 Jan 2016', endDate = '31 Dec 2016', tickers = ['EURGBP'])

d = fx.data.DB()

d.startDB()

d.writeData2DB(df, 'EURGBP')

import client

Example running main in client.py:

![act_v_predict](https://github.com/1nwonknu/MLP4timeSeries/blob/master/actual_v_predicted.png "EUR/GBP Exchange rate")

# Requirements
- Mongodb
- tensorflow
- pyTorch
- arctic
- findatapy
