import subprocess
from findatapy.util import SwimPool; SwimPool()

from findatapy.market import Market, MarketDataRequest, MarketDataGenerator

import numpy as np
import pandas as pd

from arctic import Arctic
from arctic.date import DateRange

import torch
import torch.utils.data as utils

class DB():
    def __init__(self,
        mongod=r'E:\mongodb-win32-x86_64-2008plus-ssl-4.0.6\bin\mongod.exe',
        dbPath=r'E:\mongodb-win32-x86_64-2008plus-ssl-4.0.6\bin\data',
        address='127.0.0.1'):

        self.mongod = mongod
        self.dbPath = dbPath
        self.address=address

    def startDB(self, storename = 'fx'):
        self.con = subprocess.Popen("%s %s %s" %(self.mongod,
                                    "--dbpath",
                                    self.dbPath),
                                    shell=True)

        self.store = Arctic(self.address)

        if not self.store.library_exists('fx'):
            self.store.initialize_library(storename)

        self.library = self.store[storename]
        self.rc =  self.con.returncode


    def readFxData(self, name = 'EURUSD', version = 1, start = '2016-07-01', end = '2016-07-02'):
        return self.library.read(name, as_of = version, date_range=DateRange(start, end))


    def writeData2DB(self, df, name):
        self.library.write(name, df)


    def __del__(self):
        print ("terminating db connection.")
        self.con.terminate()


class Reader():
    def __init__(self, datasource = 'dukascopy', category = 'fx'):
        self.datasource = datasource
        self.category = category

    def getFxData(self, startDate = '14 Jun 2016', endDate = '15 Jun 2016',
                  tickers = ['EURUSD'], fields = ['close'], frequency = 'tick'):
        md_request = MarketDataRequest(start_date = startDate, finish_date = endDate,
                                       category = self.category, fields = fields,
                                       freq = frequency, data_source = self.datasource,
                                       tickers = tickers)

        market = Market(market_data_generator = MarketDataGenerator())
        return market.fetch_market(md_request)


class Interval():
    def __init__(self, interval):
        self.interval = interval

    def toInt(self):
        return int(''.join([d for d in self.interval if d.isdigit()]))

    def toString(self):
        return str(''.join([d for d in self.interval if not d.isdigit()]))

    def toFullString(self):
        if self.toString().upper() in('H', 'HOUR'):
            return 'hour'
        elif self.toString().upper() in ('M', 'MINUTE', 'MIN'):
            return 'minute'
        elif self.toString().upper() in ('S', 'SECOND', 'SEC'):
            return 'second'
        elif self.toString().upper() in ('D', 'DAY'):
            return 'day'
        else:
            raise NotImplementedError


class Imputator():
    def __init__(self, df, interval):
        self.df = df
        self.interval = interval

    def impute(self):

        if self.interval.toFullString() == 'hour':
            df_mean = self.df.groupby(self.df.index.hour).mean()
        elif self.interval.toFullString() == 'minute':
            df_mean = self.df.groupby(self.df.index.minute).mean()
        elif self.interval.toFullString() == 'day':
            df_mean = self.df.groupby(self.df.index.day).mean()

        for i in range(len(self.df)):
            if np.isnan(self.df.iloc[i]):
                if self.interval.toFullString() == 'hour':
                    self.df.iloc[i] = df_mean[self.df.index[i].hour]
                elif self.interval.toFullString() == 'minute':
                    self.df.iloc[i] = df_mean[self.df.index[i].minute]
                elif self.interval.toFullString() == 'day':
                    self.df.iloc[i] = df_mean[self.df.index[i].day]

        return self.df


class Aggregator():
    def __init__(self, df, interval='1Min', aggregation_function = np.average):
        self.interval = Interval(interval)
        self.df = df
        self.ag_df = None
        self.agg_func = aggregation_function


    def impute(self):
        return self.imputator.impute()


    def aggregate(self):

        self.ag_df = self.df.groupby(pd.TimeGrouper(freq=self.interval.toString())).aggregate(self.agg_func)

        #self.ag_df = self.df.groupby(pd.Grouper(freq=self.interval.toString())).transform(self.agg_func)
        self.imputator = Imputator(self.ag_df, self.interval)

        self.ag_df = self.impute()

class Slicer():
    def __init__(self, df, start, end, loockback = 5,
            n_step_ahead_prediction = 1):
        self.df = df
        self.start = start
        self.end = end
        self.loockback = loockback
        #self.training_window = training_window
        self.n_step_ahead_prediction = n_step_ahead_prediction

    def getDf(self):
        start = self.df.index.get_loc(self.start)
        end = self.df.index.get_loc(self.end)

        return self.df.iloc[start:end]

    def getTrainingSet(self):

        look_back_bins = self.loockback

        n_step_ahead_prediction = self.n_step_ahead_prediction

        training_variables = self.df.values
        fit_variables = self.df.values

        start = self.df.index.get_loc(self.start)
        end = self.df.index.get_loc(self.end)

        regression_window = end - start

        assert(len(training_variables) >= start-end-look_back_bins-regression_window)

        tensor_x = torch.stack([torch.FloatTensor(training_variables[start - look_back_bins - i - 1:
                                                                     start - 1 - i].reshape(look_back_bins,
                                                                    1)) for i in range(regression_window)])

        tensor_y = torch.stack([torch.FloatTensor(fit_variables[start - i - n_step_ahead_prediction:
                                                                start - i])for i in range(regression_window)])

        dataset = utils.TensorDataset(tensor_x, tensor_y)
        dataloader = utils.DataLoader(dataset)

        return dataloader