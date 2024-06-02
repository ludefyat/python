import time
from keras.layers import LSTM

TICKER_CFG = [
        {"ticker": "AMZN", "name": "AMAZON", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 32, "epochs": 300},
        {"ticker": "AMZN", "name": "AMAZON", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 64, "epochs": 300}
]

try:
    from my_conf import *
    TICKER_CFG += MY_TICKER
except ImportError:
    print("No personal ticker found")

#define dir to store stock data, log, checkpoint data, dashboard, "" to ignore
DIRS_CFG = {
    "data": "data",
    "log": "logs",
    "checkpoint": "results",
    "dash": "dashboard"
}

#ticker = "AMZN"  # Amazon stock market
# features to use
FEATURE_COLUMNS = ["open", "high", "low", "close", "adjclose", "volume"]
PREDICT_COLUMNS = ["close", "low"]
#PREDICT_COLUMNS = ["high", "low"]
#DIR_DASH = "dashboard"
interval_dict = {"day": '1d', "week": '1wk', "month": '1mo'}
date_now = time.strftime("%Y-%m-%d") #"2024-5-01" 

""" TICKER_CFG = [
    #{"ticker": "AMZN", "name": "AMAZON", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 32, "epochs": 300},
    #{"ticker": "0991.HK",   "name": "大唐发电", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 20, "batch_size": 32, "epochs": 300},
    {"ticker": "601811.SS", "name": "新华文轩", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 32, "epochs": 300},
    {"ticker": "601811.SS", "name": "新华文轩", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 32, "epochs": 200},
    {"ticker": "001965.SZ", "name": "招商公路", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 1, "epochs": 100},
    {"ticker": "001965.SZ", "name": "招商公路", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 64, "epochs": 300},
    {"ticker": "600025.SS", "name": "华能水电", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 32, "epochs": 300},
    {"ticker": "600025.SS", "name": "华能水电", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 32, "epochs": 200},
    {"ticker": "0991.HK",   "name": "大唐发电", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 32, "epochs": 300},
    {"ticker": "0991.HK",   "name": "大唐发电", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 32, "epochs": 200},
    {"ticker": "601816.SS", "name": "京沪高铁", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 32, "epochs": 300},
    {"ticker": "601816.SS", "name": "京沪高铁", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 64, "epochs": 200},
    {"ticker": "600377.SS", "name": "宁沪高速", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 32, "epochs": 200},
    {"ticker": "600377.SS", "name": "宁沪高速", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 64, "epochs": 200},
    {"ticker": "600348.SS", "name": "华阳股份", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 32, "epochs": 300},
    {"ticker": "600348.SS", "name": "华阳股份", "interval": "week",  "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 64, "epochs": 200},
    {"ticker": "000858.SZ", "name": "五粮液",   "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 32, "epochs": 300},
    {"ticker": "000858.SZ", "name": "五粮液",   "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 64, "epochs": 100},
    {"ticker": "601985.SS", "name": "中国核电", "interval": "month", "n_layers": 2, "units": 128, "dropout": 0.0, "n_steps": 20, "batch_size": 32, "epochs": 200},
    {"ticker": "601985.SS", "name": "中国核电", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 32, "epochs": 300},
    {"ticker": "000729.SZ", "name": "燕京啤酒", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 32, "epochs": 200},
    {"ticker": "000729.SZ", "name": "燕京啤酒", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 64, "epochs": 200},
    {"ticker": "601198.SS", "name": "东兴证券", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 30, "batch_size": 32, "epochs": 300},
    {"ticker": "601198.SS", "name": "东兴证券", "interval": "week",  "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 32, "epochs": 200},
] """

# whether to scale feature columns & output price as well
SCALE = True
CELL = LSTM
BIDIRECTIONAL = False
LOSS = "huber_loss"
OPTIMIZER = "adam"
LOOKUP_STEP = 1
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
# whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2





