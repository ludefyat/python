import time
from keras.layers import LSTM

TICKER_CFG = [
    {"ticker": "AAPL", "name": "Apple", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 20, "batch_size": 16, "epochs": 300},
    {"ticker": "AAPL", "name": "Apple", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 80, "batch_size": 128, "epochs": 300},
    {"ticker": "NVDA", "name": "NVIDIA", "interval": "month", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 40, "batch_size": 32, "epochs": 300},
    {"ticker": "NVDA", "name": "NVIDIA", "interval": "week", "n_layers": 2, "units": 256, "dropout": 0.0, "n_steps": 50, "batch_size": 64, "epochs": 300}
]

try:
    from my_conf import *
    TICKER_CFG = MY_TICKER
except ImportError:
    print("No personal ticker found")

#define dir to store stock data, log, checkpoint data, dashboard, "" to ignore
DIRS_CFG = {
    "data": "data",
    "log": "logs",
    "checkpoint": "results",
    "dash": "dashboard"
}

# features to use
FEATURE_COLUMNS = ["open", "high", "low", "close", "adjclose", "volume"]
#PREDICT_COLUMNS = ["close", "high", "low"]
PREDICT_COLUMNS = ["close", "low"]
interval_dict = {"day": '1d', "week": '1wk', "month": '1mo'}
WEEK_BACKTRACK = 0

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





