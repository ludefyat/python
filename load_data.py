import tensorflow as tf
import numpy as np
import pandas as pd
import random
import sys
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
from datetime import datetime, timedelta

# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

def  end_date_si(ticker, end_date, interval='1wk'):
    if interval == '1wk':
        if end_date.weekday() < 5:
            end_date -= timedelta(days = end_date.weekday() + 7)
        else:
            end_date -= timedelta(days = end_date.weekday())
    elif interval == '1mo':
        next_date = end_date + timedelta(days = 1)
        #judge if the current month data is ready
        if end_date.weekday() < 5 or next_date.month == end_date.month:
            end_date -= timedelta(days=end_date.day)
        end_date = end_date.replace(day=1)
    
    #adjust timezone to EDT
    if ticker[-3:] in [".SS",".SZ",".HK"]:
        end_date -= timedelta(hours=12)
        end_date = end_date.replace(hour=0,minute=0,second=0,microsecond=0)
    
    return end_date

def load_data(ticker, str_end_date, interval):
    end_date = datetime.strptime(str_end_date, "%Y-%m-%d")
    if end_date > datetime.now():
        raise ValueError(f"end_date {str_end_date} INVALID")
    end_date_adjust = end_date_si(ticker=ticker, end_date=end_date, interval=interval)
    df = si.get_data(ticker, end_date=str_end_date, interval=interval)
    print(f"date of latest interval: {end_date_adjust}, orginal df:\n{df.tail(5)}")
    if df.index[-1] == end_date_adjust:
        print(f"latest interval {interval} is OK")
        return df
    
    df = df.loc[:end_date_adjust]
    print(f"df with {end_date_adjust}+ removed:\n{df.tail(3)}")
    return df

          
def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    
def load_multioutput_data(ticker, str_end_date, interval, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                          test_size=0.2, list_columns=["open", "high", "low", "close", "adjclose", "volume"],
                          predict_cols=['adjclose', 'high', 'low']):
    
    if isinstance(ticker, str):
        df = load_data(ticker, str_end_date=str_end_date, interval=interval)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    
    # remove NA lines
    df.dropna(inplace=True)
    print(f"adjusted df:\n{df.tail(3)}")
    #input()

    result = {}
    result['df'] = df.copy()

    for col in list_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    if "date" not in df.columns:
        df["date"] = df.index

    if scale:
        column_scaler = {}
        for column in list_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler
    
    # add the target column (label) by shifting by `lookup_step`
    for predict_col in predict_cols:
        df[f'future_{predict_col}'] = df[predict_col].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column, get them before dropping NaNs
    last_sequence = np.array(df[list_columns].tail(lookup_step))

    df.dropna(inplace=True) # drop NaNs again due to future column
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    future_columns = [f'future_{predict_col}'for predict_col in predict_cols]
    for entry, target in zip(df[list_columns + ["date"]].values, df[future_columns].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    last_sequence = list([s[:len(list_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence

    # construct the X's and y's
    X, y = [], {f'y_{i}': [] for i in range(len(predict_cols))}
    for seq, target in sequence_data:
        X.append(seq)
        for i, predict_col in enumerate(predict_cols):
            y[f'y_{i}'].append(target[i])

    # Convert to numpy arrays
    X = np.array(X)
    for i in range(len(predict_cols)):
        y[f'y_{i}'] = np.array(y[f'y_{i}'])
    
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["X_test"]  = X[train_samples:]
        for i in range(len(predict_cols)):
            result[f"y_train_{i}"] = y[f'y_{i}'][:train_samples]
            result[f"y_test_{i}"] = y[f'y_{i}'][train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            for i in range(len(predict_cols)):
                shuffle_in_unison(result["X_train"], result[f"y_train_{i}"])
                shuffle_in_unison(result["X_test"], result[f"y_test_{i}"])
    else:
        for i in range(len(predict_cols)):
            if i == 0:
                result["X_train"], result["X_test"], result[f"y_train_{i}"], result[f"y_test_{i}"] = \
                    train_test_split(X, y[f'y_{i}'], test_size=test_size, shuffle=shuffle, random_state=42)
            else:
                _, _, result[f"y_train_{i}"], result[f"y_test_{i}"] = \
                    train_test_split(X, y[f'y_{i}'], test_size=test_size, shuffle=shuffle, random_state=42)
        
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]

    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(list_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(list_columns)].astype(np.float32)

    return result


def create_multioutput_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                             loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False, predict_cols=['adjclose', 'high', 'low']):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        if abs(dropout) > sys.float_info.epsilon:        
            model.add(Dropout(dropout))

    # 输出层修改
    model.add(Dense(len(predict_cols), activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model
