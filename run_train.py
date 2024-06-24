import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import subprocess
from load_data import create_multioutput_model, load_multioutput_data
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from parameters import *
from datetime import datetime, timedelta

def find_update_row(csv_file, column = "parameter", new_row = {}):
    detect_only = len(new_row) == 1 and column in new_row
    if not os.path.isfile(csv_file):
        if detect_only:
            return False
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(list(new_row.keys()))
            writer.writerow(list(new_row.values()))
        return True

    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if detect_only:
        cols_str = '\n\n'.join([row.get(column) for row in rows])
        return (new_row.get(column) in cols_str)
    
    for i, row in enumerate(rows):
        if new_row.get(column) in row.get(column):
            rows[i] = new_row
            with open(csv_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(new_row.keys()))
                writer.writeheader()
                writer.writerows(rows)
            break
    else:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(new_row.keys()))
            if os.path.getsize(csv_file) == 0:
                writer.writeheader()
            writer.writerow(new_row)
    return True


# 定义一个自定义的 Callback 类
class CustomEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, custom_parameter):
        super(CustomEpochCallback, self).__init__()
        self.custom_parameter = custom_parameter

    def on_epoch_begin(self, epoch, logs=None):
        print(f"{self.custom_parameter}", end=' ')


def plot_multi_graph(test_df, predict_cols=['close','high','low'], future=[0,0,0], pic_info="no_info", predit_info=[]):
    plt.rcParams['font.sans-serif'] = 'Songti SC'
    width, height = 4*len(predict_cols), 3*len(predict_cols)
    fig, axs = plt.subplots(len(predict_cols), 1, sharex='col', sharey='col', figsize=(width, height))

    x_label = pic_info[(pic_info.find(' (') + 3) : pic_info.find(')')] + 's'
    x_label = x_label.capitalize()

    for i, predict_col in enumerate(predict_cols):
        axs[i].plot(test_df[f'true_{predict_col}_{LOOKUP_STEP}'], c='b')
        axs[i].plot(test_df[f'{predict_col}_{LOOKUP_STEP}'], c='r')
        axs[i].set_xlabel(x_label)
        axs[i].set_ylabel("Price")
        axs[i].set_title(f"{predict_col}: {future[i]:.2f}",loc = 'center')
        axs[i].text(0.985, 0.985, predit_info[f"metrics_{predict_col}"], fontsize=9, ha='right', va='top', transform=axs[i].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    fig.legend(["Actual Price", "Predicted Price"])
    title = pic_info.replace("\n", " ").replace("~","\n")
    plt.text(0.005, 0.995, f"{title}", fontsize=12, ha='left', va='top', transform=fig.transFigure)
    plt.subplots_adjust(hspace=0.3)
    return plt


def get_final_df(model, data, predict_cols = ['close','high','low']):
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_pred = y_pred.T

    y_test = []
    for i, predict_col in enumerate(predict_cols):
        y_test.append(data[f"y_test_{i}"])

    for i, row in enumerate(y_pred):
        if SCALE:
            y_pred[i] = np.squeeze(data["column_scaler"][f"{predict_cols[i]}"].inverse_transform(np.expand_dims(row, axis=0)))
            y_test[i] = np.squeeze(data["column_scaler"][f"{predict_cols[i]}"].inverse_transform(np.expand_dims(y_test[i], axis=0)))
    
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    for i, predict_col in enumerate(predict_cols):
        test_df[f"{predict_col}_{LOOKUP_STEP}"] = y_pred[i]
        test_df[f"true_{predict_col}_{LOOKUP_STEP}"] = y_test[i]
    # sort the dataframe by date
    test_df.sort_index(inplace=True)

    # Calculate buy and sell profits
    for predict_col in predict_cols:
        final_df = test_df
        # add the buy profit column
        final_df[f"buy_profit_{predict_col}"] = list(map(buy_profit,
                                                    final_df[f"{predict_col}"],
                                                    final_df[f"{predict_col}_{LOOKUP_STEP}"],
                                                    final_df[f"true_{predict_col}_{LOOKUP_STEP}"]))
        # add the sell profit column
        final_df[f"sell_profit_{predict_col}"] = list(map(sell_profit,
                                                    final_df[f"{predict_col}"],
                                                    final_df[f"{predict_col}_{LOOKUP_STEP}"],
                                                    final_df[f"true_{predict_col}_{LOOKUP_STEP}"]))
    return final_df


def predict(model, data, predict_cols):
    # Retrieve the last sequence from data
    last_sequence = data["last_sequence"][-ticker_cfg['n_steps']:]
    # Expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # Get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # Get the prices (by inverting the scaling)
    predicted_prices = []
    for i, predict_col in enumerate(predict_cols):
        if SCALE:
            predicted_price = data["column_scaler"][f"{predict_col}"].inverse_transform(prediction[:, i:i+1])[0][0]
        else:
            predicted_price = prediction[0][i]
        predicted_prices.append(predicted_price)
    return predicted_prices


def next_period_str(cur_date, interval):
    if interval == '1wk':
        predit_date = cur_date + timedelta(days=7)
        date_next = predit_date.strftime('%Y-%m-%d')+ " (1week)"
    elif interval == '1mo':
        predit_date = (cur_date + timedelta(days=32)).replace(day=1)
        date_next = predit_date.strftime('%Y-%m-%d')  + " (1month)"
    elif interval == '1d':
        days_dlt = 1 if cur_date.weekday() < 4 else (7 - cur_date.weekday())
        predit_date = cur_date + timedelta(days=days_dlt)
        date_next = predit_date.strftime('%Y-%m-%d')  + " (1day)"
    else:
        raise ValueError(f"INVALID interval {interval}")
    return date_next
    

# create these folders if they does not exist
for key, dir in DIRS_CFG.items():
    if len(dir) and not os.path.isdir(dir):
        os.mkdir(dir)

#prevent going sleep
caffeinate_process = subprocess.Popen(["caffeinate"])
os.system('clear')
fixed_para = f"{shuffle_str}-{scale_str}-{split_by_date_str}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{LOOKUP_STEP}"
fixed_para += "-b" if BIDIRECTIONAL else ""
print(f"{fixed_para}")

for week_dlt in range(0, WEEK_BACKTRACK + 1):
    date_now = datetime.now().date() if week_dlt == 0 else date_now - timedelta(days=7)
    print(f"current date: {date_now}")
    str_date_now = date_now.strftime("%Y-%m-%d")
    for row_cfg, ticker_cfg in enumerate(TICKER_CFG):
        ticker      = ticker_cfg['ticker']
        interval    = ticker_cfg['interval']
        predit_ticker = {}
        model_name = f"{ticker}_{ticker_cfg['n_steps']}-{ticker_cfg['n_layers']}-{ticker_cfg['units']}"
        print(f"{model_name}\n{time.strftime('%Y-%m-%d %H:%M')}\n{ticker}\n{ticker_cfg['name']}...str_date_now: {str_date_now}...{interval_dict[interval]}")

        data = load_multioutput_data(ticker, str_date_now, interval_dict[interval], ticker_cfg['n_steps'], scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                        shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                        list_columns=FEATURE_COLUMNS, predict_cols=PREDICT_COLUMNS)
        date_next = next_period_str(data['df'].index[-1], interval_dict[interval])
        print(f"stock {ticker}, latest in db {data['df'].index[-1]}, interval {interval}, next {date_next}, train data, evaluate model, predict: {PREDICT_COLUMNS}.....")
        param_info = (f"{date_next}\ndrop: {ticker_cfg['dropout']}\nstep: {ticker_cfg['n_steps']}\nlook: {LOOKUP_STEP}\nlayers: {ticker_cfg['n_layers']}\n" + 
                        f"units: {ticker_cfg['units']}\nbat: {ticker_cfg['batch_size']}\nepo: {ticker_cfg['epochs']}")
        predit_ticker["parameter"] = param_info
        mon_future_label = str_date_now[:-3]
        if not os.path.isdir(mon_future_label):
            os.mkdir(mon_future_label)

        chl_str = ''.join([word[0] for word in PREDICT_COLUMNS])
        #filename = os.path.join(mon_future_label, f"{ticker_cfg['name']}_predit_{mon_future_label}_{chl_str}")
        filename = os.path.join(mon_future_label, f"{ticker_cfg['name']}_{ticker}_{mon_future_label}_{chl_str}")
        csv_stock_filename = f"{filename}.csv"
        #check if the record is alreadly exists in result file
        if find_update_row(csv_stock_filename, "parameter", new_row = predit_ticker):
            print(f"parameter found in {csv_stock_filename}, ignore the {interval} predict\n{predit_ticker['parameter']} ")
            continue
        
        # save the dataframe
        if DIRS_CFG['data']:
            ticker_data_filename = os.path.join(DIRS_CFG['data'], f"{ticker}_{str_date_now}.csv")
            data["df"].to_csv(ticker_data_filename) #save to dir of data

        # create the model
        model = create_multioutput_model(ticker_cfg['n_steps'], len(FEATURE_COLUMNS), loss=LOSS, units=ticker_cfg['units'], cell=CELL, n_layers=ticker_cfg['n_layers'],
                            dropout=ticker_cfg['dropout'], optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL, predict_cols=PREDICT_COLUMNS)

        # define callbacks
        epoch_callback = CustomEpochCallback(f"{ticker_cfg['name']} {row_cfg + 1}/{len(TICKER_CFG)} {date_now} {interval} {week_dlt + 1}/{WEEK_BACKTRACK + 1}")
        early_stopping_err  = EarlyStopping(monitor='val_mean_absolute_error', patience=10, verbose=1)
        my_callbacks = [epoch_callback, early_stopping_err]
        if DIRS_CFG["checkpoint"]:
            file_chkpnt = os.path.join(DIRS_CFG["checkpoint"], fixed_para, model_name + ".h5")
            checkpointer = ModelCheckpoint(file_chkpnt, save_weights_only=True, save_best_only=True, verbose=1)
            my_callbacks.append(checkpointer)
        if DIRS_CFG["log"]:
            tensorboard = TensorBoard(log_dir=os.path.join(DIRS_CFG["log"], model_name))
            my_callbacks.append(tensorboard)

        # train the model and save the weights whenever we see 
        history = model.fit(data["X_train"], [data[f"y_train_{i}"] for i in range(len(PREDICT_COLUMNS))],
                            batch_size=ticker_cfg['batch_size'],
                            epochs=ticker_cfg['epochs'],
                            validation_data=(data["X_test"], [data[f"y_test_{i}"] for i in range(len(PREDICT_COLUMNS))]),
                            callbacks=my_callbacks,
                            verbose=1)

        # evaluate the model
        loss, mae = model.evaluate(data["X_test"], [data[f"y_test_{i}"] for i in range(len(PREDICT_COLUMNS))], verbose=0)
        if early_stopping_err.stopped_epoch is not None:
            predit_ticker["parameter"] += f"\nstop_epo: {early_stopping_err.stopped_epoch}"

        final_df = get_final_df(model, data, predict_cols=PREDICT_COLUMNS)
        future_price = predict(model, data, predict_cols=PREDICT_COLUMNS)

        # calculate the mean absolute error (inverse scaling)
        for i, predict_col in enumerate(PREDICT_COLUMNS):
            predit_ticker[predict_col] = round(future_price[i], 2)
            print(f"{ticker} {ticker_cfg['name']} predit in {date_next} {predict_col}: ", predit_ticker[predict_col])
            if SCALE:
                mae = data["column_scaler"][predict_col].inverse_transform([[mae]])[0][0]
            print(f"{ticker} {predict_col}, Mean Absolute Error:, {mae:.2f}")  

            # we calculate the accuracy by counting the number of positive profits
            accuracy_score = (len(final_df[final_df[f"sell_profit_{predict_col}"] > 0]) + \
                            len(final_df[final_df[f"buy_profit_{predict_col}"] > 0])) / len(final_df)
            total_buy_profit  = final_df[f"buy_profit_{predict_col}"].sum()
            total_sell_profit = final_df[f"sell_profit_{predict_col}"].sum()
            total_profit = total_buy_profit + total_sell_profit
            profit_per_trade = total_profit / len(final_df) # dividing total profit by number of testing samples (number of trades)
            predit_ticker[f'metrics_{predict_col}'] = \
                f"Accuracy: {accuracy_score:.3f}\nBuy profit: {total_buy_profit:.3f}\n" + \
                f"Sell profit: {total_sell_profit:.3f}\nProfit: {total_profit:.3f}\n" + \
                f"Profit 1trade: {profit_per_trade:.3f}" 

        graph_info = f"{ticker_cfg['name']}~" + param_info
        plt = plot_multi_graph(final_df, predict_cols=PREDICT_COLUMNS, future=future_price, pic_info=graph_info, predit_info=predit_ticker)
        plt.savefig(f"{filename}.png")
        plt.pause(3)
        plt.close()
        
        #save file per stock
        find_update_row(csv_stock_filename, "parameter", predit_ticker)
        print(f"{row_cfg + 1} of {len(TICKER_CFG)} {str_date_now} finished......")

print(f"all train finished......")
caffeinate_process.terminate()

