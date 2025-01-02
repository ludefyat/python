# Basic Purpose:
There are stock prediction codes available online, but they can only learn and predict a single indicator, such as "close". If you want to predict "high", you have to modify the configuration and retrain, which often leads to basic errors like predicting a closing price higher than the highest price. This project addresses this issue: during each training session, multiple indicators such as ["open", "high", "low", "close", "adjclose", "volume"] are imported for learning, and predictions are made on specific indicators. This minimizes basic errors like a closing price being higher than the highest price.

Additionally, K-line periods have been expanded to include monthly and weekly data.

To compare predicted data with actual data, statistical functions have also been added.

# Environment Requirements:
A Python environment that supports TensorFlow is required. Other Python libraries should be installed based on the source code, such as the Yahoo stock API. This project has been debugged on macOS.

# Program Guide:
This project consists of the following files:  
1) `parameters.py` - Parameter configuration; cannot be run independently.  
2) `load_data.py` - Loads training data; cannot be run independently.  
3) `run_train.py` - Runs the machine learning process; can be run independently.  
4) `sum_excel.py` - Generates Excel reports with statistical data to compare predictions with actual results.  
