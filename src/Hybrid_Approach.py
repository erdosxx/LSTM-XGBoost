import scipy.stats as st
import tensorflow as tf
import numpy as np
import yfinance as yf
from src.model.module import *

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

stock_prices = yf.download("AAPL")
SPY = yf.download("SPY", start="2001-11-30")["Close"]

close = stock_prices["Close"]

stock_prices.drop("Close", 1, inplace=True)
stock_prices["Close"] = close


stock_prices.head(20)

rets = np.log(stock_prices["Close"] / stock_prices["Close"].shift(1))
vol = np.array(stock_prices["2019-01-01":"2022-01-27"]["Volume"])


stock_prices.describe()


tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "SPY"]


log_rets = sp500_log_rets(tickers)


# Annual rets for the selected tickers
ann_rets = {}
for ticker in log_rets.keys():
    ann_rets[ticker] = (
        str(round(annualized_rets(log_rets[ticker]) * 100, 2)) + "%"
    )
ann_rets


fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_alpha(0.2)
for i in log_rets.keys():
    if i == "AAPL":
        mu = np.mean(log_rets["AAPL"])
        sigma = np.std(log_rets["AAPL"])
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)
        pdf = st.norm.pdf(x, mu, sigma)
        ax[1].plot(x, pdf, lw=2, color="black")
        ax[1].hist(log_rets[i], bins=40, color="red")
        ax[0].boxplot(stock_prices["Close"])
        ax[0].set_title("Box plot log returns")
        ax[1].set_title("Log returns ditribution")
    else:
        ax[1].hist(log_rets[i], bins=40, color="grey", alpha=0.3)
        ax[0].set_title("Box plot log returns")
        ax[1].set_title("Log returns ditribution")


fig, ax = plt.subplots(figsize=(15, 5))
fig.patch.set_alpha(0.2)
for i in log_rets.keys():
    if i == "AAPL":
        ax.plot((1 + log_rets[i]).cumprod(), color="red")

    else:
        ax.plot((1 + log_rets[i]).cumprod(), color="grey", alpha=0.3)

ax.legend(tickers)
ax.set_title("AAPL returns over time compared to other tech stocks")
# ax.text(18970, 4, ann_rets["AAPL"], size=9, color='red')

plt.show()


# # Simple Regressions

PERCENTAGE = 0.995
WINDOW = 2
PREDICTION_SCOPE = 0


stock_prices = feature_engineering(stock_prices, SPY)


train_reg, test_reg = train_test_split(stock_prices, WINDOW)
train_split_reg, validation_split_reg = train_validation_split(
    train_reg, PERCENTAGE
)


print(train_reg.shape)
print(test_reg.shape)


train_set_reg = np.array(train_split_reg)
validation_set_reg = np.array(validation_split_reg)


X_train_reg, y_train_reg, X_val_reg, y_val_reg = windowing(
    train_set_reg, validation_set_reg, WINDOW, PREDICTION_SCOPE
)


# Reshaping the Data

X_train_reg = np.array(X_train_reg)
y_train_reg = np.array(y_train_reg)


X_val_reg = np.array(X_val_reg)
y_val_reg = np.array(y_val_reg)


X_train_reg = X_train_reg.reshape(X_train_reg.shape[0], -1)
X_val_reg = X_val_reg.reshape(X_val_reg.shape[0], -1)


print(y_train_reg.shape)
print(X_train_reg.shape)
print(X_val_reg.shape)
print(y_val_reg.shape)


X_test_reg = np.array(test_reg.iloc[:, :-1])
y_test_reg = np.array(test_reg.iloc[:, -1])

print(X_test_reg.shape)


X_test_reg = X_test_reg.reshape(1, -1)

print(X_test_reg.shape)


# ### Linear Regression

lr = LinearRegression()

lr.fit(X_train_reg, y_train_reg)

y_hat_lr = lr.predict(X_val_reg)

mae_lr = mean_absolute_error(y_val_reg, y_hat_lr)

print("MSE: {}".format(np.mean((y_hat_lr - y_val_reg) ** 2)))
print("MAE: {}".format(mae_lr))


# ### Random Forest Regressor

rf = RandomForestRegressor()

rf.fit(X_train_reg, y_train_reg)

y_hat_rf = rf.predict(X_val_reg)

mae_rf = mean_absolute_error(y_val_reg, y_hat_rf)

print("MSE: {}".format(np.mean((y_hat_rf - y_val_reg) ** 2)))
print("MAE: {}".format(mae_rf))


y_hat_rf = np.ravel(y_hat_rf)
y_hat_lr = np.ravel(y_hat_lr)
y_val_reg = np.ravel(y_val_reg)


fig, ax = plt.subplots(figsize=(15, 8))

ax.plot(y_val_reg, color="red")
ax.plot(y_hat_lr, color="orange")
ax.plot(y_hat_rf, color="grey", alpha=0.2)

ax.legend(["True Returns", "Linear Regression", "Random Forest"])


pred_test_lr = lr.predict(X_test_reg)


plotting(y_val_reg, y_test_reg, pred_test_lr, mae_lr, WINDOW, PREDICTION_SCOPE)


# # XGBoost
#
# XGBoost, is one of the most highly used supervised ML algorithms nowadays.
#
# The algorithm uses a more optimized way to implement a tree based algorithm.
#
# The methodology followed by this algorithm is the following.
# XGBoost uses a Greedy algorithm for the building of its tree,
# meaning it uses a simple intuitive way to optimze the algorithm.
# This is done by making a prediction (which acts as a threshols),
# before starting to evaluate the rest of the observations,
# which then turn into other thresholds, and so on.
#
# This methods enables it to manage huge amount of data very quickly.
#
# In the case of regression, it basically build up a Regression Tree
# through the residuals of each data point to the initial prediction.
# Then we split the data into portions and compare one to another and
# see which one is better at splitting the residuals into clusters of
# similar values.
#
# For more insights into how this algorithm works, check out this video
# from [StatQuest](https://www.youtube.com/watch?v=OtD8wVaFm6E&t=649s)
#

# ## Feature Engineering
#
# We will firt make the analysis forecasting 1 period ahead

stock_prices = yf.download("AAPL", start="2001-11-30")


PERCENTAGE = 0.995
WINDOW = 2
PREDICTION_SCOPE = 0


stock_prices = feature_engineering(stock_prices, SPY)


train, test = train_test_split(stock_prices, WINDOW)
train_set, validation_set = train_validation_split(train, PERCENTAGE)

print(f"train_set shape: {train_set.shape}")
print(f"validation_set shape: {validation_set.shape}")
print(f"test shape: {test.shape}")


# Here are some functions that pretend to ease us the work while
# applying the same algorithm on different period forecasts

X_train, y_train, X_val, y_val = windowing(
    train_set, validation_set, WINDOW, PREDICTION_SCOPE
)

# Convert the returned list into arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")


# Reshaping the Data

X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")


mae, xgb_model = xgb_model(X_train, y_train, X_val, y_val, plotting=True)


plt.figure(figsize=(16, 16))
fig, ax = plt.subplots(1, 1, figsize=(26, 17))

plot_importance(xgb_model, ax=ax, height=0.5, max_num_features=10)
ax.set_title("Feature Importance", size=30)
plt.xticks(size=30)
plt.yticks(size=30)
plt.ylabel("Feature", size=30)
plt.xlabel("F-Score", size=30)
plt.show()


# ## Add the predictions (if needed)

# try:
# y_hat_train = np.expand_dims(xgb_model.predict(X_train), 1)
# array = np.empty((stock_prices.shape[0]-y_hat_train.shape[0], 1))
# array[:] = np.nan
# predictions = np.concatenate((array, y_hat_train))
# except NameError:
# print("No Model")


# new_stock_prices = feature_engineering(stock_prices, SPY, predictions=predictions)


# train, test = train_test_split(new_stock_prices, WINDOW)

# train_set, validation_set = train_validation_split(train, PERCENTAGE)
# X_train, y_train, X_val, y_val = windowing(train_set, validation_set, WINDOW, PREDICTION_SCOPE)

# Reshaping the data
# X_train = np.array(X_train)
# y_train = np.array(y_train)

# X_val = np.array(X_val)
# y_val = np.array(y_val)

# X_train = X_train.reshape(X_train.shape[0], -1)
# X_val = X_val.reshape(X_val.shape[0], -1)


# new_mae, new_xgb_model = xgb_model(X_train, y_train, X_val, y_val, plotting=True)

# print(new_mae)


# ## Evaluation on the Test Set

X_test = np.array(test.iloc[:, :-1])
y_test = np.array(test.iloc[:, -1])
X_test = X_test.reshape(1, -1)

print(f"X_test shape: {X_test.shape}")


# Apply the xgboost model on the Test Data

pred_test_xgb = xgb_model.predict(X_test)


plotting(y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE)


# ## Saving the XGBoost parameters

joblib.dump(xgb_model, "XGBoost.pkl")


# ## MULTIPLE EVALUATIONS

plots = {}


for window in [1, 2, 3, 4, 5, 6, 7, 10, 20, 25, 30, 35]:

    for percentage in [0.92, 0.95, 0.97, 0.98, 0.99, 0.995]:

        WINDOW = window
        pred_scope = 0
        PREDICTION_SCOPE = pred_scope
        PERCENTAGE = percentage

        train = stock_prices.iloc[: int(len(stock_prices)) - WINDOW]
        test = stock_prices.iloc[-WINDOW:]

        train_set, validation_set = train_validation_split(train, PERCENTAGE)

        X_train, y_train, X_val, y_val = windowing(
            train_set, validation_set, WINDOW, PREDICTION_SCOPE
        )

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_val = np.array(X_val)
        y_val = np.array(y_val)

        X_test = np.array(test.iloc[:, :-1])
        y_test = np.array(test.iloc[:, -1])

        X_train = X_train.reshape(X_train.shape[0], -1)
        try:
            X_val = X_val.reshape(X_val.shape[0], -1)
            X_test = X_test.reshape(1, -1)
        except ValueError:
            break

        xgb_model = xgb.XGBRegressor(gamma=1)
        xgb_model.fit(X_train, y_train)

        pred_val = xgb_model.predict(X_val)

        mae = mean_absolute_error(y_val, pred_val)

        pred_test = xgb_model.predict(X_test)
        plotii = [y_test[-1], pred_test]

        plots[str(window) + str(pred_scope)] = [
            y_val,
            y_test,
            pred_test,
            mae,
            WINDOW,
            PREDICTION_SCOPE,
            PERCENTAGE,
        ]


print()
print(plots["20"])
print(plots["10"])


window_optimization(plots)


for key in list(plots.keys())[5:9]:
    plotting(
        plots[key][0],
        plots[key][1],
        plots[key][2],
        plots[key][3],
        plots[key][4],
        plots[key][5],
    )


# # LSTM
#
#
# Long Short Term Memory or LSTM is a type of Recurrent Neural Network,
# which instead of only processing the information they receive from the
# previous neuron and apply the activation function from scratch, they actually
# divide the neuron into three main parts from which to set up the input from
# the next layer of neurons: Learn, Unlearn and Retain gate.
#
# The idea behind this method is to ensure that you are using the information
# given from previous data and the data returned from a neural that is in the
# same layer, to get the input for the next nuron.
#
# This is specially usefull, when you are relying on the temporal distribution
# of the data, i.e. text, time series mainly.
#
# In this work we will see how the LSTM is used for predicting the next period
# from Apple stock. Through hyperparameter tuning there was a need to define,
# similar to normal RNN, the input and hidden layer size, the batch_size,
# number of epochs and the rolling window size for the analysis.
#
# The data ranging from 2001 till now, gained from the Yahoo Finance API,
# got splitted into a train, validation and test set to see how the model
# performed on different distributions. After that, the test set was settled
# to be the last width of the input data in order to predict the next period.
#
# The parameters are showed below.

# Parameters for the LSTM
# Split train/val and test set
PERCENTAGE = 0.98
# Used to stop training the Network when the MAE from
# the validation set reached a performance below 3.1%
CALLBACK = 0.031
# Number of samples that will be propagated through the network.
# I chose almost a trading month
BATCH_SIZE = 20
# Settled to train the model
EPOCH = 50
# The window used for the input data
WINDOW_LSTM = 30
# How many period to predict, being 0=1
PREDICTION_SCOPE = 0


train_lstm, test_lstm = train_test_split(stock_prices, WINDOW_LSTM)
train_split_lstm, validation_split_lstm = train_validation_split(
    train_lstm, PERCENTAGE
)

train_split_lstm = np.array(train_split_lstm)
validation_split_lstm = np.array(validation_split_lstm)


# ## Rescaling to train the LSTM

scaler = MinMaxScaler()
scaler.fit(train_split_lstm)

train_scale_lstm = scaler.transform(train_split_lstm)
val_scale_lstm = scaler.transform(validation_split_lstm)
test_scale_lstm = scaler.transform(test_lstm)

print(train_scale_lstm.shape)
print(val_scale_lstm.shape)
print(test_scale_lstm.shape)


X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm = windowing(
    train_scale_lstm, val_scale_lstm, WINDOW_LSTM, PREDICTION_SCOPE
)


X_train_lstm = np.array(X_train_lstm)
y_train_lstm = np.array(y_train_lstm)

X_val_lstm = np.array(X_val_lstm)
y_val_lstm = np.array(y_val_lstm)

X_test_lstm = np.array(test_scale_lstm[:, :-1])
y_test_lstm = np.array(test_scale_lstm[:, -1])

print(X_train_lstm.shape)
print(X_val_lstm.shape)
print(X_test_lstm.shape)


model_lstm = lstm_model(
    X_train_lstm,
    y_train_lstm,
    X_val_lstm,
    y_val_lstm,
    EPOCH,
    BATCH_SIZE,
    CALLBACK,
    plotting=True,
)


# Set up predictions for train and validation set
y_hat_lstm = model_lstm.predict(X_val_lstm)
y_hat_train_lstm = model_lstm.predict(X_train_lstm)

# Validation Transormation
mae_lstm = mean_absolute_error(y_hat_lstm, y_hat_lstm)
real_val, pred_val = inverse_transformation(X_val_lstm, y_val_lstm, y_hat_lstm)
mae_lstm = mean_absolute_error(real_val.iloc[:, 49], pred_val.iloc[:, 49])


plt.figure(figsize=(15, 6))

plt.plot(real_val.iloc[:, 49])
plt.plot(pred_val.iloc[:, 49])

plt.title(f"MAE for this period: {round(mae_lstm, 2)}")


real_train, pred_train = inverse_transformation(
    X_train_lstm, y_train_lstm, y_hat_train_lstm
)


plt.figure(figsize=(18, 6))

plt.plot(real_train.iloc[4000:, 49])
plt.plot(pred_train.iloc[4000:, 49])


# ## Prediction

X_test_formula = X_test_lstm.reshape(
    X_test_lstm.shape[0], 1, X_test_lstm.shape[1]
)


X_test_formula.shape


X_test_lstm = X_test_formula.reshape(
    1, X_test_formula.shape[0], X_test_formula.shape[2]
)


X_test_lstm.shape


y_hat_test_lstm = model_lstm.predict(X_test_lstm)


real_test, pred_test = inverse_transformation(
    X_test_lstm, y_test_lstm, y_hat_test_lstm
)


y_val_lstm = np.array(real_val.iloc[-30:, 49])
y_test_lstm = np.array(real_test.iloc[:, 49])
pred_test = np.array(pred_test.iloc[-1:, 49])
mae_lstm = mean_absolute_error(real_val.iloc[:, 49], pred_val.iloc[:, 49])


plotting(
    y_val_lstm, y_test_lstm, pred_test, mae_lstm, WINDOW_LSTM, PREDICTION_SCOPE
)


# ## Saving the Model

# model.save('./SVM')


# lstm_model = tf.keras.models.load_model("SVM")


# # COMBINATION XGBoost-LSTM
#
#
# In order to get the most out of the two models, good practice is to combine
# those two and apply a higher weight on the model which got a lower loss
# function (mean absolute error).
#
# In our case we saw that the MAE of the XGBoost was lower than the one
# from the LSTM, therefore we will gave a higher weight on the predictions
# returned from the XGBoost model.

# In[131]:


mae_xgboost = mae


xgboost_model = joblib.load("XGBoost.pkl")


pred_test


pred_test_xgb


scope = predictions(mae_lstm, mae_xgboost, pred_test_xgb, pred_test)


avg_mae = (mae_lstm + mae_xgboost) / 2


plotting(y_val, y_test, scope, avg_mae, WINDOW, PREDICTION_SCOPE)
