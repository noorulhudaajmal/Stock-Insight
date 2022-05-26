import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import tensorflow
from sklearn.preprocessing import StandardScaler
import streamlit as st
import keras
from pylab import rcParams
from plotly.offline import download_plotlyjs , iplot , init_notebook_mode , plot
import cufflinks as cf
import plotly.express as px
# init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objects as go



st.set_page_config(page_title= "Stock Insight" , layout="wide" )
st.title("Stock Insight")
user_in = st.text_input("Enter the company name : " , "MEBL")
stock = pd.read_csv("meb.csv"); #importing dataset
col1 , col2 = st.columns(2)

st.subheader(user_in , " STOCK FROM 2018 - 2022")
with col1:
    st.subheader("STOCK SUMMARY")
    st.write(stock)

with col2:
    userIn = st.text_input("Choose the variable : " , "Close")
    fg = go.Figure()
    fg.add_trace(go.Scatter(
        x = stock.index.values,
        y = stock["Open"],
        line=dict(color='gray') , name = 'Open Stock Price'
    ))

    fg.add_trace(go.Scatter(
        x = stock.index.values,
        y = stock[userIn],
        line=dict(color='orange') , name = userIn + 'Stock Price'
    ))
    st.plotly_chart(fg)


# st.markdown("<hr/>",unsafe_allow_html=True)


# stock = pd.read_csv("meb.csv"); #importing dataset

dates = list(stock['Date']) #getting dates off the dataframe (stock data)
dates = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates] #converting dates to timestamps

## creating dataset of chosen features to work ahead with LSTM-model
features = list(stock)[1:-1]
data = stock[features]
ft = data.columns.values
# st.write("The features selected to train the model are : \n [ " , ft[0] , ft[1] , ft[2] , ft[3] , " ]")

##DATA PRE-PROCESSING:
data = data.astype(float)
dataset = data.values

#feature scaling:
##for input data
sc = StandardScaler()
sc_dataset = sc.fit_transform(dataset)
#for output data
pred_sc = StandardScaler()
pred_sc.fit(dataset[:, 0:1 ])

# INPUT-OUTPUT SPLIT FOR TIME SERIES ANALYSIS
Xtrain = [] #trend to be analyzed
ytrain = [] #output for the given-trend
nFuture = 1 #60 #7 #30   # Number of days we want top predict into the future
nPast = 14 #90 #30 #100     # Number of past days we want to use to predict the future

rows = data.shape[0]
cols = data.shape[1]
for i in range(nPast ,  rows - nFuture +1):
    Xtrain.append(sc_dataset[i - nPast : i , 0:cols])
    ytrain.append(sc_dataset[i + nFuture - 1 : i + nFuture , 0])
Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
#Now the Xtrain contains all of the given dataset ,
# where as ytrain contains value of volumes to be predicted at gievn past and future figures

model = keras.models.load_model("model0.h5")
#OUTPUT
#optimizers and loss declaration
#compiling model
# FUTURE FORECAST:
nFut = 90
#GENERATING DATE SEQUENCES FOR FUTURE
futureDates = pd.date_range(dates[-1], periods=nFut, freq='1d').tolist()
futureDatesList = []
for i in futureDates:
    futureDatesList.append(i.date())
def to_Timestamp(x):
    return dt.datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')
#MODEL PREDICTIONS
futurePreds = model.predict(Xtrain[-nFut:])
trainPreds = model.predict(Xtrain[nPast:])
#INVERSE SCALING
y_predFuture = pred_sc.inverse_transform(futurePreds)
y_predTrain = pred_sc.inverse_transform(trainPreds)
#ARRANGING STUFF
FUT_PREDS = pd.DataFrame(y_predFuture, columns=["Open"]).set_index(pd.Series(futureDatesList))
TRAIN_PREDS = pd.DataFrame(y_predTrain, columns=["Open"]).set_index(pd.Series(dates[2 * nPast + nFuture - 1:]))
TRAIN_PREDS.index = TRAIN_PREDS.index.to_series().apply(to_Timestamp)
trainSet = pd.DataFrame(data, columns=features)
trainSet.index = dates
trainSet.index = pd.to_datetime(trainSet.index)

#FORECAST:
st.subheader("STOCK FORECAST")
st.subheader("Chart View")
st.write(FUT_PREDS)
st.subheader("Visuals")
fig0 = px.line(x=futureDatesList, y=FUT_PREDS["Open"] ,
              labels={"x" : "Date" , "y":"Open"} ,
              height = 500 ,width=900)
fig0.update_layout(
    title="FUTUTRE FORECAST OF 3 MONTHS",
    xaxis_title="Time",
    yaxis_title="OPEN PRICES",
    legend_title="Legend Title",
)
st.plotly_chart(fig0)
st.subheader("FORECASTING MODEL SUMMARY")
st.subheader("Plot 1:")
#PLOTTING actual vs. predicted
# Plotting
STARTDATE = TRAIN_PREDS.index[0]
# import plotly.graph_objects as go
fg = go.Figure()
fg.add_trace(go.Scatter(
    x = trainSet.loc[STARTDATE:].index,
    y = trainSet.loc[STARTDATE:]["Open"],
    line=dict(color='blue') , name = 'Actual Stock Price'
))

fg.add_trace(go.Scatter(
    x = FUT_PREDS.index,
    y = FUT_PREDS["Open"],
    line=dict(color='red') , name = 'Future Predicted Price'
))

fg.add_trace(go.Scatter(
    x = TRAIN_PREDS.loc[STARTDATE:].index,
    y = TRAIN_PREDS.loc[STARTDATE:]["Open"],
    line=dict(color='orange'), name = 'Predicted Train Prices'
))
fg.add_vline(x=min(FUT_PREDS.index), line_width=1.5, line_dash="dash", line_color="green")

fg.update_layout(
    title="Pred vs. Actual",
    xaxis_title="Time",
    yaxis_title="Stock Open Prices",
    height = 500 ,
    width = 900
)
st.plotly_chart(fg)