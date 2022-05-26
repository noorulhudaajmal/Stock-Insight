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
import datetime
# init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objects as go


st.set_page_config(page_title= "Stock Insight" , layout="wide" )

title_temp = """
<div style="background-color:{};padding:10px;border-radius:10px">
<h1 style="color:{};text-align:center;">{}</h1>
</div>
"""
sub_title_temp = """
<div style="background-color:{};padding:0.5px;border-radius:5px;">
<h4 style="color:{};text-align:center;">{}</h6>
</div>
"""

st.markdown(title_temp.format('#1E3231','white' , "STOCK INSIGHT"),unsafe_allow_html=True)
st.write("")
st.write("")

user_in = st.text_input("Enter the company name : " , "MEBL")
stock = pd.read_csv("meb.csv"); #importing dataset

st.markdown(sub_title_temp.format("#89B6A5" , "white" , user_in+" STOCK FROM 2018 - 2022"),unsafe_allow_html=True)
# st.subheader(user_in , " STOCK FROM 2018 - 2022")
stock_wd_date = stock.set_index("Date")
col1 , col2 = st.columns((1,1.5))
with col1:
    fig = go.Figure(
        data = [go.Table (columnorder = [0,1,2,3,4,5], columnwidth = [15,10,10,10,10,10],
                          header = dict(
                              values = list(stock.columns),
                              font=dict(size=12, color = 'white'),
                              fill_color = '#264653',
                              line_color = 'rgba(255,255,255,0.2)',
                              align = ['left','center'],
                              #text wrapping
                              height=40
                          )
                          , cells = dict(
                values = [stock[K].tolist() for K in stock.columns],
                font=dict(size=12),
                align = ['left','center'],
                line_color = 'rgba(255,255,255,0.2)',
                height=30))])
    fig.update_layout(title_text="HISTORICAL DATA",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=480)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fg = px.line(x=stock["Date"], y=stock["Open"] ,
                 labels={"x" : "Date" , "y":"Open"})
    fg.update_layout(
        title = "Quick Summary",
        xaxis_title="Time",
        yaxis_title="Stock Open Prices",
        width = 750,
        height = 500
    )
    st.plotly_chart(fg)
    # st.subheader("Quick Summary")
    # st.line_chart(stock["Open"] , use_container_width=True , height= 300 , width= 600 ,)
cal1, em1, emp2, emp3  = st.columns((2,1,1,1))
d = "2018-01-02";
with cal1:
    ch_d = st.date_input(" Choose Day:",datetime.date(2018, 1, 1))
    d = str(ch_d)

m1, m2, m3, m4, m5,m6 = st.columns((1,1,1,1,1.7,1.7))
avb_days = stock_wd_date.index.to_list()
pi = avb_days.index(d) - 1
if(d in stock_wd_date.index.to_list()):
    [o,h,l,c,v] = stock_wd_date.loc[d].tolist()
    [po,ph,pl,pc,pv] = stock_wd_date.iloc[pi].tolist()
else:
    [o,h,l,c,v] = stock_wd_date.loc["2018-01-02"].tolist()
    [po,ph,pl,pc,pv] = stock_wd_date.iloc[1].tolist()
change  = ((c - pc) / pc ) * (100)
ch = str(change.__round__(1)) + "%"

with m1 :
    s = "Open  " + str(o)
    new_title = '<p style="background : #3C6997; height : 100% ;padding : 3px; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">'+s+'</p>'
    st.markdown(new_title, unsafe_allow_html=True)
with m2 :
    s = "High  " + str(h)
    new_title = '<p style="background : #3C6997; height : 100%;padding : 3px ; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">'+s+'</p>'
    st.markdown(new_title, unsafe_allow_html=True)
with m3 :
    s = "Low " + str(l)
    new_title = '<p style="background : #3C6997; height : 100% ;padding : 3px; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">'+s+'</p>'
    st.markdown(new_title, unsafe_allow_html=True)
with m4 :
    s = "Close  " + str(c)
    new_title = '<p style="background : #3C6997; height : 100% ;padding : 2px; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">'+s+'</p>'
    st.markdown(new_title, unsafe_allow_html=True)
with m5 :
    s = "Volume  " + str(v)
    new_title = '<p style="background : #5B8C5A; height : 100%;padding : 1px ; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">'+s+'</p>'
    st.markdown(new_title, unsafe_allow_html=True)
with m6 :
    s = "Change  " + str(ch)
    new_title = '<p style="background : #92AC86; height : 100% ;padding : 1px; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">'+s+'</p>'
    st.markdown(new_title, unsafe_allow_html=True)


st.markdown(sub_title_temp.format("#89B6A5" , "white" , "OPEN RELATIONSHIP"),unsafe_allow_html=True)
# st.subheader("OPEN RELATIONSHIP")
variables = stock.columns[1:]
choice = st.selectbox('With :', variables, help = 'Filter stock to show relationship with one variable.')
plot1 , plot2 = st.columns((1.5,1))
with plot1:
    fg = go.Figure()
    fg.add_trace(go.Scatter(
        x = stock.index.values,
        y = stock["Open"],
        line=dict(color='gray') , name = 'Open Stock Price'
    ))

    fg.add_trace(go.Scatter(
        x = stock.index.values,
        y = stock[choice],
        line=dict(color='orange') , name =  choice + 'Stock Price'
    ))
    fg.update_layout(
        xaxis_title = "Time", yaxis_title = "Value",
        title = "Trends over time",
        width = 700, height = 500
    )
    st.plotly_chart(fg)
with plot2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    fg = px.scatter(stock , x = choice , y = "Open")
    fg.update_layout(
        title = "Open vs."+choice,
        width = 500, height = 400
    )
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
#  ***********************************************************************************
#FORECAST:
st.markdown(sub_title_temp.format("#89B6A5" , "white" , "STOCK FORECAST"),unsafe_allow_html=True)
# st.subheader("STOCK FORECAST")
nFut = 90
opt1 , e1 , e2 = st.columns((1,2,2))
with opt1:
    nDs = st.text_input("Enter the days range to forecast (1-90)" , 90);
    nFut = int(nDs)
# FUTURE FORECAST:
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

# ****************************************************************************

chart , visual = st.columns((1,1.5))
with chart:
    fut_preds = FUT_PREDS.reset_index()
    fig = go.Figure(
        data = [go.Table (columnorder = [0,1], columnwidth = [15,10],
                          header = dict(
                              values = ["Date" , "Open"],
                              font=dict(size=12, color = 'white'),
                              fill_color = '#264653',
                              line_color = 'rgba(255,255,255,0.2)',
                              align = ['left','center'],
                              #text wrapping
                              height=40
                          )
                          , cells = dict(
                values = [fut_preds[K].tolist() for K in fut_preds.columns],
                font=dict(size=12),
                align = ['left','center'],
                line_color = 'rgba(255,255,255,0.2)',
                height=30))])
    fig.update_layout(title_text="CHART VIEW",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=480)
    st.plotly_chart(fig, use_container_width=True)
with visual:
    fig0 = px.line(x=futureDatesList, y=FUT_PREDS["Open"] ,
                   labels={"x" : "Date" , "y":"Open"} ,
                   height = 500 ,width=750)
    fig0.update_layout(
        title="VISUALS",
        xaxis_title="Time",
        yaxis_title="OPEN PRICES",
        legend_title="Legend Title",
    )
    st.plotly_chart(fig0)

st.markdown(sub_title_temp.format("#89B6A5" , "white" , "MODEL SUMMARY"),unsafe_allow_html=True)
# st.subheader("FORECASTING MODEL SUMMARY")
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
    width = 1200
)
st.plotly_chart(fg)

# VOLUME FORECAST

st.markdown(sub_title_temp.format("#89B6A5" , "white" , "VOLUME FORECAST"),unsafe_allow_html=True)


# stockV = pd.read_csv("meb.csv")
stockv = pd.read_csv("meb.csv")

model1 = keras.models.load_model("model1.h5")
datesForV = list(stockv['Date'])
datesForV = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in datesForV]


featuresForV = list(stockv)[1:]
dataForV = stock[featuresForV]
dataForV = dataForV.astype(float)
datasetForV = dataForV.values

print(dataForV.columns)

scForV = StandardScaler()
sc_datasetForV = scForV.fit_transform(datasetForV)
pred_scForV = StandardScaler()
pred_scForV.fit_transform(datasetForV[:, -1: ]) #for output values

XtrainForV = [] #trend to be analyzed
ytrainForV = []

nFutureForV = 1 #60 #7 #30   # Number of days we want top predict into the future
nPastForV = 30

rowsForV = dataForV.shape[0]
colsForV = dataForV.shape[1]

rowsForV = datasetForV.shape[0]
colsForV = datasetForV.shape[1]
for i in range(nPastForV ,  rowsForV - nFutureForV +1):
    XtrainForV.append(sc_datasetForV[i - nPastForV : i , 0:colsForV])
    ytrainForV.append(sc_datasetForV[i + nFutureForV - 1 : i + nFutureForV , -1])

XtrainForV = np.array(XtrainForV)
ytrainForV = np.array(ytrainForV)

nFutForV = 30

opt11 , e11 , e22 = st.columns((1,2,2))
with opt11:
    nDsV = st.text_input("Enter the days range to forecast (1-90)" , 30);
    nFutForV = int(nDsV)


futureDatesForV = pd.date_range(datesForV[-1], periods=nFutForV, freq='1d').tolist()
futureDatesListForV = []
for i in futureDatesForV:
    futureDatesListForV.append(i.date())
futurePredsForV = model1.predict(XtrainForV[-nFutForV:])
trainPredsForV = model1.predict(XtrainForV[nPastForV:])

y_predFutureForV = pred_scForV.inverse_transform(futurePredsForV)
y_predTrainForV = pred_scForV.inverse_transform(trainPredsForV)

FUT_PREDSforV = pd.DataFrame(y_predFutureForV, columns=["Volume"]).set_index(pd.Series(futureDatesListForV))
TRAIN_PREDSforV = pd.DataFrame(y_predTrainForV, columns=["Volume"]).set_index(pd.Series(datesForV[2 * nPastForV + nFutureForV - 1:]))

TRAIN_PREDSforV.index = TRAIN_PREDSforV.index.to_series().apply(to_Timestamp)

trainSetForV = pd.DataFrame(dataForV, columns=featuresForV)
trainSetForV.index = datesForV
trainSetForV.index = pd.to_datetime(trainSetForV.index)


chartV , visualV = st.columns((1,1.5))
with chartV:
    fut_preds = FUT_PREDSforV.reset_index()
    fig = go.Figure(
        data = [go.Table (columnorder = [0,1], columnwidth = [15,10],
                          header = dict(
                              values = ["Date" , "Volume"],
                              font=dict(size=12, color = 'white'),
                              fill_color = '#264653',
                              line_color = 'rgba(255,255,255,0.2)',
                              align = ['left','center'],
                              #text wrapping
                              height=40
                          )
                          , cells = dict(
                values = [fut_preds[K].tolist() for K in fut_preds.columns],
                font=dict(size=12),
                align = ['left','center'],
                line_color = 'rgba(255,255,255,0.2)',
                height=30))])
    fig.update_layout(title_text="CHART VIEW",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=480)
    st.plotly_chart(fig, use_container_width=True)
with visualV:
    fig0 = px.line(x=futureDatesListForV, y=FUT_PREDSforV["Volume"] ,
                   labels={"x" : "Date" , "y":"Volume"} ,
                   height = 500 ,width=750)
    fig0.update_layout(
        title="VISUALS",
        xaxis_title="Time",
        yaxis_title="Volume",
        legend_title="Legend Title",
    )
    st.plotly_chart(fig0)




st.markdown(sub_title_temp.format("#89B6A5" , "white" , "MODEL SUMMARY"),unsafe_allow_html=True)
# st.subheader("FORECASTING MODEL SUMMARY")
#PLOTTING actual vs. predicted
# Plotting
STARTDATE = TRAIN_PREDSforV.index[0]
# import plotly.graph_objects as go
fg0 = go.Figure()
fg0.add_trace(go.Scatter(
    x = trainSetForV.loc[STARTDATE:].index,
    y = trainSetForV.loc[STARTDATE:]["Volume"],
    line=dict(color='blue') , name = 'Actual Volume'
))

fg0.add_trace(go.Scatter(
    x = FUT_PREDSforV.index,
    y = FUT_PREDSforV["Volume"],
    line=dict(color='red') , name = 'Future Predicted Volume'
))

fg0.add_trace(go.Scatter(
    x = TRAIN_PREDSforV.loc[STARTDATE:].index,
    y = TRAIN_PREDSforV.loc[STARTDATE:]["Volume"],
    line=dict(color='orange'), name = 'Predicted Train Volume'
))
fg0.add_vline(x=min(FUT_PREDSforV.index), line_width=1.5, line_dash="dash", line_color="green")

fg0.update_layout(
    title="Pred vs. Actual",
    xaxis_title="Time",
    yaxis_title="Stock Volume",
    height = 500 ,
    width = 1200
)
st.plotly_chart(fg0)
