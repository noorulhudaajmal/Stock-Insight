from tensorflow.keras.models import load_model
import datetime
import pandas as pd
import streamlit as st
from psx import stocks, tickers
from plots import historical_data_table, quick_summary_plot, stock_trend_overtime, stock_variables_relation_plot, \
    forecast_table, forecast_plot, forecast_model_performance
from forecast.volume.volume import volumeForecast
from forecast.open.open import openForecast


st.set_page_config(page_title="Stock Insight", layout="wide")

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
head_title_temp = """<h6 style="text-align:left;margin-top:2px">{}</h6>"""

st.markdown(title_temp.format('#1E3231', 'white', "STOCK INSIGHT"), unsafe_allow_html=True)
st.write("")
st.write("")

# Initialize session state for stock data
if 'stock' not in st.session_state:
    st.session_state.stock = pd.read_csv("data/meb.csv")
    st.session_state.stock['Date'] = pd.to_datetime(st.session_state.stock['Date'])

def update_stock_data():
    # Get the selected ticker and date range
    ticker = st.session_state.ticker
    start_date = datetime.datetime.combine(st.session_state.start_date, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(st.session_state.end_date, datetime.datetime.max.time())

    # Fetch new stock data
    new_stock_data = stocks(ticker, start=start_date, end=end_date)
    new_stock_data = new_stock_data.reset_index()
    new_stock_data['Date'] = pd.to_datetime(new_stock_data['Date'])

    # Update session state
    st.session_state.stock = new_stock_data

# Create input choices
input_choices = st.columns(4)

with input_choices[0]:
    ticker = st.selectbox(
        label="Choose Ticker",
        options=list(tickers()['symbol']),
        key='ticker'
    )

with input_choices[1]:
    start_date = st.date_input(
        label="Choose Start Date",
        min_value=datetime.date(2015, 1, 1),
        max_value=datetime.date.today(),
        value=datetime.date(2015, 1, 1),
        format="DD-MM-YYYY",
        key='start_date'
    )

with input_choices[2]:
    end_date = st.date_input(
        label="Choose End Date",
        min_value=datetime.date(2015, 12, 31),
        max_value=datetime.date.today(),
        value=datetime.date.today(),
        format="DD-MM-YYYY",
        key='end_date'
    )

with input_choices[3]:
    st.write("###### ")
    st.button(label="Search", on_click=update_stock_data)

stock = st.session_state['stock']

st.markdown(sub_title_temp.format("#646F58", "white", ticker + f" STOCK FROM {stock['Date'].min().year} - {stock['Date'].max().year}"), unsafe_allow_html=True)

stock_wd_date = stock.set_index("Date")
col1, col2 = st.columns((1, 1.5))
with col1:
    fig = historical_data_table(stock)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = quick_summary_plot(stock)
    st.plotly_chart(fig, use_container_width=True)

cal1, em1, emp2, emp3 = st.columns((2, 1, 1, 1))
ch_d = cal1.date_input(" Choose Day:",
                       min_value=stock['Date'].min().date(),
                       max_value=stock['Date'].max().date(),
                       value=stock['Date'].min().date(),
                       format="DD-MM-YYYY")
d = datetime.datetime.combine(ch_d, datetime.datetime.min.time())

try:
    m1, m2, m3, m4, m5, m6 = st.columns((1, 1, 1, 1, 1.7, 1.7))
    avb_days = stock_wd_date.index.to_list()
    pi = avb_days.index(d) - 1
    if d in stock_wd_date.index.to_list():
        [o, h, l, c, v] = stock_wd_date.loc[d].tolist()
        [po, ph, pl, pc, pv] = stock_wd_date.iloc[pi].tolist()
    else:
        [o, h, l, c, v] = stock_wd_date.iloc[0].tolist()
        [po, ph, pl, pc, pv] = stock_wd_date.iloc[1].tolist()
    change = ((c - pc) / pc) * 100
    ch = str(change.__round__(1)) + "%"

    with m1:
        s = "Open  " + str(o)
        new_title = '<p style="background : #3C6997; height : 100% ;padding : 3px; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">' + s + '</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    with m2:
        s = "High  " + str(h)
        new_title = '<p style="background : #3C6997; height : 100%;padding : 3px ; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">' + s + '</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    with m3:
        s = "Low " + str(l)
        new_title = '<p style="background : #3C6997; height : 100% ;padding : 3px; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">' + s + '</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    with m4:
        s = "Close  " + str(c)
        new_title = '<p style="background : #3C6997; height : 100% ;padding : 2px; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">' + s + '</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    with m5:
        s = "Volume  " + str(v)
        new_title = '<p style="background : #5B8C5A; height : 100%;padding : 1px ; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">' + s + '</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    with m6:
        s = "Change  " + str(ch)
        new_title = '<p style="background : #92AC86; height : 100% ;padding : 1px; margin : 10%;color:White; font-size: 25px; text-align : center;border-radius : 5px">' + s + '</p>'
        st.markdown(new_title, unsafe_allow_html=True)
except ValueError:
    st.error("No stock data listed!!!")


st.markdown(sub_title_temp.format("#89B6A5" , "white" , "EXPLORING RELATIONSHIPS AMONG VARIABLES"),unsafe_allow_html=True)
variables = stock.columns[1:]
filter_cols = st.columns(5)
choice = filter_cols[0].selectbox('With :', variables, help = 'Filter stock to show relationship with one variable.')
plot1 , plot2 = st.columns((1.5,1))
with plot1:
    fig = stock_trend_overtime(stock, choice)
    st.plotly_chart(fig, use_container_width=True)
with plot2:
    fig = stock_variables_relation_plot(stock, choice)
    st.plotly_chart(fig, use_container_width=True)

# ***********************************************************************************

model0 = load_model("forecast/models/model0.keras")
model1 = load_model("forecast/models/model1.keras")

# ***********************************************************************************
# FORECAST:
st.markdown(sub_title_temp.format("#89B6A5" , "white" , "STOCK FORECAST"),unsafe_allow_html=True)
nFut = 90
opt1 , e1 , e2 = st.columns((1,2,2))
with opt1:
    nDs = st.text_input("Enter the days range to forecast (1-90)" , 90);
    nFut = int(nDs)
# ****************************************************************************
FUT_PREDS , TRAIN_PREDS , futureDatesList , trainSet = openForecast(stock , model0 ,nFut)

chart , visual = st.columns((1,1.5))
with chart:
    fut_preds = FUT_PREDS.reset_index()
    fig = forecast_table(fut_preds, var='Open')
    st.plotly_chart(fig, use_container_width=True)
with visual:
    fig = forecast_plot(futureDatesList, FUT_PREDS, var='Open')
    st.plotly_chart(fig, use_container_width=True)


st.markdown(sub_title_temp.format("#89B6A5" , "white" , "MODEL SUMMARY"),unsafe_allow_html=True)
#PLOTTING actual vs. predicted
# Plotting
STARTDATE = TRAIN_PREDS.index[0]
# import plotly.graph_objects as go
st.markdown(head_title_temp.format("Predicted vs. Actual"),unsafe_allow_html=True)
fig = forecast_model_performance(trainSet, STARTDATE, FUT_PREDS, TRAIN_PREDS, var='Open', suffix='Price')
st.plotly_chart(fig, use_container_width=True)


# VOLUME FORECAST
st.markdown(sub_title_temp.format("#89B6A5" , "white" , "VOLUME FORECAST"),unsafe_allow_html=True)

nFutForV = 30

opt11 , e11 , e22 = st.columns((1,2,2))
with opt11:
    nDsV = st.text_input("Enter the days range to forecast (1-90)" , 30);
    nFutForV = int(nDsV)

FUT_PREDSforV , TRAIN_PREDSforV , futureDatesListForV , trainSetForV = volumeForecast(stock, model1 , nFutForV)

chartV , visualV = st.columns((1,1.5))
with chartV:
    fut_preds = FUT_PREDSforV.reset_index()
    fig = forecast_table(fut_preds, var='Volume')
    st.plotly_chart(fig, use_container_width=True)

with visualV:
    fig = forecast_plot(futureDatesListForV, FUT_PREDSforV, var='Volume')
    st.plotly_chart(fig, use_container_width=True)


st.markdown(sub_title_temp.format("#89B6A5" , "white" , "MODEL SUMMARY"),unsafe_allow_html=True)
#PLOTTING actual vs. predicted
# Plotting
STARTDATE = TRAIN_PREDSforV.index[0]

fig = forecast_model_performance(trainSetForV, STARTDATE, FUT_PREDSforV, TRAIN_PREDSforV, var='Volume', suffix='Volume')
st.plotly_chart(fig, use_container_width=True)
