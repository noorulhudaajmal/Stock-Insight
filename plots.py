import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def historical_data_table(data):
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    fig = go.Figure(
        data = [go.Table (columnorder = [0,1,2,3,4,5], columnwidth = [15,10,10,10,10,10],
                          header = dict(
                              values = list(df.columns),
                              font=dict(size=12, color = 'white'),
                              fill_color = '#264653',
                              line_color = 'rgba(255,255,255,0.2)',
                              align = ['left','center'],
                              #text wrapping
                              height=40
                          )
                          , cells = dict(
                values = [df[K].tolist() for K in df.columns],
                font=dict(size=12 , color = "black"),
                align = ['left','center'],
                line_color = 'rgba(255,255,255,0.2)',
                height=30))]
    )
    fig.update_layout(title_font_color = '#264653',
                      title_x=0,
                      margin= dict(l=0,r=10,b=10,t=30),
                      height=480,
                      title="HISTORICAL DATA")

    return fig


def quick_summary_plot(df):
    fig = px.line(x=df["Date"], y=df["Open"] ,
                 labels={"x" : "Date" , "y":"Open"})
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Stock Open Prices",
        height = 500,
        title="QUICK SUMMARY"
    )

    return fig


def stock_trend_overtime(df, choice):
    stock = df.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = stock.index.values,
        y = stock["Open"],
        line=dict(color='gray') , name = 'Open Stock Price'
    ))

    fig.add_trace(go.Scatter(
        x = stock.index.values,
        y = stock[choice],
        line=dict(color='orange') , name =  choice + 'Stock Price'
    ))
    fig.update_layout(
        xaxis_title = "Time", yaxis_title = "Value",
        title="TRENDS OVER TIME"
    )

    return fig

def stock_variables_relation_plot(df, choice):
    stock = df.copy()

    fig = px.scatter(stock , x = choice , y = "Open")
    fig.update_layout(
        title = "OPEN vs. "+choice
    )

    return fig


def forecast_table(fut_preds, var):
    fig = go.Figure(
        data = [go.Table (columnorder = [0,1], columnwidth = [15,10],
                          header = dict(
                              values = ["Date" , var],
                              font=dict(size=12, color = 'white'),
                              fill_color = '#264653',
                              line_color = 'rgba(255,255,255,0.2)',
                              align = ['left','center'],
                              #text wrapping
                              height=40
                          )
                          , cells = dict(
                values = [fut_preds[K].tolist() for K in fut_preds.columns],
                font=dict(size=12 , color = "black"),
                align = ['left','center'],
                line_color = 'rgba(255,255,255,0.2)',
                height=30))])
    fig.update_layout(title_font_color='#264653',
                      title_x=0,
                      margin= dict(l=0,r=10,b=10,t=30),
                      height=480,
                      title="Forecast Table")

    return fig


def forecast_plot(futureDatesList, FUT_PREDS, var):
    fig = px.line(x=futureDatesList, y=FUT_PREDS[var] ,
                   labels={"x" : "Date" , "y":var} ,
                   height = 500 ,width=750)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=f"{var}",
        title="Forecast Plot"
    )

    return fig


def forecast_model_performance(trainSet, STARTDATE, FUT_PREDS, TRAIN_PREDS, var, suffix):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = trainSet.loc[STARTDATE:].index,
        y = trainSet.loc[STARTDATE:][var],
        line=dict(color='blue') , name = f'Actual Stock {suffix}'
    ))

    fig.add_trace(go.Scatter(
        x = FUT_PREDS.index,
        y = FUT_PREDS[var],
        line=dict(color='red') , name = f'Future Predicted {suffix}'
    ))

    fig.add_trace(go.Scatter(
        x = TRAIN_PREDS.loc[STARTDATE:].index,
        y = TRAIN_PREDS.loc[STARTDATE:][var],
        line=dict(color='orange'), name = f'Predicted Train {suffix}'
    ))
    fig.add_vline(x=min(FUT_PREDS.index), line_width=1.5, line_dash="dash", line_color="green")

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=f"Stock {suffix}",
        height = 500 ,
    )

    return fig

