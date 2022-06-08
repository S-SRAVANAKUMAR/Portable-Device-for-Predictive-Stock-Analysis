import streamlit as st 
from datetime import date 
import time
from datetime import datetime 
import datetime 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as dat
from neuralprophet import NeuralProphet

import tensorflow as tf
 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import model_from_json
from keras.models import load_model


import yfinance as yf
from plotly import graph_objs as go


START = "2019-01-01"
CURRENT = date.today().strftime("%Y-%m-%d")

#st.title("SIMPLEX STOCKS")

title_temp = """
    <h1 style="color:white;text-align:center;"> SIMPLEX STOCKS </h1>
   """
st.markdown(title_temp,unsafe_allow_html=True)

st.markdown('##')

html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Powered by IoT </h2>
    </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)

stocks = ("APPLE","GOOGLE","MICROSOFT","AMAZON")

st.markdown("##")
st.markdown("##")

selected_stock = st.selectbox("Search Company Data",stocks)

fin_stock=''

if(selected_stock == 'APPLE'):
    fin_stock = 'AAPL'

if(selected_stock == 'GOOGLE'):
    fin_stock = 'GOOG'

if(selected_stock == 'MICROSOFT'):
    fin_stock = 'MSFT'  

if(selected_stock == 'AMAZON'):
    fin_stock = 'AMZN'        


st.markdown("##")

n_years = st.slider("Years of Prediction: ",1,5)

period = n_years*365

def load_data(ticker):
    data = yf.download(ticker,START,CURRENT)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Recieving Real-Time Data for you!...")
data=load_data(fin_stock)
data_load_state.text("Data Recieved!")  

st.markdown('##')

st.subheader('Data Preview')
st.write(data.head(10))
st.markdown('##')
st.write(data.tail(10))

st.markdown('##')
#This will add space vertically

def plot_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'],y = data['Open'],name = 'Opening Stock'))
    fig.add_trace(go.Scatter(x = data['Date'],y = data['Close'],name = 'Closing Stock'))
    fig.layout.update(title_text="Stock Price (vs) Years",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

if st.button("Graphical Analysis"):
    plot_data()

st.markdown('##')

st.subheader('Forecast Preview')    

#Neural Prophet forecasting 

from neuralprophet import NeuralProphet

df = data

df = df.rename(columns={"Date": "ds","Open":"y"})

df = df[['ds','y']]

m = NeuralProphet()

metrics = m.fit(df, freq="D")

st.markdown('##')

df['ds'] = pd.to_datetime(df['ds'])

df['ds'] = df['ds'].dt.date


future = m.make_future_dataframe(df,periods=15)
forecast_df  =m.predict(future)
forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
forecast_df['ds'] = forecast_df['ds'].dt.date
forecast_df = forecast_df[['ds','yhat1']]

st.write(forecast_df)

def plot_forecast():
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x = data['Date'],y = data['Open'],name = 'Opening Stock'))
    fig1.add_trace(go.Scatter(x = data['Date'],y = data['Close'],name = 'Closing Stock'))
    fig1.add_trace(go.Scatter(x = forecast_df['ds'],y = forecast_df['yhat1'],name = 'Forecasted Stock Price'))
    fig1.layout.update(title_text="Forecasted Stock Prices",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

st.markdown('##')

if st.button("Forecast Graphical Analysis"):
    plot_forecast()

st.markdown('##')
st.markdown('##')


# Risk Analysis

st.subheader('Risk Analysis')

open_df = dat.DataReader(['AAPL','GOOG','MSFT','AMZN'],'yahoo',START,CURRENT)['Open']
tech_rets = open_df.pct_change()


# Set up our time-horizon
days = 15

# Now our delta
dt = 1/days

# Now let's grab our mu (drift) from the expected return data we got for GOOG
mu_goog = tech_rets.mean()['GOOG']
mu_msft = tech_rets.mean()['MSFT']
mu_aapl = tech_rets.mean()['AAPL']
mu_amzn = tech_rets.mean()['AMZN']

# Now let's grab the volatility of the stock from the std() of the average return
sigma_goog = tech_rets.std()['GOOG']
sigma_msft = tech_rets.std()['MSFT']
sigma_aapl = tech_rets.std()['AAPL']
sigma_amzn = tech_rets.std()['AMZN']


def stock_monte_carlo(start_price,days,mu,sigma):
    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''
    # Define a price array
    price = np.zeros(days)
    price[0] = start_price
    # ShoCk and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    # Run price array for number of days
    for i in range(1,days):
        # Calculate Shock
        shock[i] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift
        drift[i] = mu * dt
        # Calculate Price
        price[i] = price[i-1] + (price[i-1] * (drift[i] + shock[i]))
        
    return price

# Set a large number of runs
runs = 1000

# Create an empty matrix to hold the end price data
simulations = np.zeros(runs)

# Set the print options of numpy to only display 0-5 points from an array to suppress output
np.set_printoptions(threshold=3)

start_price = 0

if fin_stock == 'GOOG':
    start_price = open_df['GOOG'].iloc[0]
    for run in range(runs):    
        # Set the simulation data point as the last stock price for that run
        simulations[run] = stock_monte_carlo(start_price,days,mu_goog,sigma_goog)[days-1]

if fin_stock == 'MSFT':
    start_price = open_df['MSFT'].iloc[0]
    for run in range(runs):    
        # Set the simulation data point as the last stock price for that run
        simulations[run] = stock_monte_carlo(start_price,days,mu_msft,sigma_msft)[days-1]

if fin_stock == 'AAPL':
    start_price = open_df['AAPL'].iloc[0]
    for run in range(runs):    
        # Set the simulation data point as the last stock price for that run
        simulations[run] = stock_monte_carlo(start_price,days,mu_aapl,sigma_aapl)[days-1]     

if fin_stock == 'AMZN':
    start_price = open_df['AMZN'].iloc[0]
    for run in range(runs):    
        # Set the simulation data point as the last stock price for that run
        simulations[run] = stock_monte_carlo(start_price,days,mu_amzn,sigma_amzn)[days-1]           

# Now we'll define q as the 1% empirical quantile, this basically means that 99% of the values should fall between this
q = np.percentile(simulations, 1)
    
st.markdown('##')

if st.button("Analyze Investment Risk"):
    st.markdown('##')
    st.write("We are predicting a Value at Risk (VaR) Proportion of: ")
    st.write((start_price - q))


# News Analysis

st.markdown('##')
st.markdown('##')

st.subheader('Stock Trading News')

from bs4 import BeautifulSoup
from urllib.request import urlopen,Request
import pandas as pd

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN','GOOG','MSFT','AAPL']


news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker
    
    req = Request(url=url,headers = {'user-agent': 'my-app'})
    response = urlopen(req)
    
    html = BeautifulSoup(response,'html')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table
        
amazon_data = news_tables['AMZN']
amazon_rows = amazon_data.findAll('tr')

for index,row in enumerate(amazon_rows):
    title = row.a.text
    timestamp = row.td.text
    print(timestamp + " " + title)

parsed_data = []

for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title])

stock_df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
stock_df['date'] = pd.to_datetime(stock_df.date).dt.date

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))

def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

stock_df['title'] = stock_df['title'].apply(lambda text: cleaning_stopwords(text))

import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations


def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


stock_df['title'] = stock_df['title'].apply(lambda text: cleaning_punctuations(text))


import re
def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)

stock_df['title'] = stock_df['title'].apply(lambda text: cleaning_repeating_char(text))

def cleaning_numbers(text):
    return re.sub('[0-9]+', '', text)

stock_df['title'] = stock_df['title'].apply(lambda text: cleaning_numbers(text))

import nltk
from nltk.tokenize import word_tokenize

stock_df['tokenized_sents'] = stock_df.apply(lambda row: nltk.word_tokenize(row['title']), axis=1)

lm = nltk.WordNetLemmatizer()

def lemmatizer_on_text(text):
    text = [lm.lemmatize(word) for word in text]
    return text

stock_df['tokenized_sents'] = stock_df['tokenized_sents'].apply(lambda text: lemmatizer_on_text(text))    

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

x = lambda title: sia.polarity_scores(title)['compound']
stock_df['compound'] = stock_df['title'].apply(x)

def get_score(score):
    if score >= 0.02:
        return 'Buy'
    
    else:
        return 'Sell'

stock_df['Analysis'] = stock_df['compound'].apply(get_score) 

stock_df = stock_df[['ticker','date','time','title','Analysis']]

stock_df_aapl = stock_df[stock_df.ticker == 'AAPL']
stock_df_amzn = stock_df[stock_df.ticker == 'AMZN']
stock_df_msft = stock_df[stock_df.ticker == 'MSFT']
stock_df_goog = stock_df[stock_df.ticker == 'GOOG']

st.markdown("##")

news = 0
stock_name = 0

if(fin_stock == 'AAPL'):
    stock_name = 0

if(fin_stock == 'GOOG'):
    stock_name = 1
    
if(fin_stock == 'MSFT'):
    stock_name = 2  

if(fin_stock == 'AMZN'):
    stock_name = 3  

if stock_df_aapl['Analysis'].iloc[0] == 'Buy':
    news = 1 

if stock_df_goog['Analysis'].iloc[0] == 'Buy':
    news = 1       

if stock_df_amzn['Analysis'].iloc[0] == 'Buy':
    news = 1 

if stock_df_msft['Analysis'].iloc[0] == 'Buy':
    news = 1             

if st.button("News Articles"):
    st.markdown("##")
    st.markdown("##")
    if fin_stock == 'AAPL':
        st.write(stock_df_aapl) 

    if fin_stock == 'GOOG':
        st.write(stock_df_goog)       

    if fin_stock == 'AMZN':
        st.write(stock_df_amzn)

    if fin_stock == 'MSFT':
        st.write(stock_df_msft)      

# Parameters

open_price = forecast_df['yhat1'].iloc[0] #val1
#val2 -> (start_price - q)
trend_df = forecast_df[['yhat1']]
trend_df = trend_df.pct_change()
trend = trend_df.mean()['yhat1'] #val3 
#val5 -> news


# Function to post data to Thingspeak Cloud 

import urllib.request

def thingspeak_post_aapl(valm1,valm2,valm3,valm4,valm5):
    val1 = valm1
    val2 = valm2
    val3 = valm3
    val4 = valm4
    val5 = valm5
    URl = 'https://api.thingspeak.com/update?api_key='
    KEY = 'ZA4FI7MH2VJ9D16R'
    HEADER = '&field1={}&field2={}&field3={}&field4={}&field5={}'.format(val1,val2,val3,val4,val5)
    NEW_URL = URl+KEY+HEADER
    print(NEW_URL)
    data_thingspeak = urllib.request.urlopen(NEW_URL)
    print(data_thingspeak)

def thingspeak_post_amzn(valm1,valm2,valm3,valm4,valm5):
    val1 = valm1
    val2 = valm2
    val3 = valm3
    val4 = valm4
    val5 = valm5
    URl = 'https://api.thingspeak.com/update?api_key='
    KEY = 'JUX9NP24N8KI6PT7'
    HEADER = '&field1={}&field2={}&field3={}&field4={}&field5={}'.format(val1,val2,val3,val4,val5)
    NEW_URL = URl+KEY+HEADER
    print(NEW_URL)
    data_thingspeak = urllib.request.urlopen(NEW_URL)
    print(data_thingspeak)

def thingspeak_post_msft(valm1,valm2,valm3,valm4,valm5):
    val1 = valm1
    val2 = valm2
    val3 = valm3
    val4 = valm4
    val5 = valm5
    URl = 'https://api.thingspeak.com/update?api_key='
    KEY = '4C61WM21LMGDQZK2'
    HEADER = '&field1={}&field2={}&field3={}&field4={}&field5={}'.format(val1,val2,val3,val4,val5)
    NEW_URL = URl+KEY+HEADER
    print(NEW_URL)
    data_thingspeak = urllib.request.urlopen(NEW_URL)
    print(data_thingspeak)

def thingspeak_post_goog(valm1,valm2,valm3,valm4,valm5):
    val1 = valm1
    val2 = valm2
    val3 = valm3
    val4 = valm4
    val5 = valm5
    URl = 'https://api.thingspeak.com/update?api_key='
    KEY = '6BKSY4REPIWFBOZ1'
    HEADER = '&field1={}&field2={}&field3={}&field4={}&field5={}'.format(val1,val2,val3,val4,val5)
    NEW_URL = URl+KEY+HEADER
    print(NEW_URL)
    data_thingspeak = urllib.request.urlopen(NEW_URL)
    print(data_thingspeak)    

st.markdown('##')
st.markdown('##')    
  
load_state = st.text("Configuring and Sending data to your IoT Device..") 
if fin_stock == 'AAPL':
    thingspeak_post_aapl(open_price,(start_price-q),trend,stock_name,news)

if fin_stock == 'AMZN':
    thingspeak_post_amzn(open_price,(start_price-q),trend,stock_name,news)

if fin_stock == 'GOOG':
    thingspeak_post_goog(open_price,(start_price-q),trend,stock_name,news)

if fin_stock == 'MSFT':
    thingspeak_post_msft(open_price,(start_price-q),trend,stock_name,news)     
load_state.text("Data Sent!")  
