#Version: 2017-09-11


import numpy as np
import pandas as pd
import oandapy as op
from datetime import datetime, timedelta
import pycurl
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
import oandapyV20 as opV20
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import scipy.stats as scs
from arch import arch_model
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from arch.unitroot import ADF
from arch.univariate import ARX
from arch import arch_model
import time, threading

##oanda20
account_id ="<your oanda account id>"
key = "<your oanda API token>"
curl = pycurl.Curl()
curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer <API token>'])
oanda20 = opV20.API(access_token=key,environment="practice")

##oanda
# mainly use oandapyV20, oandapy is for legacy oanda account. I use oandapy for streaming price only
oanda = op.API(environment="practice", access_token=key)
instruments = "EUR_USD"

position_table=[0,0]

order={
"order": {
"instrument": "X",
"units": "100",
"type": "MARKET",
"positionFill": "DEFAULT"
        }
       } 

position={
"longUnits": "ALL",
"shortUnits": "NONE"
}

histdata=pd.read_csv('\\\\192.168.88.250\\BigUSB\\Research\ChanPoKin\\Data\\EUR_USD.csv')
histdata=histdata[histdata['times']>'2017-08-12'].iloc[:,1:10]


#params ={
#"instruments": instruments
#}
#
#r = pricing.PricingStream(accountID=account_id, params=params)
#oanda20.request(r) 
#
#t=0
#while t<10:
#    t += 1
#    for R in oanda20.request(r) :
#        print(json.dumps(R, indent=2))
# 
##        if n > 10:
##            s.terminate("maxrecs received: {}".format(MAXREC))
#        time.sleep(59)

#except V20Error as e:
#    print("Error: {}".format(e))
 

temp_table=histdata
table=pd.DataFrame()
table["times"]=pd.to_datetime(temp_table["times"])
table["open"]=(temp_table["openAsk"]+temp_table["openBid"])/2
table["high"]=(temp_table["highAsk"]+temp_table["highBid"])/2
table["low"]=(temp_table["lowAsk"]+temp_table["lowBid"])/2
table["close"]=(temp_table["closeAsk"]+temp_table["closeBid"])/2
data=table
returns=np.log(data["close"].shift(1)/data["close"]).dropna() 
returns.index=data["times"][-len(data["times"]):-1]
best_aic = np.inf 
best_order = None
best_mdl = None
pq_rng = range(7) # [0,1,2,3,4]
d_rng = range(1) # [0,1]
for i in pq_rng:
    for d in d_rng:
        for j in pq_rng:
            try:
                tmp_mdl= smt.ARIMA(returns, order = (i,d,j)).fit(trend='c',method='css-mle')              
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, d, j)
                    best_mdl = tmp_mdl
            except: continue
print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
   
     
n = 0
time_now=datetime.now()
k=int(np.round(timedelta.total_seconds((time_now-pd.to_datetime(histdata.iloc[-1,0]))/60)))
while n <=100:
    n +=1
    time_start=datetime.now()
#    now=datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
#   hist=oanda.get_history(instrument=["EUR_USD"],granularity="M1",end=now,count=5000)
#   temp_table=pd.DataFrame.from_dict(hist["candles"])
#   table=pd.DataFrame()
#   table["times"]=pd.to_datetime(temp_table["time"], format="%Y-%m-%dT%H:%M:%S.%fZ")
#   table["open"]=(temp_table["openAsk"]+temp_table["openBid"])/2
#   table["high"]=(temp_table["highAsk"]+temp_table["highBid"])/2
#   table["low"]=(temp_table["lowAsk"]+temp_table["lowBid"])/2
#   table["close"]=(temp_table["closeAsk"]+temp_table["closeBid"])/2
#   histdata=table
        
#    start = histdata.iloc[-1,0].strftime('%Y-%m-%dT%H:%M:%S.%fZ') 
#    response = oanda.get_history(instrument=["EUR_USD"],granularity="M1",end=now,start=start)
#    temp_table=pd.DataFrame.from_dict(response["candles"])
#    temp_table=histdata
#    table=pd.DataFrame()
#    table["times"]=pd.to_datetime(temp_table["times"], format="%Y-%m-%dT%H:%M:%S.%fZ")
#    table["times"]=pd.to_datetime(temp_table["times"])
#    table["open"]=(temp_table["openAsk"]+temp_table["openBid"])/2
#    table["high"]=(temp_table["highAsk"]+temp_table["highBid"])/2
#    table["low"]=(temp_table["lowAsk"]+temp_table["lowBid"])/2
#    table["close"]=(temp_table["closeAsk"]+temp_table["closeBid"])/2
#    data=table
#    data = histdata.append(table)
#    data=data.reset_index(drop=True)
#    histdata=data
    #cost=np.mean((temp_table["closeAsk"]-temp_table["closeBid"])/2)
    cost=0
        
    #model
#    returns=np.log(data["close"].shift(1)/data["close"]).dropna() 
#    returns.index=data["times"][-len(data["times"]):-1]
#    best_aic = np.inf 
#    best_order = None
#    best_mdl = None
#    pq_rng = range(7) # [0,1,2,3,4]
#    d_rng = range(1) # [0,1]
#    for i in pq_rng:
#        for d in d_rng:
#            for j in pq_rng:
#                try:
#                    tmp_mdl= smt.ARIMA(returns, order = (i,d,j)).fit(trend='c',method='css-mle')              
#                    tmp_aic = tmp_mdl.aic
#                    if tmp_aic < best_aic:
#                           best_aic = tmp_aic
#                           best_order = (i, d, j)
#                           best_mdl = tmp_mdl
#                except: continue
#    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    predict=best_mdl.forecast(steps=k+30)[0]
    predict_price=[]
    last_price =data["close"].iloc[-1]
    #Strategy
    for i in range(0,k+30):
        predict_price.append(last_price*(np.exp(predict[i])))
        last_price=last_price*(np.exp(predict[i])) 
        
    response = oanda.get_prices(instruments=instruments,tz="UTC+8")
    prices = response["prices"]
    bidding_price = float(prices[0]["bid"])
    asking_price = float(prices[0]["ask"])
    instrument = prices[0]["instrument"]
    time_end_1=datetime.now()
    #mid=(bidding_price +asking_price)/2
    #model_time_1=int(np.round(timedelta.total_seconds(time_end_1-time_start)/60))
    model_time=int(np.round(timedelta.total_seconds(time_end_1-data.iloc[-1,0])/60))
   
    predict_price_n=predict_price[(model_time):(k+30)]                         
    if (max(predict_price_n)-asking_price > cost) and ((bidding_price-min(predict_price_n)) < (max(predict_price_n)-asking_price)) and position_table[-1] !=1:                
        position_table.append(1)         
        order["order"]["type"]="MARKET"
        order["order"]["units"]= str(position_table[-1]*100)      
        order["order"]["instrument"]=instruments           
        r_order = orders.OrderCreate(account_id, data=order)
        oanda20.request(r_order)
        trade_list = trades.TradesList(accountID=account_id,params=instruments)
        trade_reponse=oanda20.request(trade_list)
        trade_price=trade_reponse['trades'][0]['price'] 
    elif (bidding_price-min(predict_price_n) > cost) and ((bidding_price-min(predict_price_n)) > (max(predict_price_n)-asking_price)) and position_table[-1] !=-1:        
        position_table.append(-1)
        order["order"]["type"]="MARKET"        
        order["order"]["units"]= str(position_table[-1]*100)      
        order["order"]["instrument"]=instruments
        r_order = orders.OrderCreate(account_id, data=order)
        oanda20.request(r_order)
        trade_list = trades.TradesList(accountID=account_id,params=instruments)
        trade_reponse=oanda20.request(trade_list)
        trade_price=trade_reponse['trades'][0]['price']                      
    elif (max(predict_price_n)-asking_price)*position_table[-1] < cost or (min(predict_price_n)-bidding_price)*position_table[-1]>cost:
        position_table.append(position_table[-2])
        
        
#    trade_list = trades.TradesList(accountID=account_id,params=instruments)
#    trade_reponse=oanda20.request(trade_list)
#    trade_price=trade_reponse['trades'][0]['price']   
        
    j=0
    timer_a=now=datetime.now()
    while (j<=282000) and (position_table[-1]!=0):
        j+=1
        response = oanda.get_prices(instruments=instruments,tz="UTC+8")
        prices = response["prices"]
        bidding_price = float(prices[0]["bid"])
        asking_price = float(prices[0]["ask"])
        instrument = prices[0]["instrument"]
        
        if position_table[-1] == 1:
            if  bidding_price>=float(trade_price) + 0.00007:
                position["longUnits"]="ALL"
                position["shortUnits"]="NONE"
                close=positions.PositionClose(accountID = account_id,
                                              instrument=instruments,
                                              data=position)
                oanda20.request(close)
                position_table.append(0)
                 
        elif position_table[-1] == -1:
            if  asking_price<=(float(trade_price) - 0.00007):
                position["longUnits"]="NONE"
                position["shortUnits"]="ALL"
                close=positions.PositionClose(accountID = account_id,
                                              instrument=instruments,
                                              data=position)
                oanda20.request(close) 
                position_table.append(0)
        
    timer_b=now=datetime.now()                    
    check=timedelta.total_seconds(timer_b-timer_a)
    
    if position_table[-1] == 1:
        position["longUnits"]="ALL"
        position["shortUnits"]="NONE"
        close=positions.PositionClose(accountID = account_id,
                                              instrument=instruments,
                                              data=position)
        oanda20.request(close)
        position_table.append(0)                              
    elif position_table[-1] == -1:
        position["longUnits"]="NONE"
        position["shortUnits"]="ALL"
        close=positions.PositionClose(accountID = account_id,
                                              instrument=instruments,
                                              data=position)
        oanda20.request(close) 
        position_table.append(0)    
    
    
    
    
    time_end=datetime.now()
    wait=timedelta.total_seconds(time_end-time_start)
    k=model_time
    print('k:',k)
    print("time looking for each tick:",check)
    print("time for each loop:",wait)
    print("positions:",position_table)

    
    

    
    
    




