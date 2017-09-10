#####Not Complete yet
#####Not Tested
#2017-09-11


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

##other settings
instruments = "EUR_USD"

#This is external data source, it will be appended with streaming price to calculate expected price repeatedly 
histdata = EUR_USD_price

#intital input params. for oanda api.
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


#data preparation for external data 
histdata["times"] = "" + histdata.index + " "+ histdata["times"]
histdata=histdata.reset_index(drop=True)
histdata["times"]=pd.to_datetime(histdata["times"], format="%Y.%m.%d %H:%M")
position_table=[0]
##########################################################################Trading
def trade():
    r_orderpend=orders.OrdersPending(accountID=account_id)
    pending=oanda20.request(r_orderpend)
    # no pending orders:
    if pending['orders']==[]:
        
        now=datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        start = histdata.iloc[0,0].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        response = oanda.get_history(instrument=["EUR_USD"],granularity="M1",end=now)
        temp_table=pd.DataFrame.from_dict(response["candles"])
        table=pd.DataFrame()
        table["times"]=pd.to_datetime(temp_table["time"], format="%Y-%m-%dT%H:%M:%S.%fZ")
        table["open"]=(temp_table["openAsk"]+temp_table["openBid"])/2
        table["high"]=(temp_table["highAsk"]+temp_table["highBid"])/2
        table["low"]=(temp_table["lowAsk"]+temp_table["lowBid"])/2
        table["close"]=(temp_table["closeAsk"]+temp_table["closeBid"])/2
        data=table
        data = histdata.append(table)
        data=data.reset_index(drop=True)
        cost=np.mean(temp_table["closeAsk"]-temp_table["closeBid"])
        

##Model    
        #arima
        #reference source:http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016
        returns=np.log(data["close"].shift(1)/data["close"]).dropna() 
        returns.index=data["times"][-len(data["times"]):-1]
        best_aic = np.inf 
        best_order = None
        best_mdl = None
        pq_rng = range(5) # [0,1,2,3,4]
        d_rng = range(2) # [0,1]
        for i in pq_rng:
            for d in d_rng:
                for j in pq_rng:
                    try:
                        tmp_mdl = smt.ARIMA(returns, order=(i,d,j)).fit(method='mle', trend='nc')                
                        tmp_aic = tmp_mdl.aic
                        if tmp_aic < best_aic:
                            best_aic = tmp_aic
                            best_order = (i, d, j)
                            best_mdl = tmp_mdl
                    except: continue
        print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
        predict=best_mdl.forecast(steps=20)[0]
        predict_price=[]
        last_price =data["close"].iloc[-1]
        
###Strategy: predict mid price for next 20 mins (1-min interval)      
        for i in range(0,20):
            predict_price.append(last_price*(np.exp(predict[i])))
            last_price=last_price*(np.exp(predict[i])) 
            #retain about 5 mins to run the model, so [5:19] is used             
        if (max(predict_price[5:19])-data["close"].iloc[-1] > cost) & (data["close"].iloc[-1]-min(predict_price[5:19]) > max(predict_price[5:19])-data["close"].iloc[-1]) & position_table[-1] !=1:                
            position_table.append(1)         
            order["order"]["type"]="MARKET"
            order["order"]["units"]= str(position_table[-1]*1)      
            order["order"]["instrument"]=instruments           
            r_order = orders.OrderCreate(account_id, data=order)
            oanda20.request(r_order)     
        elif (data["close"].iloc[-1]-min(predict_price[5:19]) > cost) & (data["close"].iloc[-1]-min(predict_price[5:19]) < max(predict_price[5:19])-data["close"].iloc[-1]) & position_table[-1] !=-1:        
            position_table.append(-1)
            order["order"]["type"]="MARKET"        
            order["order"]["units"]= str(position_table[-1]*100)      
            order["order"]["instrument"]=instruments
            r_order = orders.OrderCreate(account_id, data=order)
            oanda20.request(r_order)             
        elif (max(predict_price[5:19])-data["close"])*position_table[-1] < 0 or (min(predict_price[5:19])-data["close"])*position_table[-1]<0:
             position_table[-1].append(0)
             positions.close
             
        def look():
           response = oanda.get_prices(instruments=instruments,tz="UTC+8")
           prices = response["prices"]
           bidding_price = float(prices[0]["bid"])
           asking_price = float(prices[0]["ask"])
           instrument = prices[0]["instrument"]
        
           if position_table[-1] == 1:
               if asking_price>=max(predict_price[5:19]):
                   position["longUnits"]="ALL"
                   position["shortUnits"]="NONE"
                   close=positions.PositionClose(accountID = account_id,
                                                 instrument=instruments,
                                                 data=position)
                   oanda20.request(r_order)                                             
           elif position_table[-1] == -1:
               if  asking_price<=min(predict_price[5:19]):
                   position["longUnits"]="NONE"
                   position["shortUnits"]="ALL"
                   close=positions.PositionClose(accountID = account_id,
                                           instrument=instruments,
                                           data=position)
                   oanda20.request(r_order)                    
                     
        #determine whether close the position (by each 1-min interval)
        s=threading.Timer(60, look)
        s.start()
        
        if len(position_table) == 21:
            s.cancel()
                   

#run the trading algo every 20 mins
t = threading.Timer(1200,trade)
t.start()
    

#t.cancel()




