#Description: use KNN to learn the historical price returns to predict next returns


##library
import numpy as np
import pandas as pd
import oandapy as op
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import datetime,timedelta
import pycurl
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.trades as transactions
import oandapyV20 as opV20
from numba import jit
import sys
sys.path.append('C:\\Users\\user\\Desktop\\MATHS&IT\\Project\\OANDA')
## params of  settings
#oanda params
account_id ="<OANDA ACCOUNT ID>"
key = "<OANDA API>"
curl = pycurl.Curl()
curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 9c274108805eae5e68978df2344d7050-f3d570fa97f654c8067629af5714b8ee'])
oanda20 = opV20.API(access_token=key,environment="practice")
oanda = op.API(environment="practice", access_token=key)

#system params
instruments = ["EUR_USD"]
position_table={instruments[0]:[0,0],instruments[1]:[0,0]}
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
trade_units=100

#external data
#read the large historical dataset and later append it to real time streaming rate
histdata_0=pd.read_csv('C:\\Users\\user\\Desktop\\MATHS&IT\\Project\\OANDA\\data\\'+instruments[0] + "\\"+instruments[0]+'.csv',usecols=[0,1,2,3])
#histdata_0=pd.read_csv('C:\\Users\\user\\Desktop\\MATHS&IT\\Project\\OANDA\\data\\'+instruments[0] + "\\"+instruments[0]+'_stream.csv',usecols=[0,1,2,3],names=["instrument","times","bid","ask"])
histdata_0=histdata_0[["instrument","times","bid","ask"]]

#histdata_1=pd.read_csv('C:\\Users\\user\\Desktop\\MATHS&IT\\Project\\OANDA\\data\\'+instruments[1] + "\\"+instruments[1]+'.csv',usecols=[0,1,2,3])
#histdata_1=pd.read_csv('C:\\Users\\user\\Desktop\\MATHS&IT\\Project\\OANDA\\data\\'+instruments[1] + "\\"+instruments[1]+'_stream.csv',usecols=[0,1,2,3],names=["instrument","times","bid","ask"])
#histdata_1=histdata_1[["instrument","times","bid","ask"]]

price_dict={"instrument":[],'times':[],'bid':[],'ask':[]}

#%%
##############################class and functions
class settings:
    def __init__(self,account_id,key,curl,oanda20,oanda,instruments,position_table,order,position,trade_units_0):
        self.account_id=account_id
        self.key=key
        self.curl=curl
        self.oanda20=oanda20
        self.oanda=oanda
        self.instruments=instruments
        self.position_table=position_table
        self.order=order
        self.position=position 
        self.trade_units_0=trade_units_0

class price_streaming(settings):   
    def looping_price(self,histdata,i):
        global price_dict
        global price_table
        #while True:
        time.sleep(1)    
        response_i = self.oanda.get_prices(instruments=instruments[i],tz="UTC+8")
        time.sleep(1)
        prices_i = response_i["prices"]
        time.sleep(1)
        price_dict["times"].append(prices_i[0]["time"])  #The new data stored in price_dict 
        price_dict["bid"].append(float(prices_i[0]["bid"]))         
        price_dict["ask"].append(float(prices_i[0]["ask"]))
        price_dict["instrument"].append(prices_i[0]["instrument"])
        price_table=pd.DataFrame.from_dict(price_dict)[["instrument",'times','bid','ask']]
        price_table['times']=pd.to_datetime(price_table['times'],format="%Y-%m-%dT%H:%M:%S.%fZ").apply(lambda x: x.replace(microsecond=0))
        price_table['times']=price_table['times'].apply(lambda x: x.replace(second=0))
        price_table['times']=price_table['times'].apply(lambda x: x + timedelta(hours=8))
#        price_table.drop_duplicates(subset='times', keep='last', inplace=True)
#        price_table.reset_index(inplace=True,drop=True)                          
        price_table_0=price_table[price_table.instrument==self.instruments[i]]
        price_table_0.drop_duplicates(subset='times', keep='first', inplace=True)
        price_table_0.reset_index(inplace=True,drop=True)       
        data_0 = histdata.append(price_table_0)
        data_0['times']=pd.to_datetime(data_0.times)
        data_0=data_0.sort_values(by='times')
        data_0.drop_duplicates(subset='times', keep='last', inplace=True)
        data_0["mid"] =(data_0["bid"] + data_0["ask"])/2       #if price_dict is cleared, _stream_csv is distorted
        data_0.to_csv('C:\\Users\\user\\Desktop\\MATHS&IT\\Project\\OANDA\\data\\'+self.instruments[i]+"\\"+self.instruments[i]+'_stream.csv',index=False,header=False)    
        return data_0      

    def read_data(self,data,i):
        data_0=data
        data_0.columns=["instrument","times",self.instruments[i]+"_bid",self.instruments[i]+"_ask",self.instruments[i]+"_mid"]
        #data_0=pd.read_csv('C:\\Users\\user\\Desktop\\MATHS&IT\\Project\\OANDA\\data\\'+self.instruments[i]+"\\"+self.instruments[i]+'_stream.csv',names=["instrument","times",self.instruments[i]+"_bid",self.instruments[i]+"_ask",self.instruments[i]+"_mid"])
        data_0.drop('instrument', axis=1, inplace=True)
 #       chane to  5 miuntes interval:
  #      data_0.index=data_0['times']
  #      data_0.index=data_0.index.to_datetime()
#        min_5= data_0.index.to_series().dt.minute.isin([0, 5, 10, 15, 20,25,30,35,40,45,50,55]) 
 #       data_0=data_0.loc[min_5]
#        chane to  15 miuntes interval:
#        data_0.index=data_0['times']
#        data_0.index=data_0.index.to_datetime()
#        min_15= data_0.index.to_series().dt.minute.isin([0, 15,30,45]) 
#        data_0=data_0.loc[min_15]
        return data_0  


    
class order_maker(price_streaming):       
    def create_order(self,ordertype,units,instrument_index):
        self.order["order"]["type"]=ordertype
        self.order["order"]["units"]=units                               #str(position_table[-1]*100)      
        self.order["order"]["instrument"] = self.instruments[instrument_index]           
        r_order = orders.OrderCreate(self.account_id, data=self.order)
        self.oanda20.request(r_order)
        trade_list = trades.TradesList(accountID=self.account_id,params = self.instruments[instrument_index])
        trade_reponse=self.oanda20.request(trade_list)
        trade_price=trade_reponse['trades'][0]['price']
        return (trade_price)
    def close_trade(self,instrument_index,longunits,shortunits):
        self.position["longUnits"] = longunits
        self.position["shortUnits"] = shortunits
        close=positions.PositionClose(accountID = account_id,
                                              instrument=self.instruments[instrument_index],
                                              data=position)
        self.oanda20.request(close)


class quant_models(price_streaming,settings):
    def price_return(self,dataset,field,i):
        returns = np.log(dataset[self.instruments[i]+"_"+field]/dataset[self.instruments[i]+"_"+ field].shift(1))
        return returns    
    @jit
    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    @jit
    def KNN_train(self,dataset,field,window_size,i):
        returns=self.price_return(dataset,field,i)
        returns.dropna(inplace=True)
        window=self.rolling_window(returns,window_size)
        X=window[:,1:(window_size+1)]
      
        KNN=KNeighborsRegressor(n_neighbors = round(np.sqrt(round(window.shape[0]))).astype(int)).fit(X=X,y=window[:,0].reshape(X.shape[0],1))
        return KNN
    
    def KNN_apply(self,dataset,field,window_size,model,i):
        returns=self.price_return(dataset.tail(window_size),field,i)     
        returns.dropna(inplace=True)
        returns=np.array(returns)
        X=np.transpose(returns).reshape(1,window_size-1)
        prediction=model.predict(X=X)
        return prediction
        
class strategy_trade(order_maker,quant_models,settings):
    def trade(self, dataset,window_size,i):
        dataset=pd.DataFrame(dataset)
        global trade_price_0
        global wait
        train_model_ask=self.KNN_train(dataset=dataset,field="ask",window_size=window_size,i=0)
        train_model_bid=self.KNN_train(dataset=dataset,field="bid",window_size=window_size,i=0)                
        predict_return_ask = self.KNN_apply(dataset=dataset,field="ask",window_size=window_size,model=train_model_ask,i=i)
        predict_return_bid = self.KNN_apply(dataset=dataset,field="bid",window_size=window_size,model=train_model_bid,i=i)
        last_ask=np.array(dataset[self.instruments[i]+"_ask"].tail(1))[0]
        last_bid=np.array(dataset[self.instruments[i]+"_bid"].tail(1))[0]
        predict_ask = np.array(last_ask*(np.exp(predict_return_ask[0])))
        predict_bid = np.array(last_bid*(np.exp(predict_return_bid[0])))
        print("predict_ask:",predict_ask)
        print("predict_bid:",predict_bid)
        label=0
        
        ##prediction signal
        if np.array(predict_bid-last_ask) > 0:
            label=1   
        if np.array(last_bid-predict_ask) > 0:
            label=-1           
        ##take profit due to market price event     
#        if position_table[self.instruments[i]][-1]==1:
#            if  np.array(float(trade_price_0) - dataset[self.instruments[i]+"_ask"])[0] >= cost_0:
#                self.close_trade(0,longunits="NONE",shortunits="ALL")
#                position_table[self.instruments[i]].append(0)
#                print("take profit first")
#        if position_table[self.instruments[i]][-1]==-1:
#            if  np.array(dataset[self.instruments[i]+"_bid"]  - float(trade_price_0))[0] >= cost_0:
#                self.close_trade(0,longunits="NONE",shortunits="ALL")
#                position_table[self.instruments[i]].append(0)
#                print("take profit first")
                                  
        #trading  according to signa    
        if position_table[self.instruments[i]][-1]==0 and label==1: 
            trade_price_0=float(self.create_order("MARKET",self.trade_units,0))
            position_table[self.instruments[i]].append(1)
            print("open long")
        elif position_table[self.instruments[i]][-1]==0 and label==-1 :
            trade_price_0=float(self.create_order("MARKET",self.trade_units*(-1),0))
            position_table[self.instruments[i]].append(-1)
            print("open short")
        elif position_table[self.instruments[i]][-1]==-1 and label==1 and np.array(float(trade_price_0)-last_ask) > 0:
            self.close_trade(0,longunits="NONE",shortunits="ALL")
            trade_price_0=float(self.create_order("MARKET",self.trade_units,0))
            position_table[self.instruments[i]].append(1)
            print("close short and open long")
        elif position_table[self.instruments[i]][-1]==1 and label==-1 and np.array(float(trade_price_0)-last_bid) < 0:
            self.close_trade(0,longunits="ALL",shortunits="NONE")
            trade_price_0=float(self.create_order("MARKET",self.trade_units*(-1),0))
            position_table[self.instruments[i]].append(-1)
            print("close long and open short")
            
             
#        elif label==0 and position_table[self.instruments[i]][-1] !=0 and wait ==0:
#            time.sleep(60)
#            wait=1


          
        elif label==0 and position_table[self.instruments[i]][-1] !=0 :
            if position_table[self.instruments[i]][-1]==-1 and np.array(float(trade_price_0)-last_ask*1.1)>0:
                self.close_trade(0,longunits="NONE",shortunits="ALL")
                position_table[self.instruments[i]].append(0)
                print("close all position and take profit")
            elif position_table[self.instruments[i]][-1]==1 and np.array(float(trade_price_0)-last_bid*0.9)<0:
                self.close_trade(0,longunits="ALL",shortunits ="NONE")
                position_table[self.instruments[i]].append(0)
                print("close all position and take profit")
            else:
                print("wait and hold")
        elif wait != 75 and position_table[self.instruments[i]][-1] !=0:
            wait +=1
            print("do nothing or wait close position")
            print("label",label," position",position_table[self.instruments[i]][-1])
        elif wait == 75 and position_table[self.instruments[i]][-1] !=0:
            if position_table[self.instruments[i]][-1]==-1: 
                self.close_trade(0,longunits="NONE",shortunits="ALL")
                position_table[self.instruments[i]].append(0)
                print("cut loss or stop trailing/holding to take profit")
                wait=0
            elif position_table[self.instruments[i]][-1]==1: 
                self.close_trade(0,longunits="ALL",shortunits="NONE")
                position_table[self.instruments[i]].append(0)
                print("cut loss or stop trailing/holding to take profit")
                wait=0
        else:
            print("logic hole")
            print("label:",label)
            print("position:",position_table[self.instruments[i]][-1])
        
            
#%%        
################### Applying Class
price_streaming_var=price_streaming(account_id=account_id,key=key,curl=curl,oanda20=oanda20,
                                    oanda=oanda,instruments=instruments,position_table=position_table,order=order,
                                    position=position,histdata_0=histdata_0,price_dict=price_dict,trade_units=trade_units)

price_streaming_var.looping_price(i=0) 
dataset = price_streaming_var.read_data(i=0)
quant_models_var=quant_models(account_id=account_id,key=key,curl=curl,oanda20=oanda20,
                              oanda=oanda,instruments=instruments,position_table=position_table,order=order,
                              position=position,histdata_0=histdata_0,price_dict=price_dict,trade_units=trade_units)
    

window_size= 120

#If training a model is two slow and not re-train every time, just train here and put the trained models to trade() method  
#train_model_ask=quant_models_var.KNN_train(dataset=dataset,field="ask",window_size=window_size,i=0)
#train_model_bid=quant_models_var.KNN_train(dataset=dataset,field="bid",window_size=window_size,i=0)                

strategy_trade_var=strategy_trade(account_id=account_id,key=key,curl=curl,oanda20=oanda20,
                              oanda=oanda,instruments=instruments,position_table=position_table,order=order,
                              position=position,histdata_0=histdata_0,price_dict=price_dict,trade_units=trade_units)


#%%
########################### Trade
index=0
wait=0
while 1==1:
    time_start=datetime.now()
    time.sleep(1)
    data_0=price_streaming_var.looping_price(histdata=histdata_0,i=0)
    dataset=price_streaming_var.read_data(data=data_0,i=0)
    print(dataset.tail(1))
    try:    
        strategy_trade_var.trade(dataset=dataset,window_size=window_size,i=0)
        print("trading")
    except:
        print("error")
        continue
    time_end=datetime.now()
    print("time:running a strategy:",timedelta.total_seconds( time_start-time_end))
    
    
    
    




