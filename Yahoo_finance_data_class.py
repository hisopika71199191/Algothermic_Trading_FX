from pandas_datareader import data
from datetime import datetime
import numpy as np
import pandas as pd
import fix_yahoo_finance

class financialdata:
    def __init__(self,S_Y,S_M,S_D,E_Y,E_M,E_D, p:list = []):
        self.S_Y=S_Y
        self.S_M=S_M
        self.S_D=S_D
        self.E_Y=E_Y
        self.E_M=E_M
        self.E_D=E_D
        self.stocklabels=p
    def yahoofinance_price (self,stock_sym: str):   
        a=data.DataReader(stock_sym,'yahoo',datetime(self.S_Y,self.S_M, 
        self.S_D),datetime(self.E_Y,self.E_M,self.E_D))      
        csv=stock_sym+".csv"
        a.to_csv(csv)
        b =pd.read_csv(csv)
        return b
    def close_price(self,column):
        a=1
        for i in self.stocklabels:
            if a==1:
                y=self.yahoofinance_price(i)[column]            
                a=2
            else:
                b=self.yahoofinance_price(i)[column]      
                x=pd.concat([y, b], axis=1)
                y=x
        return y
    def read_csv_close_price(self,column):
        a=1
        for i in self.stocklabels:
            if a==1:
                y=pd.read_csv(i)[column]    
            
                a=2
            else:
                b=pd.read_csv(i)[column]            
                x=pd.concat([y, b], axis=1)
                y=x
        return y 
