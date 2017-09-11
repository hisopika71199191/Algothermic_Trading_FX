def backtest(testPredict):
    initial_capital = 100000
    position=1 # 1 means the money is used to purchase stock
    old = testY[0][0]
    strategy_1=[initial_capital]
    for i in range(0,len(testPredict)):
        if testPredict[i]<old:
            if position == 1:
                initial_capital = initial_capital*(1+(testY[0][i+1]- old )/old) #or initial_capital*(1+log)                          
                print('Day:',i+1,',','Action:sell,' ,'capital:',initial_capital)
                position=0
                old = testY[0][i+1]
                strategy_1.append(initial_capital)
            else:
                    print('Day:',i+1,',','NO ACTION') 
                    strategy_1.append(initial_capital)
        else :
            if position == 0:
                old = testY[0][i]  
                position=1
                print("Day:",i+1,",",'Action:buy' ,'capital:',initial_capital)
                strategy_1.append(initial_capital)
            else:
                print ('Day:',i+1,',','NO ACTION')
                strategy_1.append(initial_capital)
       
        if position ==1:
            initial_capital = initial_capital*(1+(testY[0][len(testPredict)-1]- old )/old)     
            strategy_1[len(strategy_1)-1]=initial_capital

    return(strategy_1)





     
