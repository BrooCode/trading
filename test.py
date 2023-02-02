from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler
from keras import models

scaler=MinMaxScaler(feature_range=(0,1))
df_nse = pd.read_csv("RELIANCE.csv")
df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']
data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','wap'])
for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["wap"][i]=data["wap"][i]
new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)
dataset=new_data.values
train=dataset[0:750,:]
valid=dataset[750:878,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
x_train,y_train=[],[]
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
model=models.load_model("saved_model.h5")
inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)
X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)
train=new_data[:750]
valid=new_data[750:878]
date = df_nse['Date']
date = date.tolist()
Expiry_date = df_nse['Expiry_date']
Expiry_date = Expiry_date.tolist()
lot_size = df_nse['lot_size']
lot_size = lot_size.tolist()
valid['Predictions']=closing_price
valid['Date'] = date[750:878]
valid['Expiry_date']=Expiry_date[750:878]
valid['lot_size']=lot_size[750:878]
plt.figure(figsize=(16,8))
# print(valid[['wap',"Predictions","Date",'Expiry_date','lot_size']])
plt.plot(valid[['wap',"Predictions"]])
# print(valid[['wap',"Predictions"]])
valid = valid.values.tolist()


# print(valid)
#predicted average
count=0
last_expiry = str(valid[0][3])
avg = []
amt=0
for i in range(len(valid)):
   curr_expiry = str(valid[i][3])
   if curr_expiry!=last_expiry:
      last_expiry=curr_expiry
      avg.append(round(amt/count,2))
      amt=int(valid[i][1])
      count=0
   else:
      amt=amt+int(valid[i][1])
   count=count+1
# print("Predicted average")
# print(avg)

#actual average
# print("actual average")
count=0
last_expiry = str(valid[0][3])
avg_a = []
amt=0
for i in range(len(valid)):
   curr_expiry = str(valid[i][3])
   if curr_expiry!=last_expiry:
      last_expiry=curr_expiry
      avg_a.append(round(amt/count,2))
      amt=int(valid[i][0])
      count=0
   else:
      amt=amt+int(valid[i][0])
   count=count+1
# print(avg_a)


# count=0
# counter=0
# sell = 0
# buy = 0
# total_selling_price = 0
# total_buying_price = 0
# last_expiry = str(valid[0][3])
# for i in range(len(valid)):
#     curr_expiry = str(valid[i][3])
#     print(valid[i][0],valid[i][1])
#     if curr_expiry!=last_expiry:
#         last_expiry=curr_expiry
#         counter=counter+1
#         print(counter)
#         break
#     if counter>=len(avg):
#         counter=len(avg)-1
#     if (int(valid[i][1])>avg[counter]):

#         plt.plot(valid[i][2], int(valid[i][1]), marker='v', color="red")
#         #  print("On " + str(valid[i][2]) + " Sell signal")
        
#         sell=sell+1
#         total_selling_price=total_selling_price+round(abs(avg[counter]-int(valid[i][1])),2)*int(valid[i][4])
#     else:
#         plt.plot(valid[i][2], int(valid[i][1]), marker='v', color="green")
        
#         buy=buy+1
#         total_buying_price=total_buying_price+round(avg[counter]-int(valid[i][1]),2)*int(valid[i][4])


# print("Total number of selling : "+str(sell) +" total selling price is ₹"+str(total_selling_price))
# print("Total number of buying : "+str(buy) +" total buying price is ₹"+str(total_buying_price))



plt.savefig("result.png")

trade,sell_a,sell_p,buy_a,buy_p=0,0,0,0,0
count,counter,sell,buy=0,0,0,0
long,short=0,0
profit = 0
monthly_profit = 0
last_expiry = str(valid[0][3])

for i in range(len(valid)):
    curr_expiry = str(valid[i][3])
    if curr_expiry!=last_expiry:
        last_expiry=curr_expiry
        counter=counter+1
        profit=profit+monthly_profit
        if monthly_profit<0:
            print(str(counter)+"st Expiry loss " + str(monthly_profit*int(valid[i][-1])))
        else:
            print(str(counter)+"st Expiry profit " + str(monthly_profit*int(valid[i][-1])))
        monthly_profit=0
    if counter>=len(avg):
        counter=len(avg)-1
    if (int(valid[i][1])>avg[counter] and trade==0):
        sell_a=int(valid[i][0])
        sell_p=int(valid[i][1])
        sell=1
        trade=1
        long=long+1
    elif(int(valid[i][1])<avg[counter] and trade==0 ):
        buy_a=int(valid[i][0])
        buy_p=int(valid[i][1])
        buy=1
        trade=1
        short=short+1
    elif(int(valid[i][1])<avg[counter] and trade==1 and sell==1):
        monthly_profit=monthly_profit + (sell_a-int(valid[i][0]))
        trade,sell,sell_p,sell_a=0,0,0,0
    elif(int(valid[i][1])>avg[counter] and trade==1 and buy==1):
        monthly_profit=monthly_profit + (int(valid[i][0])-buy_a)
        trade,buy,buy_p,buy_a=0,0,0,0

if profit<0:
    print("Total Loss : " + str(profit*int(valid[i][-1])))
else:
    print("Total Loss : " + str(profit*int(valid[i][-1])))

print("No of Long trade : " + str(long))
print("No of short trade : " + str(short))