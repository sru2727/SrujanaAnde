#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


# In[29]:


data=pd.read_csv("Clean_Dataset.csv")


# In[30]:


data.head()


# In[31]:


data.shape


# In[32]:


data.columns


# In[33]:


data.isnull().sum()


# In[34]:


data.describe()


# In[35]:


data.info()


# In[36]:


data["Travelling_Cities"] = data["source_city"] +'-'+ data["destination_city"]


# In[37]:


data['Travelling_Cities'] = data.Travelling_Cities.replace('Delhi-Mumbai','Mumbai-Delhi')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Delhi-Bangalore','Bangalore-Delhi')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Bangalore-Mumbai','Mumbai-Bangalore')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Mumbai-Kolkata','Kolkata-Mumbai')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Delhi-Kolkata','Kolkata-Delhi')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Delhi-Chennai','Chennai-Delhi')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Mumbai-Hyderabad','Hyderabad-Mumbai')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Mumbai-Chennai','Chennai-Mumbai')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Bangalore-Kolkata','Kolkata-Bangalore')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Delhi-Hyderabad','Hyderabad-Delhi')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Bangalore-Hyderabad','Hyderabad-Bangalore')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Hyderabad-Kolkata','Kolkata-Hyderabad')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Chennai-Kolkata','Kolkata-Chennai')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Chennai-Bangalore','Bangalore-Chennai')
data['Travelling_Cities'] = data.Travelling_Cities.replace('Hyderabad-Chennai','Chennai-Hyderabad')


# In[38]:


# Adding Distance column
data['distance'] = 0
data.loc[data['Travelling_Cities'] == 'Mumbai-Delhi', 'distance' ] = 1148
data.loc[data['Travelling_Cities'] == 'Bangalore-Delhi', 'distance' ] = 1740
data.loc[data['Travelling_Cities'] == 'Mumbai-Bangalore', 'distance'] = 842
data.loc[data['Travelling_Cities'] == 'Kolkata-Mumbai', 'distance' ] = 1652
data.loc[data['Travelling_Cities'] == 'Kolkata-Delhi', 'distance' ] = 1305
data.loc[data['Travelling_Cities'] == 'Chennai-Delhi', 'distance' ] = 1760
data.loc[data['Travelling_Cities'] == 'Hyderabad-Mumbai', 'distance'] = 617
data.loc[data['Travelling_Cities'] == 'Kolkata-Bangalore','distance'] = 1560
data.loc[data['Travelling_Cities'] == 'Chennai-Mumbai', 'distance' ] = 1028
data.loc[data['Travelling_Cities'] == 'Hyderabad-Delhi', 'distance' ] = 1253
data.loc[data['Travelling_Cities'] == 'Hyderabad-Bangalore','distance'] = 503
data.loc[data['Travelling_Cities'] == 'Kolkata-Hyderabad', 'distance'] = 1180
data.loc[data['Travelling_Cities'] == 'Kolkata-Chennai', 'distance' ] = 1366
data.loc[data['Travelling_Cities'] == 'Bangalore-Chennai', 'distance'] = 284
data.loc[data['Travelling_Cities'] == 'Chennai-Hyderabad', 'distance'] = 521


# In[39]:


data['price_per_100km']=round(data['price']/data['distance']*100,2) # price_per_100km
data['log_price'] = np.log(data['price']) # log Transformation of price column
data = data.drop(columns = ['Unnamed: 0', 'flight' ],axis=1)
data.columns = data.columns.str.replace('class','Ticket_Cat')
data.sample(n = 5, replace = False)


# In[40]:


data.skew()


# ### From the above skew measure we can see that the price and price_per_100km these columns are skewed

# In[41]:


#Checking outilers in the price column


# In[42]:


sns.boxplot(data["price"])


# In[43]:


sns.distplot(data["price"])


# In[44]:


data["price"].hist()


# ### From above plots and skew measure we can see that our price column is positively skewed.

# ### Also from boxplot we can see that price column contains outliers

# # checking distribution of price

# In[18]:


plt.figure(figsize=(10,5))
sns.histplot(data['price'])
plt.title('Histogram of prices',fontsize=18)


# # Handling outliers

# ## Here our price column distribution is positively skewed so, we need to use IQR method to impute outliers 

# In[19]:


IQR=data["price"].quantile(0.75)-data["price"].quantile(0.25)
IQR


# In[20]:


ub=data["price"].quantile(0.75)+(3*IQR)
lb=data["price"].quantile(0.25)-(3*IQR)
ub,lb


# In[21]:


data["price"].describe()


# In[22]:


155735-123071


# In[23]:


data.loc[data["price"]>155735,"price"]=155735


# In[24]:


sns.distplot(data["price"])


# ### Here if we can use 3IQR then their are not much otliers present in the price column

# In[25]:


# 


# In[45]:


data["log_price"].hist()


# In[ ]:


data['log_price'] = np.log(data['price'])


# In[46]:


data["log_price"].skew()


# #### Extract Dataframe for economy and business class

# In[47]:


df_econ = data[data['Ticket_Cat']=='Economy']
df_buss = data[data['Ticket_Cat']=='Business']


# In[ ]:


# data['departure_time']=data['departure_time'].map('{} departure'.format)
# data['arrival_time']=data['arrival_time'].map('{} arrival'.format)


# In[48]:


data.head()


# In[49]:


pct_airline = data.airline.value_counts()/data.airline.value_counts().sum()*100
plt.figure(figsize=(8,5))
plt1=pct_airline.plot.barh(edgecolor = 'black')
plt.title('Airlines',fontsize=18)
plt.xlabel('Percentage')
plt.ylabel('Airline')
for i in range(len(pct_airline)):
    plt.text(pct_airline[i]+.25,i,str(round(pct_airline,2)[i])+'%',va='center',color = 'black')


# #### Maximum number of flight tickets are of Vistara Airline (42.6%) and minimum of Spicejet (3.0%).

# In[50]:


pct_dep = data['departure_time'].value_counts()/data['departure_time'].value_counts().sum()*100
pct_arr = data['arrival_time'].value_counts()/data['arrival_time'].value_counts().sum()*100
plt.figure(figsize=(8,5))
plt1=pct_dep.plot.barh(edgecolor = 'black')
plt.title('Departure time',fontsize=18)
plt.xlabel('Percentage')
plt.ylabel('Departure time')
for i in range(len(pct_dep)):
    plt.text(pct_dep[i]+.25,i,str(round(pct_dep,2)[i])+'%',va='center',color = 'black')


# #### About 46% of the flights have Departure time Morning and Early Morning while just 0.44% have Late night departure. Thus,  Most of people prefer to start the journey in the Morning.

# In[51]:


plt.figure(figsize=(8,5))
plt1=pct_arr.plot.barh(edgecolor = 'black')
plt.title('Arrival Time',fontsize=18)
plt.xlabel('Percentage')
plt.ylabel('Arrival time')
for i in range(len(pct_arr)):
    plt.text(pct_arr[i]+.25,i,str(round(pct_arr,2)[i])+'%',va='center',color = 'black')


# #### About 56% of the flights have Arrival time Evening and Night while just 4.66% flights have Late night arrival. Most of people prefer to reach destination by Night.

# In[52]:


pct_stops = data['stops'].value_counts()/data['stops'].value_counts().sum()*100
plt1=pct_stops.plot.barh(edgecolor = 'black')
plt.title('Stops',fontsize=18)
plt.xlabel('Percentage')
plt.ylabel('No. of Stops')
for i in range(len(pct_stops)):
    plt.text(pct_stops[i]+1,i,str(round(pct_stops,2)[i])+'%',va='center',color = 'black')


# #### About 83.58% of flights takes single stop and just 4.43% takes more than one stops between the source and destination. Thus, people prefers to take single stop before reaching to the destination.

# In[53]:


pct_Ticket_Cat = data['Ticket_Cat'].value_counts()/data['Ticket_Cat'].value_counts().sum()*100
plt1=pct_Ticket_Cat.plot.barh(edgecolor = 'black')
plt.title('Class',fontsize=18)
plt.xlabel('Percentage')
plt.ylabel('Class')
for i in range(len(pct_Ticket_Cat)):
    plt.text(pct_Ticket_Cat[i]+1,i,str(round(pct_Ticket_Cat,2)[i])+'%',va='center',color = 'black')


# #### There are 68.85% of tickets are of Economy class while 31.15% are of Business class

# In[54]:


pct_routes = data['Travelling_Cities'].value_counts()/data['Travelling_Cities'].value_counts().sum()*100
plt.figure(figsize=(12,8))
plt1=pct_routes.plot.barh(edgecolor = 'black')
plt.title('Travelling Cities',fontsize=18)
plt.xlabel('Percentage')
plt.ylabel('Travelling Cities')
for i in range(len(pct_routes)):
    plt.text(pct_routes[i]+.25,i,str(round(pct_routes,2)[i])+'%',va='center',color = 'black')


# #### Number of flights is maximum between Mumbai and Delhi (10.03%) and minimum between Chennai and Hyderabad (4.16%)

# In[55]:


plt.figure(figsize=(14,6))
sns.boxplot(x='airline',y='log_price',hue='Ticket_Cat',data=data.sort_values('price',ascending=False))
plt.title('Boxplot of log Prices per Airline and Class',fontsize=18)


# #### There are only two airlines have business class and is much more expensive than Economy class.The median prices of Vistara Airline is higher as compared with others, while Air Asiahas lower. Also variation in Economy is higher than business class

# In[56]:


plt.figure(figsize=(12,6))
sns.catplot(x='Ticket_Cat',y='price',data=data.sort_values('price',ascending=False),kind='boxen')
plt.title('Prices in Business and Economy Class')
plt.show()


# #### The prices of Business Class is much higher and have higher variation than Economy Class. Also, the distribution of Price in  Economy class is highly positively skewed as compared to the Business class

# In[57]:


plt.figure(figsize=(10,5))
sns.histplot(data['log_price'])
plt.title('Histogram of log prices',fontsize=18)


# #### The distribution of log prices looks like in a two parts, i.e. it is of mixture of business and economy class prices.
# 

# In[58]:


plt.figure(figsize=(15,5))
sns.boxplot(x='Travelling_Cities',y='price',notch= True,data=data).tick_params(axis='x', rotation=45)


# #### The median price is higher for the flights between the Hyderabad and Bangalore. The disribution of price is highly positively skewed for every routes.

# In[59]:


plt.figure(figsize=(10,5))
sns.boxplot(x='departure_time',notch= True,y='price',data=data).tick_params(axis='x', rotation=45)


# #### The distribution of Late_night departure price has minimum variation and have minimum price, i.e. Prices for tickets doesn’t much changes for the  late night flights
# 

# In[60]:


plt.figure(figsize=(10,5))
sns.boxplot(x='arrival_time',notch= True,y='price',data=data).tick_params(axis='x', rotation=45)


# #### The distribution of price of tickets has very less variation for the flights which have late night or early morning arrival time. Also median price is minimum for Late night arrival flights.

# ### Q) How is the price affected when tickets are bought in just 1 or 2 days before departure?
# 

# In[61]:


plt.rcParams['figure.figsize'] = (12,8)
plt.title('Price in Economy Class against Days before departure')
sns.barplot(x='days_left',y= 'price', data=df_econ)
plt.axhline(y=5700)


# In[62]:


plt.rcParams['figure.figsize'] = (12,8)
plt.title('Price in Business Class against Days before departure')
sns.barplot(x='days_left',y= 'price', data=df_buss)
plt.axhline(y=55000)


# #### Here, We can see that ticket prices on an average are high when there are few days left to depature and are less expensive with more days till depature.  As the mean cost slopes downward as the number of days increases. The variation in distribution of mean price is higher in Economy Class. One has to buy minimum 15-18 days before departure to get tickets at lower price in economy class. While in business class, one has to buy just 6 days before

# In[63]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
corr = df_buss.drop(columns = ['price_per_100km','price'],axis=1).corr()
sns.heatmap(corr, annot=True, square=True)
plt.yticks(rotation=0)
plt.title('Correlation Matrix of Business Class',fontsize = 18)
plt.subplot(1,2,2)
corr = df_econ.drop(columns = ['price_per_100km','price'],axis=1).corr()
sns.heatmap(corr, annot=True, square=True)
plt.yticks(rotation=0)
plt.title('Correlation Matrix of Economy Class',fontsize = 18)
plt.show()


# # Fitting Multiple linear regression

# In[65]:


Predictors = [['airline','departure_time','stops','arrival_time','Travelling_Cities','Ticket_Cat','duration','days_left']]
cat_columns = ['airline','departure_time','stops','arrival_time','Travelling_Cities','Ticket_Cat']
num_columns = ['duration','days_left']


# In[67]:


X = data[['airline','departure_time','stops','arrival_time','Travelling_Cities','Ticket_Cat','duration','days_left']]
Y = data['log_price']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state = 15)
df_train = pd.concat([x_train,y_train],axis = 1)


# In[71]:


# defining accuracy measures
def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
# def evaluate(true, predicted):
#     mae = metrics.mean_absolute_error(true, predicted)
#     mse = metrics.mean_squared_error(true, predicted)
#     rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
#     r2_square = metrics.r2_score(true, predicted)
#     return mae, mse, rmse, r2_square


# In[69]:


from statsmodels.formula.api import ols
fit = ols('log_price ~C(Travelling_Cities)+C(airline)+C(departure_time)+C(stops)+C(arrival_time)+C(Ticket_Cat)+(duration)+(days_left)', data=df_train).fit()
fit.summary()


# In[72]:


test_pred = fit.predict(x_test)
train_pred =fit.predict(x_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
results_df = pd.DataFrame(data=[["Decision Tree Regression",*evaluate(y_test, test_pred)]],columns=['Model','MAE','MSE', 'RMSE','R2 Square'])


# In[73]:


X = data[['airline','departure_time','stops','arrival_time','Travelling_Cities','Ticket_Cat','duration','days_left']]
Y1 = data['price']
x1_train,x1_test,y1_train,y1_test = train_test_split(X,Y1,test_size=0.25,random_state = 15)
df_train1 = pd.concat([x1_train,y1_train],axis = 1)


# In[74]:


from statsmodels.formula.api import ols
fit = ols('price ~C(Travelling_Cities)+C(airline)+C(departure_time)+C(stops)+C(arrival_time)+C(Ticket_Cat)+(duration)+(days_left)', data=df_train1).fit()
fit.summary()


# In[75]:


data.head()


# In[76]:


data.stops.value_counts()


# In[77]:


data.stops.replace({"zero":0, "one":1, "two_or_more":2,},inplace=True)


# In[78]:


data.head()


# In[79]:


Predictors = ['stops','distance','Duration','days_left']
X =data[['stops','distance','duration','days_left']]
Y =data['log_price']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=15)
l1 = LinearRegression()
output1 = l1.fit(x_train,y_train)
finaloutput1 =l1.predict(x_test)
finaloutput1


# In[80]:


r2_score(y_test,finaloutput1)


# In[ ]:




