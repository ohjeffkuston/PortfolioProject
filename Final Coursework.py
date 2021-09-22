#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#importing dataset
OR = pd.read_excel("C:/Users/ohjef/Downloads/Online Retail.xlsx")

#Checking dataset
OR.head(5)


# In[3]:


print(OR.shape)


# In[4]:


print(OR.sample(5))


# In[5]:


# describing the data
print(OR.describe())


# In[6]:


# taking out information about the data
print(OR.info())


# In[7]:


# checking the data-types of the data
print(OR.dtypes)


# In[8]:


# checking if there is any NULL values present in the data
print(OR.isnull().sum())


# In[9]:


#Removing the Null values from the data.
OR = OR[pd.notnull(OR['CustomerID'])]
print(OR.isnull().sum())


# In[10]:


#Removing the negative values from UnitPrice and Quantity
OR = OR[OR.Quantity > 0]
OR = OR[OR.UnitPrice > 0]
OR.describe()


# In[11]:


# visualizing the unitprice
sns.histplot(OR['UnitPrice'], color = 'darkred')
plt.title('Distribution of Unit price', fontsize = 10)
plt.xlabel('Different Unit Price for different items')
plt.ylabel('count')
plt.show()


# In[12]:


# checking the different values for country in the dataset
OR['Country'].value_counts().head(20).plot.bar(figsize = (20, 8))
plt.title('Top 20 Countries having Online Retail Market', fontsize = 20)
plt.xlabel('Names of Countries')
plt.ylabel('Count')
plt.show()

OR['Country'].value_counts().tail(20).plot.bar(figsize = (20, 8))
plt.title('Bottom 20 Countries having Online Retail Market', fontsize = 20)
plt.xlabel('Names of Countries')
plt.ylabel('Count')
plt.show()


# In[13]:


# checking how many quantity of products have been sold online from each country
OR['Quantity'].groupby(OR['Country']).agg('sum')


# In[14]:


OR['InvoiceDate'] = pd.to_datetime(OR['InvoiceDate'])
OR['InvoiceYearMonth'] = OR['InvoiceDate'].map(lambda date: 100*date.year + date.month)
OR['Date'] = OR['InvoiceDate'].dt.strftime('%Y-%m')


# In[15]:


OR_agg = OR.groupby("Date").Quantity.sum()
OR_agg


# In[16]:


#converting series to dataframe and resetting index, and changing the column name to 'Orders'
OR_agg=pd.DataFrame(OR_agg)
OR_agg=OR_agg.reset_index()
OR_agg.head()


# In[17]:


def plot_OR(OR, x, y, title="", xlabel='Date', ylabel='Orders', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.plot(x, y, color='tab:Blue', marker='o')
    plt.show()


# In[18]:


plot_OR(OR_agg, x=OR_agg.Date, y=OR_agg.Quantity,title='Orders in 2011')


# In[19]:


#Revenue = Order Count * Average Revenue per Order
OR['Revenue'] = OR['Quantity']*OR['UnitPrice']


# In[20]:


sns.boxplot(x=OR['Quantity'])


# In[21]:


OR.head()


# In[22]:


#Monthly Revenue
OR_revenue = OR.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()
OR_revenue.tail()


# In[3]:


# basket analysis
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[4]:


#importing dataset
OR = pd.read_excel("C:/Users/ohjef/Downloads/Online Retail.xlsx")


# In[5]:


OR.head()


# In[6]:


#some of the descriptions have spaces that need to be removed. 
#We’ll also drop the rows that don’t have invoice numbers and remove the credit transactions 
#(those with invoice numbers containing C).
OR['Description'] = OR['Description'].str.strip()
OR.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
OR['InvoiceNo'] = OR['InvoiceNo'].astype('str')
OR = OR[~OR['InvoiceNo'].str.contains('C')]
OR.head()


# In[7]:


#Sales for France (to keep the data small)

#For each invoice , the count of each item is calculated,


basket = (OR[OR['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
basket.head()


# In[8]:


#any positive values are converted to a 1 and anything less the 0 is set to 0.
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)
basket_sets.head()


# In[9]:


frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
frequent_itemsets.head()


# In[10]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()


# In[11]:


rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]


# In[ ]:


# The support value for the first rule is 0.003. This number is calculated by dividing the number of transactions containing ‘avocado,’ ‘spaghetti,’ and ‘milk’ by the total number of transactions.

# The confidence level for the rule is 0.416, which shows that out of all the transactions that contain both ‘avocado’ and ‘spaghetti’, 41.6 percent contain ‘milk’ too.

# The lift of 1.241 tells us that ‘milk’ is 1.241 times more likely to be bought by the customers who buy both ‘avocado’ and ‘spaghetti’ compared to the default likelihood sale of ‘milk.’

