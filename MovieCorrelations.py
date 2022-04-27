#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import libraries

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) #Adjusts the configuration of the plots we will create

# Read in the data

df = pd.read_csv('/Users/beckysu/Desktop/DATA PORTFOLIO/PROJ 3/movies.csv')


# In[4]:


# Look at data
df.head()


# In[5]:


# Check for missing data

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[6]:


# Drop null values
df = df.dropna()


# In[7]:


# Double check after dropna

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[8]:


df.dtypes


# In[9]:


# Change data type

df = pd.DataFrame(df)
df = df.astype({"budget":"int","votes":"int", "gross":"int", "runtime":"int"})


# In[30]:


# Confirm data type change
df.head()


# In[11]:


# Create correct Year column
df['yearcorrect'] = df['released'].str.extract(pat = '([0-9]{4})').astype(int)
df


# In[12]:


df = df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[13]:


# Setting data view to output all data into a scrollable table instead of viewing head
pd.set_option('display.max_rows', None)


# In[14]:


# Check for any duplicates

df.duplicated()


# In[29]:


df.head()


# In[16]:


# Scatter plot with budget vs gross

plt.scatter(x=df['budget'], y=df['gross'])

# Label plot
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget for Film')
plt.ylabel('Gross Earnings')

plt.show()


# In[17]:


df.head()


# In[ ]:





# In[18]:


# Plot budget vs gross using seaborn

sns.regplot(x='budget', y='gross', data=df, scatter_kws={'color': 'red'}, line_kws={'color':'blue'})


# In[19]:


# Correlation
df.corr()


# In[ ]:


# High correlation between budget and gross


# In[20]:


correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()


# In[21]:


# Looks at Company
df.head()


# In[28]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'): 
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized.head()


# In[23]:


correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[24]:


df_numerized.corr()


# In[25]:


correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs


# In[26]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[27]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]
high_corr


# In[ ]:


# Votes and budget have the highest correlation to gross earnings

