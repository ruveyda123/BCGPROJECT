#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(color_codes=True)


# # Importing Data

# In[2]:


client_df = pd.read_csv('./client_data.csv')
price_df = pd.read_csv('./price_data.csv')


# In[3]:


client_df.head(3)


# In[4]:


price_df.head(3)


# # Statistical Analysis

# In[5]:


client_df.info()


# In[6]:


price_df.info()


# In[7]:


client_df.describe()


# In[8]:


price_df.describe()


# # Data Visualization

# In[9]:


def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):

    ax = dataframe.plot(
        kind="bar",
        stacked=True,
        figsize=size_,
        rot=rot_,
        title=title_
    )

    annotate_stacked_bars(ax, textsize=14)
    plt.legend(["Retention", "Churn"], loc=legend_)
    plt.ylabel("Company base (%)")
    plt.show()

def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    for p in ax.patches:

        value = str(round(p.get_height(),1))

        if value == '0.0':
            continue
        ax.annotate(
            value,
            ((p.get_x()+ p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
            color=colour,
            size=textsize
        )

def plot_distribution(dataframe, column, ax, bins_=50):

    temp = pd.DataFrame({"Retention": dataframe[dataframe["churn"]==0][column],
    "Churn":dataframe[dataframe["churn"]==1][column]})
    temp[["Retention","Churn"]].plot(kind='hist', bins=bins_, ax=ax, stacked=True)
    ax.set_xlabel(column)
    ax.ticklabel_format(style='plain', axis='x')


# In[10]:


churn = client_df[['id', 'churn']]
churn.columns = ['Companies', 'churn']
churn_total = churn.groupby(churn['churn']).count()
churn_percentage = churn_total / churn_total.sum() * 100
plot_stacked_bars(churn_percentage.transpose(), "Churning status", (5, 5), legend_="lower right")


# In[11]:


churn_total


# In[12]:


churn


# In[13]:


churn_percentage


# In[14]:


churn_percentage.transpose()


# In[15]:


consumption = client_df[['id', 'cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons', 'has_gas', 'churn']]

fig, axs = plt.subplots(nrows=4, figsize=(18, 35))

plot_distribution(consumption, 'cons_12m', axs[0])
plot_distribution(consumption, 'cons_gas_12m', axs[1])
plot_distribution(consumption, 'cons_last_month', axs[2])
plot_distribution(consumption, 'imp_cons', axs[3])


# In[16]:


consumption = client_df[['id', 'cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons', 'has_gas', 'churn']]

fig, axs = plt.subplots(nrows=4, figsize=(18, 35))

sns.boxplot(consumption.cons_12m,ax= axs[0])
sns.boxplot(consumption.cons_gas_12m, ax=axs[1])
sns.boxplot(consumption.cons_last_month, ax=axs[2])
sns.boxplot(consumption.imp_cons, ax=axs[3])


# In[17]:


forecast = client_df[
    [ "forecast_cons_12m",
    "forecast_cons_year","forecast_discount_energy","forecast_meter_rent_12m",
    "forecast_price_energy_off_peak","forecast_price_energy_peak",
    "forecast_price_pow_off_peak","churn"
    ]
]

fig, axs = plt.subplots(nrows=7, figsize=(18,50))

plot_distribution(client_df, "forecast_cons_12m", axs[0])
plot_distribution(client_df, "forecast_cons_year", axs[1])
plot_distribution(client_df, "forecast_discount_energy", axs[2])
plot_distribution(client_df, "forecast_meter_rent_12m", axs[3])
plot_distribution(client_df, "forecast_price_energy_off_peak", axs[4])
plot_distribution(client_df, "forecast_price_energy_peak", axs[5])
plot_distribution(client_df, "forecast_price_pow_off_peak", axs[6])


# In[18]:


margin = client_df[[ 'margin_gross_pow_ele', 'margin_net_pow_ele', 'net_margin','churn']]
margin.head()


# In[19]:


fig, axs = plt.subplots(nrows=3, figsize=(18,50))

plot_distribution(client_df, "margin_gross_pow_ele", axs[0])
plot_distribution(client_df, "margin_net_pow_ele", axs[1])
plot_distribution(client_df, "net_margin", axs[2])


# In[20]:


fig, axs = plt.subplots(nrows=3, figsize=(18, 35))
sns.boxplot(client_df["margin_gross_pow_ele"],ax= axs[0])
sns.boxplot(client_df["margin_net_pow_ele"], ax=axs[1])
sns.boxplot(client_df["net_margin"], ax=axs[2])


# In[21]:


power = client_df[['id', 'pow_max', 'churn']]
fig, axs = plt.subplots(nrows=1, figsize=(18, 10))
plot_distribution(power, 'pow_max', axs)


# In[22]:


power


# In[24]:


corr = client_df.corr(numeric_only=True)
plt.figure(figsize=(16,5))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap='Greens',fmt=".2f")


# Observation:
# 
# 1.cons_last_month and cons_12m is highly correlated
# 2.forecast_cons_year and forecoast_cons_12m is highly correlated
# 3.imp_cons and forecast_cons_year is highly correlated
# 4.forecoast_cons_12m and net_margin is highly correlated
# 
# Findings:
# 
# 1.From the client data analysis on the consumption, forecast, margin and power related columns are done
# 2.Highly positive skews data in almost all areas. This need to be handled properly for doing data modelling
# 3.Analysis respresents that we have around 9.6% of customers have churned
# 
# Suggestions:
# 
# 1.Churning may take place when the competitors have given good offers at the same price
# 2.We can ask customer feedbacks, to check for the suggestions and complaints from their side
# 3.Extra benefits and offers for people who subscribed with the company for a long/specified period of time. This intiative can help to decrease the churn percentage
