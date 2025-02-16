#Forecasting medical doctors

#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

#import data
df = pd.read_csv("Member data.csv",index_col=0)
df
df.dtypes

#plot data onto graph
color_pal = sns.color_palette()
df.plot(#style='.',
        figsize=(10, 5),
        color=color_pal[0],
        title="Doctors")
plt.show()

##Time series features - Train/test split
split_date= "01/07/2024"
df_train =df.loc[df.index <= split_date].copy()
df_test = df.loc[df.index > split_date].copy()

#plot train and test to see where we applied split
df_test \
    .rename(columns={"Fellows": "Test set"}) \
    .join(df_train.rename(columns={"Fellows": "Train set"}),
          how='outer') \
    .plot(figsize=(10, 5), title='Doctors', ms=10)
plt.show()

