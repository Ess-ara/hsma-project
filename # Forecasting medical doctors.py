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

#Simple prophet model
#format the data for prophet model (ds and y)
df_train_prophet = df_train.reset_index() \
    .rename(columns={"Date":"ds",
                     "Fellows":"y"})

#check that the columns names hve chnaged and have been formatted
df_train_prophet.head()

#apply and train the model
%time
model = Prophet()
model.fit(df_train_prophet)

#test the model
df_test_prophet = df_test.reset_index() \
    .rename(columns={"Date":"ds",
                     "Fellows":"y"})

#save as new dataframe
df_test_fcst = model.predict(df_test_prophet)

#model output of predictions which includes trend
df_test_fcst.head()

#plot prophet forecast on gragh
"""
model will show upper and lower count that it believes
is reasonable for forecast of thw future
"""
fig, ax = plt.subplots(figsize=(10, 5))
fig = model.plot(df_test_fcst, ax=ax)
ax.set_title('Prophet Forecast')
plt.show()

"""
the model trends etc can also be analysed and so 
we can plot this
"""
fig = model.plot_components(df_test_fcst)
plt.show()