# Exploratory data analysis 

# Import of packages and the preprocessed data set
import pandas as pd
import numpy as np
from datetime import datetime
import collections
import glob
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

#Import the sessions data set
path = r'File path'
df = pd.read_csv(path, index_col=None, header=0)

#Import the event data set
path = r'File path'
dfEvent = pd.read_csv(path, index_col=None, header=0)
print("Data loaded")

# Only analyze the training set
df = df[~df['set'].isnull()]
df = df[df['set']=="train"]
dfEvent = dfEvent[~dfEvent['set'].isnull()]
dfEvent = dfEvent[dfEvent['set']=="train"]

print("Number of sessions:",len(df))
print("Number of events:", len(dfEvent))


#################### Convert the data types - session dataframe
# Convert the session length and max time between two events to seconds
df.session_length = pd.DataFrame(df.session_length.map(lambda x: pd.to_timedelta(x).seconds)).values
df.max_time_between_two_events = pd.DataFrame(df.max_time_between_two_events.map(lambda x: pd.to_timedelta(x).seconds)).values
df = df.astype({"user_id": 'category',"user_session_id_new":'category',"day":'int64',"weekDay":'category',"dayTime":'category',"PurchaseSession":'category', "session_length":'int64',"max_time_between_two_events":'int64'})

#################### Convert the data types - event dataframe
dfEvent= dfEvent.astype({"user_id": 'category',"user_session_id_new":'category',"day":'int64',"weekDay":'category',"dayTime":'category', "product_id":'category',"category_code":'category',"category_id":'category',"brand":"category"})
print("Converted")

# Customization of data types
#################### Convert the data types - session dataframe
# Convert the session length and max time between two events to seconds
df.session_length = pd.DataFrame(df.session_length.map(lambda x: pd.to_timedelta(x).seconds)).values
df.max_time_between_two_events = pd.DataFrame(df.max_time_between_two_events.map(lambda x: pd.to_timedelta(x).seconds)).values
df = df.astype({"user_id": 'category',"user_session_id_new":'category',"day":'int64',"weekDay":'category',"dayTime":'category',"PurchaseSession":'category', "session_length":'int64',"max_time_between_two_events":'int64'})

#################### Convert the data types - event dataframe
dfEvent= dfEvent.astype({"user_id": 'category',"user_session_id_new":'category',"day":'int64',"weekDay":'category',"dayTime":'category', "product_id":'category',"category_code":'category',"category_id":'category',"brand":"category"})
print("Converted")

df.dtypes


# Data analysis
# Analysis of the feature daytime
df.dayTime.value_counts().plot(kind="bar", figsize=(10,5))
plt.title("Number of events by daytime")
plt.ylabel("Number of events")
plt.xlabel("Daytime");


# Investigation of the event dataframe
# Parent product categories
dfEvent.parentProductCategory.value_counts().nlargest(40).plot(kind="bar", figsize=(10,5))
plt.title("Number of events by categories")
plt.ylabel("Number of events")
plt.xlabel("Parent category");


# Secondary product categories
dfEvent.secondaryProductCategory.value_counts().nlargest(40).plot(kind="bar", figsize=(10,5))
plt.title("Number of events by categories")
plt.ylabel("Number of events")
plt.xlabel("Secondary category");


# Third product categories
dfEvent.thirdProductCategory.value_counts().nlargest(40).plot(kind="bar", figsize=(10,5))
plt.title("Number of events by categories")
plt.ylabel("Number of events")
plt.xlabel("Third category");

# Brands
dfEvent.brand.value_counts().nlargest(40).plot(kind="bar", figsize=(10,5))
plt.title("Number of events by brands")
plt.ylabel("Number of events")
plt.xlabel("Brand");



# Histogram of the total number of events (view, cart) per session
# Only sessions with less than 15 events
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_palette("viridis")
sns.set_theme(style="whitegrid")
df15Events = df
df15Events =df15Events[df15Events["number_of_total_events_per_session_view_cart"]<16]
df15Events = df15Events.astype({"number_of_total_events_per_session_view_cart":'category'})
sns.countplot(data= df15Events, x = df15Events["number_of_total_events_per_session_view_cart"], palette= "viridis")
plt.xlabel("Number of total events per session")
plt.ylabel("Count")
plt.title("Number of total events per session")



# Histogram for the number of views per session
# Only sessions with less than 15 views
sns.set(rc={'figure.figsize':(11.7,8.27)})
df15Views = df
df15Views = df15Views[df15Views["number_of_views"]<16]
df15Views  = df15Views.astype({"number_of_views":'category'})
sns.set_theme(style="whitegrid")
sns.countplot(data= df15Views, x = df15Views["number_of_views"], palette= "viridis")
plt.ylabel("Count")
plt.xlabel("Number of views per session")
plt.title("Number of views per session")


# Number of purchases per day over time
df['event_time']= pd.to_datetime(df['event_time'], infer_datetime_format=True).dt.tz_localize(None)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_palette("viridis")
sns.set_theme(style="whitegrid")
NumberOfPurchasesPerDay = df["event_time"].groupby(df["event_time"].dt.floor('d')).size().reset_index(name='count')
x = NumberOfPurchasesPerDay["event_time"]
y = NumberOfPurchasesPerDay["count"]
plt.xlabel("Time")
plt.ylabel("Number of purchases")
plt.title("Number of purchases per day over time")

# Histogram of the time of all events
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_theme(style="whitegrid")
df = df.astype({"hour": 'category'})
hour = df["hour"].cat.as_ordered()
sns.displot(dfEvent, x=hour, palette= "viridis",hue="event_type", hue_order = ['purchase', 'cart','view'], multiple="stack")#, bins=10)
plt.xlabel("Hour of the day")
plt.ylabel("Count of events")
#plt.ticklabel_format(style='plain')
#plt.title("Number of events by hours")
plt.savefig('File path', format='eps')
df = df.astype({"hour": 'int'})


# Density plot of the session length
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_theme(style="whitegrid")
seconds = df.session_length
sns.displot(df, x=seconds, kind='kde',color='#46327e')
#plt.xlabel("Session length in seconds")
#plt.ylabel("Density")
#plt.title("Density of the session length in seconds")
plt.xlim(0, 4500)


# Density plot of the maximal time between two events
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_theme(style="whitegrid")
maxMinutesBetween = df.max_time_between_two_events
sns.displot(df, x=maxMinutesBetween, kind='kde',color='#46327e')
#plt.xlabel("Maximal time between two events of a session in seconds")
#plt.ylabel("Density")
#plt.title("Density of the maximal time between two events of a session in seconds")
plt.xlim(0, 2500)



# Density plot of the product prices
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_theme(style="whitegrid")
products = dfEvent.copy(deep=True).drop_duplicates("product_id")
price = products.price
sns.displot(products, x=price, kind='kde',color='#46327e')
plt.xlim(0, 1000)


# Density plot of the average value of views
sns.displot(df, x="average_value_of_views", kind='kde',color='#46327e')
plt.xlim(0, 1000)


# Create a heatmap of correlations for all numerical features
df = df.drop(columns=["day","hour","month"])
df = df.select_dtypes(include=np.number)
correlationMatrix = df.corr()
plt.subplots(figsize=(35,35))
sns.heatmap(correlationMatrix, annot = True, cmap= 'coolwarm')
plt.show()
