#Data preprocessing


# Importing
import pandas as pd
import numpy as np
from datetime import datetime
import collections
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split

#Import the data set
path = r'Path file'
# For the electronic store data set
df = pd.read_csv(path, index_col=None, header=0)
print("Data loaded")



# Customization of data types
# Convert the data type of the date to "datetime"
df['event_time']= pd.to_datetime(df['event_time'], infer_datetime_format=True).dt.tz_localize(None)
# Convert the other data types
df = df.astype({'event_type':'category', 'product_id':'category', 'category_id':'category', "category_code":'category', "brand": 'category', "user_id": 'category',"user_session":'category'})
print("Converted")

####################################################################################### Preprocessing
df = df[df['user_session'].notna()]
# Define a user session as all events that are recorded for one user (user id) in a timeframe without 30 minutes of no event (inactivity)
# Create a new column "session" that contains all events of one user in a timeframe and ends after 30 minutes of no event (inactivity)
df['session'] = (df.groupby('user_id')['event_time'].transform(lambda x: x.diff().gt('30Min').cumsum()))
# Create a new user session id: "user_session_id_new"
df["user_session_id_new"] = df["user_id"].astype(str)+df['session'].astype(str)
df["user_session_id_new"].nunique() 
df = df.astype({"user_session_id_new":'category'})
print("Sessions were redefined")

# Removal of sessions
# Count the number of different event types per session
sessionDictonaryNumberOfViews = df[df["event_type"]=="view"].groupby("user_session_id_new").size().to_dict()
sessionDictonaryNumberOfCarts = df[df["event_type"]=="cart"].groupby("user_session_id_new").size().to_dict()
sessionDictonaryNumberOfPurchases = df[df["event_type"]=="purchase"].groupby("user_session_id_new").size().to_dict()

# Remove all sessions that have no view event, but have cart events, purchase events, or cart and purchase events
sessionsToRemove = {k:v for k,v in sessionDictonaryNumberOfViews.items() if v==0}
df = df[~df.user_session_id_new.isin(sessionsToRemove.keys())]
df['user_session_id_new'] = df.user_session_id_new.cat.remove_unused_categories()
# Remove all sessions that have purchase events but no cart events
# sessionsToRemove = {k:v for k,v in sessionDictonaryNumberOfCarts.items() if (v == 0 and sessionDictonaryNumberOfPurchases[k]>0)}
# df = df[~df.user_session_id_new.isin(sessionsToRemove.keys())]

# Additional dataframes
sessionViewDf = df[df["event_type"]== "view"]
sessionCartDf = df[df["event_type"]== "cart"]
sessionPurchaseDf = df[df["event_type"]== "purchase"]
sessionEntriesViewCart = sessionViewDf.append(sessionCartDf)
sessionDictonaryNumberOfTotalEventsPerSession = sessionEntriesViewCart.groupby("user_session_id_new").size()
print("Sessions removed")





# Remove all sessions with less than two events
#NumberOfEventsPerSession = df.user_session_id_new.value_counts()
NumberOfEventsPerSession = df.groupby("user_session_id_new").size()
print("Number of remaining sessions:", df["user_session_id_new"].nunique())
print("Length of df before removing sessions with less than two events:",len(df))
df=df[df.user_session_id_new.isin(sessionDictonaryNumberOfTotalEventsPerSession.index[sessionDictonaryNumberOfTotalEventsPerSession.gt(1)])]
print("Length of df after removing sessions with less than two events:",len(df))
# Remove sessions with more than 100 events
print("Length of df before removing sessions with more than threehundred events:",len(df))
df=df[df.user_session_id_new.isin(NumberOfEventsPerSession.index[NumberOfEventsPerSession.lt(101)])]
print("Length of df after removing sessions with more than 100 events:",len(df))
# Remove unused categories of user_session_id_new
df['user_session_id_new'] = df.user_session_id_new.cat.remove_unused_categories()
df.reset_index(drop=True, inplace=True)
print("Further sessions removed")


# Update session number
df['session'] = (df.groupby('user_id')['event_time'].transform(lambda x: x.diff().gt('30Min').cumsum()))
# Create a new user session id: "user_session_id_new"
df["user_session_id_new"] = df["user_id"].astype(str)+df['session'].astype(str)
df["user_session_id_new"].nunique() 
df = df.astype({"user_session_id_new":'category', "session":'int'})
print("Sessions updated")


# Creation of new features
# Features for the time and for the product category
# Create a column for the day, month, weekDay, hour
df["day"] = df['event_time'].dt.day
df["month"] = df['event_time'].dt.month
df["weekDay"] = df['event_time'].dt.weekday
df["hour"] = df['event_time'].dt.hour
# Create a variable that indicates whether it is a weekday or not (= a weekend day)
df["weekDayOrNot"] = df["weekDay"]<5
# Create the variable dayTime with 6 different categories
df['dayTime'] = (df['event_time'].dt.hour % 24 + 4) // 4
df['dayTime'].replace({1: 'Late Night',
                      2: 'Early Morning',
                      3: 'Morning',
                      4: 'Noon',
                      5: 'Evening',
                      6: 'Night'}, inplace=True)
# Create a new variable from "category_code" for the parent category / product type
df['parentProductCategory'] = df['category_code'].str.split(".").str[0]
# Create a new variable from "category_code" for the secondary category / product type
df['secondaryProductCategory'] = df['category_code'].str.split(".").str[1]
# Create a new variable from "category_code" for the third category / product type
df['thirdProductCategory'] = df['category_code'].str.split(".").str[2]
# Change the data types of the new created variables
df = df.astype({'day':'category', 'month':'category',"weekDay": "category", "hour": "category","dayTime": "category", "parentProductCategory": "category", 'secondaryProductCategory': "category",'thirdProductCategory':"category"})
print("New variables created")

# Replacement of missing values
# Check for which columns there a missing values
df.columns[df.isnull().any()]
# Result: category_code', 'brand', 'parentProductCategory', 'secondaryProductCategory', 'thirdProductCategory'
# # Replace all missing values of the column brand (with "Unknown brand") and of the category columns (with "Unknown category")
df["brand"] = np.where(df["brand"].isnull(),"Unknown brand",df["brand"])
df["category_code"] = np.where(df["category_code"].isnull(),"Unknown category",df["category_code"])
df["parentProductCategory"] = np.where(df["parentProductCategory"].isnull(),"Unknown category",df["parentProductCategory"])
df["secondaryProductCategory"] = np.where(df["secondaryProductCategory"].isnull(),"Unknown category",df["secondaryProductCategory"])
df["thirdProductCategory"] = np.where(df["thirdProductCategory"].isnull(),"Unknown category",df["thirdProductCategory"])
print("Replacement finished")


# Features regarding the time of the session
# Create new variables based on the length of the session
# Calculate first and last timestamp of a session to calculate the "sessionLength"
timeDiff = df.groupby("user_session_id_new")['event_time'].agg(['min','max']).rename(columns={'min':'first','max':'last'})
timeDiff["sessionLength"] = timeDiff["last"] - timeDiff["first"]
timeDiff = timeDiff.reset_index() 
timeDiff = timeDiff.sort_values("user_session_id_new")
print("Calculated the session length")


# Calculate the maximal time between two events of one session
# Use only the values where there are no missing values in the column user session
#dfTime = df[df['user_session'].notna()]
dfTime = df.copy()
dfTime['diff'] = dfTime.sort_values(['user_session_id_new','event_time']).groupby('user_session_id_new')['event_time'].diff()
dfTime.loc[pd.isnull(dfTime['diff']), 'diff'] = pd.Timedelta(seconds=0)
#maxTimeDiff contains the maximal time between two events of one session
maxTimeDiff = dfTime.groupby('user_session_id_new')['diff'].max().to_dict()
print("Calculated the maximal time between two events of one session")




# Calculation of unique views and unique carts
# Update
sessionDictonaryNumberOfViews = df[df["event_type"]=="view"].groupby("user_session_id_new").size().to_dict()
sessionDictonaryNumberOfCarts = df[df["event_type"]=="cart"].groupby("user_session_id_new").size().to_dict()
sessionDictonaryNumberOfPurchases = df[df["event_type"]=="purchase"].groupby("user_session_id_new").size().to_dict()
# Calculate the total events of a session: 
sessionViewDf = df[df["event_type"]== "view"]
sessionCartDf = df[df["event_type"]== "cart"]
sessionEntriesViewCart = sessionViewDf.append(sessionCartDf)
sessionDictonaryNumberOfTotalEventsPerSessionViewAndCart = sessionEntriesViewCart.groupby("user_session_id_new").size().to_dict()
sessionDictonaryNumberOfTotalEventsPerSession = df.groupby("user_session_id_new").size().to_dict()
# Group by user session id new and add all id's of products related to view, cart and purchase events to a dictonary (with duplicates)
sessionDictonaryViewedProducts = df[df["event_type"]=="view"].groupby("user_session_id_new")['product_id'].apply(list).to_dict() 
sessionDictonaryCartedProducts  = df[df["event_type"]=="cart"].groupby("user_session_id_new")['product_id'].apply(list).to_dict()
sessionDictonaryPurchasedProducts = df[df["event_type"]=="purchase"].groupby("user_session_id_new")['product_id'].apply(list).to_dict()
# Calculate unique views and carts
numberOfUniqueViewsPerSession ={}
numberOfUniqueCartsPerSession ={}
#numberOfUniquePurchasesPerSession ={}

for key, number in sessionDictonaryViewedProducts.items():
    numberSet = set(sessionDictonaryViewedProducts[key])
    numberOfUniqueViewsPerSession[key] = len(numberSet)
    if sessionDictonaryNumberOfCarts[key]==0:
        numberOfUniqueCartsPerSession[key] = 0

for key, number in sessionDictonaryCartedProducts.items():
    numberSet = set(sessionDictonaryCartedProducts[key])
    numberOfUniqueCartsPerSession[key] = len(numberSet)
# unique purchases
#for key, number in sessionDictonaryPurchasedProducts.items():
    #numberSet = set(sessionDictonaryPurchasedProducts[key])
    #numberOfUniquePurchasesPerSession[key] = len(numberSet)
print("Calculation finished")





# Calculation of the number of different viewed categories (first, second, third) and creation of the visitor type
# Group by user session id new and add all id's of products related to view, cart and purchase events to a dictonary (with duplicates)
sessionDictonaryViewedCategoriesFirstLevel = df[df["event_type"]=="view"].groupby("user_session_id_new")['parentProductCategory'].apply(list).to_dict() 
sessionDictonaryViewedCategoriesSecondLevel = df[df["event_type"]=="view"].groupby("user_session_id_new")['secondaryProductCategory'].apply(list).to_dict() 
sessionDictonaryViewedCategoriesThirdLevel = df[df["event_type"]=="view"].groupby("user_session_id_new")['thirdProductCategory'].apply(list).to_dict() 

sessionDictonaryCartedCategoriesFirstLevel = df[df["event_type"]=="cart"].groupby("user_session_id_new")['parentProductCategory'].apply(list).to_dict() 
sessionDictonaryCartedCategoriesSecondLevel = df[df["event_type"]=="cart"].groupby("user_session_id_new")['secondaryProductCategory'].apply(list).to_dict() 
sessionDictonaryCartedCategoriesThirdLevel = df[df["event_type"]=="cart"].groupby("user_session_id_new")['thirdProductCategory'].apply(list).to_dict() 

# Calculate unique views and carts
numberOfUniqueViewsPerSessionFirstLevel ={}
numberOfUniqueViewsPerSessionSecondLevel ={}
numberOfUniqueViewsPerSessionThirdLevel ={}

numberOfUniqueCartsPerSessionFirstLevel ={}
numberOfUniqueCartsPerSessionSecondLevel ={}
numberOfUniqueCartsPerSessionThirdLevel ={}

for key, number in sessionDictonaryViewedCategoriesFirstLevel.items():
    #First level
    numberSet = set(sessionDictonaryViewedCategoriesFirstLevel[key])
    numberOfUniqueViewsPerSessionFirstLevel[key] = len(numberSet)
    #Second level
    numberSet = set(sessionDictonaryViewedCategoriesSecondLevel[key])
    numberOfUniqueViewsPerSessionSecondLevel [key] = len(numberSet)
    #Third level
    numberSet = set(sessionDictonaryViewedCategoriesThirdLevel[key])
    numberOfUniqueViewsPerSessionThirdLevel[key] = len(numberSet)
    #Sessions without a cart
    if sessionDictonaryNumberOfCarts[key]==0:
        numberOfUniqueCartsPerSessionFirstLevel[key] = 0
        numberOfUniqueCartsPerSessionSecondLevel[key] = 0
        numberOfUniqueCartsPerSessionThirdLevel[key] = 0

for key, number in sessionDictonaryCartedCategoriesFirstLevel.items():
    #First level
    numberSet = set(sessionDictonaryCartedCategoriesFirstLevel[key])
    numberOfUniqueCartsPerSessionFirstLevel [key] = len(numberSet)
    #Second level
    numberSet = set(sessionDictonaryCartedCategoriesSecondLevel[key])
    numberOfUniqueCartsPerSessionSecondLevel [key] = len(numberSet)
    #Third level
    numberSet = set(sessionDictonaryCartedCategoriesThirdLevel[key])
    numberOfUniqueCartsPerSessionThirdLevel[key] = len(numberSet)
        
# Create a column visitor type (new visitor (true) or not (false))
df["newVisitorOrNot"] = df["session"]== 0
print("Calculation finished")

# Calculation of total value of viewed and carted products
# Group by user session id new and get the total value of views per session and calculate the average value of viewed items
sessionDictonaryTotalValueOfViews = df[df["event_type"]=="view"].groupby("user_session_id_new")['price'].sum().to_dict()
sessionDictonaryAverageValueOfViews = {}
for key in sessionDictonaryTotalValueOfViews:
    if (sessionDictonaryNumberOfViews[key]>0):
        sessionDictonaryAverageValueOfViews[key]=sessionDictonaryTotalValueOfViews[key]/sessionDictonaryNumberOfViews[key]
    else: sessionDictonaryAverageValueOfViews[key] = 0
# Group by user session and get the total value of carts per session and calculate the average value of carted items
sessionDictonaryTotalValueOfCarts = df[df["event_type"]=="cart"].groupby("user_session_id_new")['price'].sum().to_dict()
sessionDictonaryAverageValueOfCarts = {}
for key in sessionDictonaryTotalValueOfCarts:
    if (sessionDictonaryNumberOfCarts[key]>0):
        sessionDictonaryAverageValueOfCarts[key]=sessionDictonaryTotalValueOfCarts[key]/sessionDictonaryNumberOfCarts[key]
    else: 
        sessionDictonaryAverageValueOfCarts[key] = 0
for key, number in sessionDictonaryViewedProducts.items():
    if key not in sessionDictonaryTotalValueOfCarts:
        sessionDictonaryTotalValueOfCarts[key]= 0
        sessionDictonaryAverageValueOfCarts[key] = 0
print("Calculation finished")



# Creation of a dataframe for sessions
# Create a new dataframe that has features on session level
sessionDf = df.copy(deep=True)
sessionDf = sessionDf.drop_duplicates(subset='user_session_id_new', keep="first")
sessionDf = sessionDf.drop(['event_type','product_id','category_id',
 'category_code',
 'brand',
 'price',
 'user_session',
 'session',
 'parentProductCategory',
 'secondaryProductCategory',
 'thirdProductCategory'], axis=1)

sessionDf = sessionDf.sort_values("user_session_id_new")
sessionDf["number_of_views"] = pd.DataFrame(sessionDictonaryNumberOfViews.items()).sort_values(0)[1].values
sessionDf["number_of_carts"] = pd.DataFrame(sessionDictonaryNumberOfCarts.items()).sort_values(0)[1].values
sessionDf["number_of_purchases"] = pd.DataFrame(sessionDictonaryNumberOfPurchases.items()).sort_values(0)[1].values
sessionDf["number_of_total_events_per_session_view_cart"] = pd.DataFrame(sessionDictonaryNumberOfTotalEventsPerSessionViewAndCart.items()).sort_values(0)[1].values
sessionDf["total_value_of_views"] = pd.DataFrame(sessionDictonaryTotalValueOfViews.items()).sort_values(0)[1].values
sessionDf["total_value_of_carts"] = pd.DataFrame(sessionDictonaryTotalValueOfCarts.items()).sort_values(0)[1].values
sessionDf["number_of_unique_views"] = pd.DataFrame(numberOfUniqueViewsPerSession.items()).sort_values(0)[1].values
sessionDf["number_of_unique_carts"] = pd.DataFrame(numberOfUniqueCartsPerSession.items()).sort_values(0)[1].values
sessionDf["number_of_unique_views_first_category_level"] = pd.DataFrame(numberOfUniqueViewsPerSessionFirstLevel.items()).sort_values(0)[1].values
sessionDf["number_of_unique_views_second_category_level"] = pd.DataFrame(numberOfUniqueViewsPerSessionSecondLevel.items()).sort_values(0)[1].values
sessionDf["number_of_unique_views_third_category_level"] = pd.DataFrame(numberOfUniqueViewsPerSessionThirdLevel.items()).sort_values(0)[1].values
sessionDf["number_of_unique_carts_first_category_level"] = pd.DataFrame(numberOfUniqueCartsPerSessionFirstLevel.items()).sort_values(0)[1].values
sessionDf["number_of_unique_carts_second_category_level"] = pd.DataFrame(numberOfUniqueCartsPerSessionSecondLevel.items()).sort_values(0)[1].values
sessionDf["number_of_unique_carts_third_category_level"] = pd.DataFrame(numberOfUniqueCartsPerSessionThirdLevel.items()).sort_values(0)[1].values
sessionDf["session_length"] = timeDiff["sessionLength"].values
sessionDf["max_time_between_two_events"] = pd.DataFrame(maxTimeDiff.items()).sort_values(0)[1].values
sessionDf["average_value_of_views"]= pd.DataFrame(sessionDictonaryAverageValueOfViews.items()).sort_values(0)[1].values
sessionDf["average_value_of_carts"] = pd.DataFrame(sessionDictonaryAverageValueOfCarts.items()).sort_values(0)[1].values
# Create a column that indicates whether it is a "purchase session" or not
sessionDf["PurchaseSession"] = np.multiply(sessionDf["number_of_purchases"]>0,1)
print("Dataframe finished")



# Selection of a subset of sessions from the dataset
# Remove sessions that dont have a view event as their first session event
dfCopy = df.copy(deep=True).sort_values(by='event_time')
sessionsFirst = dfCopy.groupby("user_session_id_new").first().reset_index()
sessionsFirstView = sessionsFirst[sessionsFirst.event_type == "view"]
sessionDfCopy = sessionDf.copy(deep=True)
sessionDfCopy = sessionDfCopy[sessionDfCopy.user_session_id_new.isin(sessionsFirstView.user_session_id_new)]
sessionDfCopy['user_session_id_new'] = sessionDfCopy.user_session_id_new.cat.remove_unused_categories()

X_session = sessionDfCopy.drop("PurchaseSession",axis=1)
y_session = sessionDfCopy.PurchaseSession

# Also remove the sessions without view event from the event dataframe
Xy_event = df.copy(deep=True)
Xy_event = Xy_event[Xy_event.user_session_id_new.isin(sessionsFirstView.user_session_id_new)]
Xy_event['user_session_id_new']= Xy_event.user_session_id_new.cat.remove_unused_categories()

# Assign a label to every row of the event dataframe (every event) (purchase session or not)
Xy_event["PurchaseSession"] = Xy_event.user_session_id_new.copy(deep=True)
Xy_event["PurchaseSession"] = Xy_event["PurchaseSession"].map(sessionDictonaryNumberOfPurchases)
Xy_event.loc[Xy_event["PurchaseSession"] > 0, 'PurchaseSession'] = 1
Xy_event.loc[Xy_event["PurchaseSession"] == 0, 'PurchaseSession'] = 0



# Split in training and test set
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_session, y_session, test_size=50000, train_size=50000, random_state=42, stratify = y_session)


# Indicate training and test set of the session and event dataframe
# session df
sessionDfCopy.loc[sessionDfCopy.user_session_id_new.isin(X_train_s.user_session_id_new), ["set"]] = "train"
sessionDfCopy.loc[sessionDfCopy.user_session_id_new.isin(X_test_s.user_session_id_new), ["set"]] = "test"

# event df
Xy_event.loc[Xy_event.user_session_id_new.isin(X_train_s.user_session_id_new), ["set"]] = "train"
Xy_event.loc[Xy_event.user_session_id_new.isin(X_test_s.user_session_id_new), ["set"]] = "test"


# Saving the processed data sets as a csv file

#######################################################################################
from pathlib import Path
# Save the session dataframe and the event dataframe as csv file
# Save final event dataframe
filepath = Path('File path')
filepath.parent.mkdir(parents=True, exist_ok=True)  
Xy_event.to_csv(filepath, encoding='utf-8', index=False)


# Save final session dataframe
filepath = Path('File path') 
filepath.parent.mkdir(parents=True, exist_ok=True)  
sessionDfCopy.to_csv(filepath, encoding='utf-8', index=False)
print("Files saved")


# Calculation of statistics

# Only calculate the statistics for the train data set
Xy_event_train = Xy_event.loc[Xy_event['set'] == "train"]
sessionDfCopy_train = sessionDfCopy.loc[sessionDfCopy['set'] == "train"]



print("---------------------------------------------")
"""**Calculate some statistics**"""
# First and last date 
print("First date:", min(Xy_event_train["event_time"]),"Last date:", max(Xy_event_train["event_time"]))
# Number of total events
print("Number of total events:", len(Xy_event_train))
# Number of sessions
print("Total Number of sessions:", Xy_event_train["user_session_id_new"].nunique(), len(sessionDfCopy_train))
# Number of users
print("Number of users:", Xy_event_train["user_id"].nunique())
# Number of sessions with only one event
sessionDictonaryNumberOfTotalEventsPerSession = Xy_event_train.groupby("user_session_id_new").size().to_dict()
sessionDictonaryNumberOfTotalEventsPerSession = {key:val for key, val in sessionDictonaryNumberOfTotalEventsPerSession.items() if val != 0}
# Only use the sessions with at least 2 events: len(allSessionsWithAtLeast4Events):
allSessionsWith1Event = { k:v for k, v in sessionDictonaryNumberOfTotalEventsPerSession.items() if v == 1 }
print("Sessions with only one event:",len(allSessionsWith1Event))
print("Sessions with only one event in percent:", len(allSessionsWith1Event)/len(sessionDictonaryNumberOfTotalEventsPerSession)*100, "%")
# Average number of clicks per session:
print("Average number of clicks per session:",sum(sessionDictonaryNumberOfTotalEventsPerSession.values()) / float(len(sessionDictonaryNumberOfTotalEventsPerSession)))
# Percentage of purchase sessions (session with at least one purchase)
sessionDictonaryPurchaseSession = Xy_event_train[Xy_event_train["event_type"]=="purchase"].groupby("user_session_id_new").size().to_dict()
numberOfPurchaseSessions = len({ k:v for k, v in sessionDictonaryPurchaseSession.items() if v > 0 })
print("Number of purchase sessions:",numberOfPurchaseSessions)
print("Share of purchase sessions:", numberOfPurchaseSessions/len(sessionDictonaryNumberOfTotalEventsPerSession)*100,"%")
print("---------------------------------------------")











