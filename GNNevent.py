# GNN - event model

# Importing
import pandas as pd
import numpy as np
from datetime import datetime
import collections
import glob
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import calendar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


#Import the sessions data set
path = r'File path'
# For the electronic store data set
df = pd.read_csv(path, index_col=None, header=0)

#Import the event data set
path = r'File path'
# For the electronic store data set
dfEvent = pd.read_csv(path, index_col=None, header=0)
print("Data loaded")




#df = df[df["number_of_total_events_per_session_view_cart"]>1]
df = df[df["set"].notna()]
dfEvent = dfEvent[dfEvent.user_session_id_new.isin(df.user_session_id_new)]
df = df.reset_index(drop=True)


#################### Convert the data types - session dataframe
# Convert the session length and max time between two events to seconds
df.session_length = pd.DataFrame(df.session_length.map(lambda x: pd.to_timedelta(x).seconds)).values
df.max_time_between_two_events = pd.DataFrame(df.max_time_between_two_events.map(lambda x: pd.to_timedelta(x).seconds)).values
df['week_of_the_month'] = np.nan
df.loc[df['day'] <= 7, 'week_of_the_month'] = 1
df.loc[(df['day'] > 7) & (df['day'] <= 15), 'week_of_the_month'] = 2
df.loc[(df['day'] > 15) & (df['day']<= 22), 'week_of_the_month'] = 3
df.loc[(df['day'] > 22) & (df['day'] <= 31), 'week_of_the_month'] = 4
df['month'] = df['month'].apply(lambda x: calendar.month_name[x])
df['weekDay'] = df['weekDay'].apply(lambda x: calendar.day_name[x])
df = df.drop(columns=["day","hour"])#,"average_value_of_carts",'average_value_of_views'
df = df.astype({"user_session_id_new":'category',"weekDay":'category',"dayTime":'category',"PurchaseSession":'int', "session_length":'int64',"max_time_between_two_events":'int64',"month": 'category','week_of_the_month':"category"})
df = df.drop(["event_time", "user_id", "number_of_purchases"], axis = 1)
df = df.rename(columns={'weekDay': 'week_day', 'weekDayOrNot': 'week_day_or_not', "dayTime":"day_time","newVisitorOrNot":"new_visitor_or_not","PurchaseSession":"purchase_session"})
#################### Convert the data types - event dataframe
dfEvent['event_time']= pd.to_datetime(dfEvent['event_time'], infer_datetime_format=True).dt.tz_localize(None)
dfEvent= dfEvent.astype({"newVisitorOrNot":"int", "user_id": 'category',"day":'category',"month":'category',"user_session_id_new":'category',"day":'int64',"weekDay":'category',"dayTime":'category', "product_id":'category',"category_code":'category',"category_id":'category',"brand":"category","event_type":'category',"user_id":'category',"parentProductCategory":'category',"secondaryProductCategory":'category',"thirdProductCategory":'category'})
dfEvent.weekDayOrNot = dfEvent.weekDayOrNot*1
dfEvent = dfEvent.drop("user_session",axis=1)
print("Converted")


# One hot encoding of categorical columns
categoricalColumns = dfEvent.select_dtypes(['category']).columns.to_list()
#dfEventOneHot = pd.get_dummies(dfEvent, columns = categoricalColumns, drop_first=True)
columnsToEncode = ['event_type','weekDay',"dayTime"]
# Drop all purchase events from the events data frame
dfWithoutPurchaseEvents = dfEvent.copy(deep=True)
dfWithoutPurchaseEvents = dfWithoutPurchaseEvents.loc[~dfWithoutPurchaseEvents['event_type'].isin(['purchase'])]
dfWithoutPurchaseEvents["event_type"] = dfWithoutPurchaseEvents["event_type"].cat.remove_unused_categories()
dfWithoutPurchaseEvents['event_type_orig'] = dfWithoutPurchaseEvents['event_type']
dfWithoutPurchaseEventsOneHot = pd.get_dummies(dfWithoutPurchaseEvents, columns = columnsToEncode)


# Calculate timedelta between events in a session
dfWithoutPurchaseEventsOneHot['sessionTime'] = dfWithoutPurchaseEventsOneHot.sort_values(['user_session_id_new','event_time']).groupby('user_session_id_new')['event_time'].diff().dt.seconds
dfWithoutPurchaseEventsOneHot['sessionTime'] = dfWithoutPurchaseEventsOneHot['sessionTime'].fillna(0)




X_train = dfWithoutPurchaseEventsOneHot[dfWithoutPurchaseEventsOneHot.set=="train"]
X_test = dfWithoutPurchaseEventsOneHot[dfWithoutPurchaseEventsOneHot.set=="test"]
# Oversampling of the training set: 
# Compare the number of purchase and no purchase sessions of the training set and oversample
def oversampling(X_train):
    # calculate an appropriate oversample rate
    sessions = X_train.drop_duplicates(subset=['user_session_id_new']).reset_index(drop=True)
    imbalance_ratio = len(sessions[sessions.PurchaseSession==1])/len(sessions[sessions.PurchaseSession==0])
    oversample_rate = max((0,(int(round(1/imbalance_ratio)/8))))#4 bei electronics, 6 cosmetics
    print("Oversample rate:", oversample_rate)
    # oversample
    X_train_original = X_train.copy(deep=True)
    if oversample_rate>0:
        for i in range(0,oversample_rate):
            X_train_purchase = X_train_original[X_train_original["PurchaseSession"]==1].copy(deep=True)
            X_train_purchase["user_session_id_new"] = X_train_purchase["user_session_id_new"].astype(str)+str("D")+str(i)
            X_train_purchase = X_train_purchase.reset_index(drop=True)
            X_train = X_train.append(X_train_purchase)
        X_train["user_session_id_new"] = X_train["user_session_id_new"].astype("category")
    return X_train



# Balance the training set old
#X_train_balanced = oversampling(X_train)
# Shuffle the rows
#X_train_balanced  = X_train_balanced.sample(frac=1).reset_index(drop=True)

#sessions = X_train.drop_duplicates(subset=['user_session_id_new']).reset_index(drop=True)
#imbalance_ratio = len(sessions[sessions.PurchaseSession==1])/len(sessions[sessions.PurchaseSession==0])
#print("Original class ratio 0:1:",imbalance_ratio)
#sessions = X_train_balanced.drop_duplicates(subset=['user_session_id_new']).reset_index(drop=True)
#imbalance_ratio = len(sessions[sessions.PurchaseSession==1])/len(sessions[sessions.PurchaseSession==0])
#print("Balanced class ratio 0:1:",imbalance_ratio)




# Balance the training set new
# Split X_train in train and validation set
X_train_user_sessions = X_train.drop_duplicates("user_session_id_new")[["user_session_id_new","PurchaseSession"]]
X_train_uid, X_validation_uid = train_test_split(X_train_user_sessions, test_size=0.3, random_state=42, stratify = X_train_user_sessions.PurchaseSession)
X_validation = X_train[X_train.user_session_id_new.isin(X_validation_uid.user_session_id_new.values)]
X_train = X_train[X_train.user_session_id_new.isin(X_train_uid.user_session_id_new.values)]
X_train = X_train.reset_index(drop=True)
X_validation = X_validation.reset_index(drop=True)

# Oversample the train set
X_train_balanced = oversampling(X_train)
# Shuffle the rows
X_train_balanced  = X_train_balanced.sample(frac=1).reset_index(drop=True)

sessions = X_train.drop_duplicates(subset=['user_session_id_new']).reset_index(drop=True)
imbalance_ratio = len(sessions[sessions.PurchaseSession==1])/len(sessions[sessions.PurchaseSession==0])
print("Original class ratio 0:1:",imbalance_ratio)
sessions = X_train_balanced.drop_duplicates(subset=['user_session_id_new']).reset_index(drop=True)
imbalance_ratio = len(sessions[sessions.PurchaseSession==1])/len(sessions[sessions.PurchaseSession==0])
print("Balanced class ratio 0:1:",imbalance_ratio)


# Fit and transform on the training data 
min_max_scaler = preprocessing.MinMaxScaler()
X_train_balanced[['sessionTime', 'price','hour']] = min_max_scaler.fit_transform(X_train_balanced[['sessionTime', 'price','hour']])
# Transform the test data 
X_validation[['sessionTime', 'price','hour']] = min_max_scaler.transform(X_validation[['sessionTime', 'price','hour']])
X_test[['sessionTime', 'price','hour']] = min_max_scaler.transform(X_test[['sessionTime', 'price','hour']])





# GNN event
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from sklearn import preprocessing
from dgl.data.utils import save_graphs

from dgl.data import DGLDataset
class ClickstreamEventDataset(DGLDataset):
    def __init__(self, datasetType):
        self.datasetType = datasetType
        super().__init__(name='synthetic')
        

    def process(self):
        if self.datasetType=="train":
            data = X_train_balanced.copy(deep=True)#_balanced
        elif self.datasetType=="test":
            data = X_test.copy(deep=True)
        elif self.datasetType=="validation":
            data = X_validation.copy(deep=True)
        data.user_session_id_new =data.user_session_id_new.cat.remove_unused_categories()
        grouped = data.sort_values(['event_time'],ascending=True).groupby('user_session_id_new')
        self.graphs = []
        self.labels = []
        for user_session_id_new, group in grouped:
            
            group.reset_index(drop=True, inplace=True)
            # Label encode product id, category id, brand
            product_id_session, unique = pd.factorize(group["product_id"])
            #category_id_session, unique = pd.factorize(group["category_id"])
            #brand_session, unique = pd.factorize(group["brand"])
            

            ############################ Define nodes and edges 
            # Define source and destination nodes
            #src_nodes = group.index.tolist()[0]
            #src_nodes = np.append(src_nodes,group.index.tolist()[:-1])
            src_nodes = group.index.tolist()[:-1]
            src_nodes = np.append(src_nodes,group.index.tolist()[1:])
            
            #dst_nodes = group.index.tolist()[0]
            #dst_nodes = np.append(dst_nodes,group.index.tolist()[1:])
            dst_nodes = group.index.tolist()[1:]
            dst_nodes = np.append(dst_nodes,group.index.tolist()[:-1])
            label = group.PurchaseSession.values[0] 
            
            ############################ Define node level features
            
            
            # Add features
            feature_view = group["event_type_view"].values
            feature_cart = group["event_type_cart"].values
            feature_price = group["price"].values
            #feature_product_id = product_id_session
            #feature_category_id = category_id_session
            #feature_brand = brand_session
            feature_session_time =group["sessionTime"].values
            
            #feature_week_day_or_not = group["weekDayOrNot"].values 
            #feature_day = group["day"].values 
            #feature_hour = group["hour"].values
            
            
            feature_view = torch.DoubleTensor(feature_view).unsqueeze(1)
            feature_cart = torch.DoubleTensor(feature_cart).unsqueeze(1)
            feature_price = torch.DoubleTensor(feature_price).unsqueeze(1)
            #feature_product_id = torch.DoubleTensor(feature_product_id).unsqueeze(1)
            #feature_category_id  = torch.DoubleTensor(feature_category_id).unsqueeze(1)
            #feature_brand  = torch.DoubleTensor(feature_brand).unsqueeze(1)
            feature_session_time = torch.DoubleTensor(feature_session_time).unsqueeze(1)
            #feature_week_day_or_not  = torch.DoubleTensor(feature_week_day_or_not).unsqueeze(1)
            #feature_day = torch.DoubleTensor(feature_day).unsqueeze(1)
            #feature_hour = torch.DoubleTensor(feature_hour).unsqueeze(1)
            

            g = dgl.graph((src_nodes, dst_nodes), num_nodes = len(feature_view))
            #g.ndata['featStack'] = torch.column_stack((feature_view, feature_cart, feature_price, feature_product_id, feature_category_id, feature_brand,feature_session_time, feature_week_day_or_not, feature_day, feature_hour))
            g.ndata['featStack'] = torch.column_stack((feature_view,  feature_cart, feature_price, feature_session_time))
            #g.ndata['featStack'] = torch.column_stack((feature_view, feature_cart ))
            #setattr(g, 'graph_features', sessionFeatStack)
            self.graphs.append(g)
            self.labels.append(label)
        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)    

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
    
    
# Construction of graph datasets
train_dataset = ClickstreamEventDataset("train")
validation_dataset = ClickstreamEventDataset("validation")
test_dataset = ClickstreamEventDataset("test")


#graph, label = dataset[5]
print("Length of training set:",len(train_dataset))
print("Length of validation set:",len(validation_dataset))
print("Length of test set:",len(test_dataset))
#graph.ndata
#print(graph.nodes())
#graph.nodes()
#graph.edges()



from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def calc_auc(predicted_class_proba, labels):
    proba = np.concatenate(predicted_class_proba, axis=0)
    proba_class1 = pd.Series([item[-1] for item in proba])
    labels = np.concatenate(labels, axis=0)
    labels = pd.Series(labels.ravel())
    return roc_auc_score(pd.Series(labels.ravel()), proba_class1)

def calc_metrics(predicted_class, labels):
    predicted_class = np.concatenate(predicted_class, axis=0)
    predicted_class = pd.Series(predicted_class.ravel())
    labels = np.concatenate(labels, axis=0)
    labels = pd.Series(labels.ravel())
    return recall_score(labels, predicted_class),precision_score(labels, predicted_class), f1_score(labels, predicted_class), accuracy_score(labels, predicted_class)


def draw_roc(predicted_class_proba, labels):
    proba = np.concatenate(predicted_class_proba, axis=0)
    proba_class1 = pd.Series([item[-1] for item in proba])
    labels = np.concatenate(labels, axis=0)
    labels = pd.Series(labels.ravel())
    fpr, tpr, threshold = metrics.roc_curve(labels, proba_class1)
    roc_auc = metrics.auc(fpr, tpr)
    # Plot
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
# Create a model
    
from dgl.nn import GraphConv
from dgl.nn import AvgPooling
from dgl.nn import MaxPooling
from dgl.nn import SumPooling
#from dgl.nn import SortPooling

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, number_of_hiddneurons,dropout_lin):
        super(GCN, self).__init__()
        allow_zero_in_degree=True
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats+in_feats, h_feats)
        self.conv3 = GraphConv(h_feats+in_feats, h_feats)
        self.conv4 = GraphConv(h_feats+in_feats, h_feats)
        self.conv5 = GraphConv(h_feats+in_feats, h_feats)
        self.conv6 = GraphConv(h_feats+in_feats, h_feats)
        self.dropout = nn.Dropout(dropout_lin)
        #self.conv2 = GraphConv(h_feats, h_feats)
        #self.conv3 = GraphConv(h_feats, h_feats)
        #self.conv6 = GraphConv(h_feats, h_feats)
        #self.convadd1 = GraphConv(3, h_feats)
        #self.convadd2 = GraphConv(h_feats, h_feats)
        #self.convadd3 = GraphConv(h_feats, h_feats)
        
        
        
        self.avgpooling = AvgPooling()
        self.maxpooling = MaxPooling()
        self.sumpooling = SumPooling()
        #self.dropout = nn.Dropout(dropout_lin)
        self.linear1 = nn.Linear(3*(h_feats+in_feats),  number_of_hiddneurons)
        #self.linear1 = nn.Linear(3*(h_feats+in_feats), number_of_hiddneurons)#256
        #self.linear2 = nn.Linear(3*(h_feats+in_feats)+2, number_of_hiddneurons)#256
        self.linear2 = nn.Linear(number_of_hiddneurons, number_of_hiddneurons)#256
        #self.linear3 = nn.Linear (500,200)
        self.classify = nn.Linear(number_of_hiddneurons, 2)#num_classes
        

    def forward(self, g, in_feat):
        h = F.relu(self.conv1(g, in_feat))
        #h = self.dropoutconv(h)
        h = torch.cat((h, in_feat), dim=1)
        h = F.relu(self.conv2(g, h))
        #h = self.dropoutconv(h)
        h = torch.cat((h, in_feat), dim=1)
        h = F.relu(self.conv3(g, h))
        #h = self.dropoutconv(h)
        h = torch.cat((h, in_feat), dim=1)
        h = F.relu(self.conv4(g, h))
        #h = self.dropoutconv(h)
        h = torch.cat((h, in_feat), dim=1)
        h = F.relu(self.conv5(g, h))
        h = torch.cat((h, in_feat), dim=1)
        h = F.relu(self.conv6(g, h))
        h = torch.cat((h, in_feat), dim=1)
        #h = F.relu(self.conv5(g, h))
        #h = F.relu(self.conv6(g, h))
        #h = torch.cat((h, in_feat), dim=1)
        #readout
        #g.ndata['h'] = h
        #mean = dgl.mean_nodes(g, 'h')
        #max_pool = dgl.maxpool(g, 'h')
        #h = self.dropout(h)
        
        #print(in_feat.shape)
        
        #i = F.relu(self.convadd1(g, in_feat2))
        #i = F.relu(self.convadd2(g, i))
        #i = F.relu(self.convadd3(g, i))
        
        #sum2 = self.sumpooling(g, i)

        mean = self.avgpooling(g, h)
        maxpool = self.maxpooling(g, h)
        sumpool = self.sumpooling(g, h)
        h = torch.cat((mean, maxpool, sumpool), dim=1)
        #h = sumpool
        #h = self.dropout(h)
        #h = F.relu(self.linear1(h))
        #h = torch.cat((h,((torch.stack([i.graph_features for i in g.g_list],axis=0).squeeze(1)))), dim=1)
        #h = torch.stack([i.graph_features for i in g.g_list],axis=0).squeeze(1)
        #h = self.dropout(h)
        h = self.dropout(h)
        h = F.relu(self.linear2(h))
        #h = self.dropout(h)
        #h = F.relu(self.linear3(h))
        return self.classify(h)
    
    
    
def _collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs) 
    batched_graph.g_list = graphs 
    return batched_graph, torch.tensor(labels)
    return _collate



from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from dgl.dataloading import GraphDataLoader


def train_and_validate(param, trial):
    
    # Create the model with given dimensions
    number_of_feats = 4  #3
    number_of_classes = 2
    number_of_hiddfeat = param['number_of_hiddfeat'] #128
    number_of_hiddneurons = param['number_of_hiddneurons']
    dropout_lin = param['dropout_lin']

        
    
    model = GCN(number_of_feats, number_of_hiddfeat, number_of_classes, number_of_hiddneurons, dropout_lin)
    batch_size = param['batch_size']
    #batch_size = 64
    best_vloss_overall = 1_000_000#0
    early_stopping = 0
    sm = torch.nn.Softmax(dim=1)  
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, threshold=0.001, threshold_mode='abs')
    weight0 = 1 #param['class_weight_zero']
    weight1 = param['class_weight_one']
    weights = [weight0, weight1]
    class_weights=torch.FloatTensor(weights)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    

    #loss_function = nn.CrossEntropyLoss()

    #graphs_train_fold = dgl.data.utils.Subset(train_dataset, train_index)
    #graphs_valid_fold = dgl.data.utils.Subset(validation_dataset, test_index)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, drop_last=False, collate_fn=_collate)
    validation_dataloader = GraphDataLoader(validation_dataset,  batch_size=batch_size, drop_last=False, collate_fn=_collate)
    epoch_training_losses = []
    epoch_validation_losses = []
    for epoch in range(1000):
        print('\nEpoch {} '.format(epoch + 1))
        
        #Training on training set
        model.train()
        predicted_class_proba = []
        predicted_class = []
        true = [] #labels
        train_loss = 0
        num_predictions = 0
        #for batched_graph, labels in iter, (bg, label) in train_dataloader:
        for iter, (batched_graph, labels) in enumerate(train_dataloader):
            

            #print(batched_graph.g_list)
            #print(batched_graph.ndata['featStack'].float().shape)
            
            optimizer.zero_grad()
            num_predictions += len(labels)
            pred = model(batched_graph, batched_graph.ndata['featStack'].float())
            probabilities = sm(pred)
            predicted_class_proba.append(probabilities.detach().numpy())
            true.append(labels.detach().numpy())
            predicted_class.append(pred.argmax(1))
            #loss = F.cross_entropy(pred, labels)
            loss = loss_function(pred,labels)
            train_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
        # Save the train loss per epoch
        train_loss_of_epoch = (train_loss / (iter+1))
        recall, precision, f1score, accuracy = calc_metrics(predicted_class, true)
        print("AUC:", calc_auc(predicted_class_proba, true))
        print("Recall:", recall)
        print("Precision:", precision)
        print("F1-Score:", f1score)
        print("Accuracy:", accuracy)
        print("--------------------------------------")   
        # Evaluation on validation set
        model.eval()
        predicted_class_proba = []
        predicted_class = []
        sm = torch.nn.Softmax(dim=1)
        true = [] #labels
        validation_loss = 0
        num_predictions = 0
        for iter, (batched_graph, labels) in enumerate(validation_dataloader): 
            num_predictions += len(labels)
            pred = model(batched_graph, batched_graph.ndata['featStack'].float())
            probabilities = sm(pred)
            predicted_class_proba.append(probabilities.detach().numpy())
            true.append(labels.detach().numpy())
            predicted_class.append(pred.argmax(1))
            #loss = F.cross_entropy(pred, labels)
            loss = loss_function(pred,labels)
            validation_loss += loss.detach().item()
        # Save the validation loss per epoch
        validation_loss_of_epoch = (validation_loss / (iter+1))
        recall, precision, f1score, accuracy = calc_metrics(predicted_class, true)
        print("AUC:", calc_auc(predicted_class_proba, true))
        print("Recall:", recall)
        print("Precision:", precision)
        print("F1-Score:", f1score)
        print("Accuracy:", accuracy)              

        roc_auc_score = calc_auc(predicted_class_proba, true)
        
        # Change learning rate if necessary
        #scheduler.step(validation_loss / len(validation_dataloader))
        #scheduler.step(validation_loss_of_epoch)
        scheduler.step(roc_auc_score)
        print("LR:",optimizer.param_groups[0]['lr'])
        print('LOSS train: {} LOSS valid: {}'.format(train_loss_of_epoch,validation_loss_of_epoch))

        
        trial.report(roc_auc_score, epoch)
        #epoch_training_losses.append(train_loss_of_epoch)
        #epoch_validation_losses.append(validation_loss_of_epoch)
        # Track best performance, and save the model's state
        if (round(roc_auc_score  ,3) < round(best_vloss_overall,3)):
            early_stopping = 0
            best_vloss_overall = roc_auc_score #roc_auc_score 
            model_path_best_overall = 'model_trial_{}'.format(trial.number)
            torch.save(model, model_path_best_overall) 
            
        else:
            early_stopping += 1
            print("Epochs without improvement: {}".format(early_stopping))
            if early_stopping > 15:
              print("Early stopping")
              break
        if (round(train_loss_of_epoch,1) == 0):
            print("Early stopping")
            break
    return best_vloss_overall
    

# Load the saved model
#saved_model = GCN(number_of_feats,number_of_hiddfeat,number_of_classes)
#saved_model.load_state_dict(torch.load(model_path_best_overall))
#model.eval()
    




import optuna
import torch.optim as optim

def objective(trial):

     params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
            'number_of_hiddfeat': trial.suggest_int("number_of_hiddfeat", 32, 256, log =True),
            'number_of_hiddneurons': trial.suggest_int("number_of_hiddneurons", 32, 1024, log =True),
          #'class_weight_zero': trial.suggest_loguniform("class_weight_zero", 0.1, 1),
         'class_weight_one': trial.suggest_loguniform("class_weight_one", 1, 25),
         'dropout_lin': trial.suggest_loguniform("dropout_lin", 0.01, 0.5),
         'batch_size': trial.suggest_int("batch_size", 128, 512, log=True),

     }
    
     #model = build_model(trial,params)
    
     best_vloss_overall= train_and_validate(params, trial)

     return best_vloss_overall
  
      
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50)
study.best_params 





########################################### Evaluate the best model on the test set
# Load the saved model
#model = GCN(number_of_feats,number_of_hiddfeat,number_of_classes)
#model.load_state_dict(torch.load(str("model_epoch_")+str(study.best_trial.number)))
model = torch.load(str("model_trial_")+str(study.best_trial.number))

batch_size=256
# Load the test set
test_dataloader = GraphDataLoader(test_dataset,  batch_size=batch_size, drop_last=False,  collate_fn=_collate)
predicted_class_proba = []
predicted_class = []
sm = torch.nn.Softmax(dim=1)
true = [] #labels
for batched_graph, labels in test_dataloader:
    model.eval()
    pred = model(batched_graph, batched_graph.ndata['featStack'].float())
    probabilities = sm(pred)
    predicted_class_proba.append(probabilities.detach().numpy())
    true.append(labels.detach().numpy())
    predicted_class.append(pred.argmax(1))

recall, precision, f1score, accuracy = calc_metrics(predicted_class, true)
print("AUC:", calc_auc(predicted_class_proba, true))
print("Recall:", recall)
print("Precision:", precision)
print("F1-Score:", f1score)
print("Accuracy:", accuracy)














