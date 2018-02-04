# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 18:56:40 2017

@author: tejasdhasrali
         apurvakatti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sb
import re
import json
import pickle

from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import sklearn.preprocessing as pre
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier



#Stop Words provided by NLTK
stop_words={'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn','1','2','3','4','5','6','7','8','9','0','vs','le','la','de','none'}


    
def replaceNullByOne(columns):          #Replaces the Null Values by one
    columnValue=columns[0]
    if pd.isnull(columnValue):
        return 1
    else:
        return columnValue
    

def replaceNullByZero(columns):         #Replaces the Null Values by zero
    columnValue=columns[0]
    if pd.isnull(columnValue):
        return 0
    else:
        return columnValue


def removeTime(columns):                #Removes the time from date object and just return Date
    date=columns[0]
    dateAndTime=date.split('T')
    return dateAndTime[0]


def calculateRate(columns):             #Calculates the rate of the numerator vs denominator
    numerator=columns[0]
    denominator=columns[1]
    if(denominator>0):
        return numerator/denominator
    else:
        return 0
    

def categoryIdToCategory():              # creates a dictionary that maps category_id to category name
   
    categoryIdTocategory = {}
    with open('category_id.json', 'r') as f:
        data = json.load(f)
        for category in data['items']:
            categoryIdTocategory[category['id']] = category['snippet']['title']
    return categoryIdTocategory
    

def convertCategoryIdToCategory(columns,categoryIdToCategory):      #Using the category to category Id dictionary returns the category name for the given category ID
    categoryId=str(columns[0])
    try:
        category=categoryIdToCategory[categoryId]
    except Exception as e:
        category=0
    return category


def convertChannelIdToChannelTitle(columns,channelDictionary):      #Using the channel Dictionary converts the given channel ID to its Channel Name
    channelId=columns[0]
    try:
        temp=channelDictionary[channelId]
        channelName=temp['channelTitle']
    except Exception as e:
        channelName=1
    return channelName




def preProcessTheData(youTubeTrendingData,youTubeNonTrendingData):      #Preprocesses the Data to create one Dataframe containing both Trending and Non trending videos
    #Preprocessing Trending Data
    #Formatting the Date
    youTubeTrendingData['trending_date'] = pd.to_datetime(youTubeTrendingData['trending_date'], format="%y.%d.%m")
    youTubeTrendingData['publish_time']=youTubeTrendingData[['publish_time']].apply(removeTime, axis=1)
    youTubeTrendingData['publish_time'] = pd.to_datetime(youTubeTrendingData['publish_time'],format=("%Y-%m-%d"))
    #Calculating the age of the video
    youTubeTrendingData['age']=(((youTubeTrendingData['trending_date']-youTubeTrendingData['publish_time']).dt.days)+1)
    #Replace Null Values in Description by One
    youTubeTrendingData['description']=youTubeTrendingData[['description']].apply(replaceNullByOne,axis=1)
    
    
    #Preprocessing Non Trending data
    #Adding the Data Retreived Date Column
    youTubeNonTrendingData['obtained_date']=pd.to_datetime('2017-12-02',format=('%Y-%m-%d'))
    #Fomatting the Date
    youTubeNonTrendingData['publishedAt']=youTubeNonTrendingData[['publishedAt']].apply(removeTime, axis=1)
    youTubeNonTrendingData['publishedAt']=pd.to_datetime(youTubeNonTrendingData['publishedAt'],format=("%Y-%m-%d"))
    #Calculate the age of the video
    youTubeNonTrendingData['age']=((((pd.to_datetime('2017-12-02',format=("%Y-%m-%d")))-youTubeNonTrendingData['publishedAt']).dt.days)+1)
    #Remove the Null Values in the Data
    youTubeNonTrendingData['tags']=youTubeNonTrendingData[['tags']].apply(replaceNullByOne,axis=1)
    youTubeNonTrendingData['description']=youTubeNonTrendingData[['description']].apply(replaceNullByOne,axis=1)
    youTubeNonTrendingData['commentCount']=youTubeNonTrendingData[['commentCount']].apply(replaceNullByZero,axis=1)
    youTubeNonTrendingData['dislikeCount']=youTubeNonTrendingData[['dislikeCount']].apply(replaceNullByZero,axis=1)
    youTubeNonTrendingData['likeCount']=youTubeNonTrendingData[['likeCount']].apply(replaceNullByZero,axis=1)
    youTubeNonTrendingData['viewCount']=youTubeNonTrendingData[['viewCount']].apply(replaceNullByZero,axis=1)
    #Add trending column with 0 for Non trending Video
    youTubeNonTrendingData['trending']=0
    
    #Convert the Channel ID to Channel Title using the 'channels_dict' Dictionary in Non trending data
    with open('channels_dict','rb') as f:
        channelDictionary = pickle.load(f)
    youTubeNonTrendingData['channel_title']=youTubeNonTrendingData[['channelId']].apply(convertChannelIdToChannelTitle,args=(channelDictionary,),axis=1)
    
    
    #Combine Trending and Non Trending data
    #Drop unused features from Trending data and rename the features to match the Columns of Non Trending data
    youTubeTrendingDataTemp=youTubeTrendingData.drop(['comments_disabled','ratings_disabled','video_error_or_removed'],axis=1)
    youTubeTrendingDataTemp.rename(columns={'trending_date':'obtained_date'}, inplace=True)
    #Drop unused features from Non Trending data and rename the features to match the Columns of Trending data
    youTubeNonTrendingDataTemp=youTubeNonTrendingData.drop(['duration','dimension','channelId','projection','license','embeddable','licencedContent','privacyStatus','caption','definition','defaultAudioLanguage'],axis=1)
    youTubeNonTrendingDataTemp.rename(columns={'V_id':'video_id','categoryId':'category_id','publishedAt':'publish_time','viewCount':'views','likeCount':'likes','dislikeCount':'dislikes','commentCount':'comment_count','thumbnail':'thumbnail_link'}, inplace=True)
    #Concatinate trending and non trending data
    youTubeData=pd.concat([youTubeTrendingDataTemp,youTubeNonTrendingDataTemp],ignore_index=True)
    #Convert the category IDs to category names using the JSON dictionary 'category_id'
    categoryDict=categoryIdToCategory()
    youTubeData['category_id']=youTubeData[['category_id']].apply(convertCategoryIdToCategory,args=(categoryDict,),axis=1)
    return youTubeData,youTubeTrendingDataTemp,youTubeNonTrendingDataTemp
    

def frequencyOfChannelTitle(columns,channelTitleCount):             #Replaces the Channel Title with its freqeuncy of occurence
    channelTitle=columns[0]
    if(type(channelTitle)==str):
        try:
            channelTitleCount=channelTitleCount[1][channelTitle]
            return channelTitleCount
        except Exception as e:
            return 0
    else:
        return channelTitle


def frequencyOfCategory(columns,categoryCount):                 #(Unused method) Replaces the Category ID with its freqeuncy of occurence
    category=columns[0]
    try:
        categoryIdCount=categoryCount[1][category]
    except Exception as e:
        return 0
    return categoryIdCount


def prepareTagRanking(youTubeTrendingData):               #Splits the Title and tags from trending data into words,removes nonascii characters, removes stop words and calculates the number occurences these words for ranking  
    tags=[]
    for tag in youTubeTrendingData['tags']:
        if(type(tag) == str):
            temp=tag.split('|')
            for t in temp:
                temp1=t.split()
                for t1 in temp1:
                    t1=re.sub('[^A-Za-z0-9]+', '', t1).lstrip()
                    if (t1):
                        tags.append(t1.lower())
    
    for title in youTubeTrendingData['title']:
        if(type(title)==str):
            temp=title.split()
            for t in temp:
                t=re.sub('[^A-Za-z0-9]+', '', t).lstrip()
                if(t):
                    tags.append(t.lower())
    

    filtered_tags = [w for w in tags if not w in stop_words]
    if(len(filtered_tags)):
        tags=pd.DataFrame(data=filtered_tags)
        tags.columns=['tags']
        tagCount=tags['tags'].value_counts()
        return tagCount
    return []

    
def processTag(columns,tagRanking):         #Splits the given tags into words, removes nonascii characters, stop words and calculates the average rank of given tag
    tag=columns[0]
    trending=columns[1]
    tagAfterSplit=[]
    if(type(tag)==str):
        if(trending==1):
            tagFirstSplit=tag.split('|')
        else:
            tagFirstSplit=tag.split(',')
        for t in tagFirstSplit:
            tagSecondSplit=t.split()
            for t in tagSecondSplit:
                t=re.sub('[^^A-Za-z0-9]+','',t).lstrip()
                if(t):
                    
                    tagAfterSplit.append(t.lower())
                    
        filteredTag=[w for w in tagAfterSplit if not w in stop_words]
        filteredUniqueTag=set(filteredTag)
        sum=0
        for f in filteredUniqueTag:
            try:
                sum+=int(tagRanking[f])
            except Exception as e:
                sum+=1
        if(len(filteredUniqueTag)>0):
            average=(sum/len(filteredUniqueTag))
            return average
        else:
            return 1
    else:
        return tag
    
    
def processTitle(columns,wordRanking):          #Splits the given title into words, removes nonascii characters, stop words and calculates the average rank of given title
    title=columns[0]
    titleAfterSplit=[]
    if(type(title)==str):
        temp=title.split()
        for t in temp:
            t=re.sub('[^^A-Za-z0-9]+','',t).lstrip()
            if(t):
                titleAfterSplit.append(t.lower())
        filteredTitle=[w for w in titleAfterSplit if not w in stop_words]
        filteredUniqueTitle=set(filteredTitle)
        sum=0
        for f in filteredUniqueTitle:
            try:
                sum+=int(wordRanking[f])
            except Exception as e:
                sum+=1
        if(len(filteredUniqueTitle)>0):
            average=(sum/len(filteredUniqueTitle))
            return average
        else:
            return 1
    else:
        return title


def processTheData(youTubeData,youTubeTrendingData):        #processes the data and makes it ready for feature selection
    #Calculates the frequency of the channel title in trending videos and replaces the channel title with its frequency
    channelTitleCount=youTubeData.groupby('trending')['channel_title'].value_counts() 
    youTubeData['channel_title']=youTubeData[['channel_title']].apply(frequencyOfChannelTitle, axis=1, args=(channelTitleCount,))
    #calculates the tag and video title ranking(i.e the frequnecy of ocuurence of every word) and replaces the video title and tags with its value
    tagRanking=prepareTagRanking(youTubeTrendingData)
    youTubeData['tags']=youTubeData[['tags','trending']].apply(processTag,axis=1,args=(tagRanking,))
    youTubeData['title']=youTubeData[['title']].apply(processTitle,axis=1,args=(tagRanking,))
    #Calculates the rate of views, likes and dislikes per day. 
    youTubeData['rateOfViews']=youTubeData[['views','age']].apply(calculateRate,axis=1)
    youTubeData['rateOfLikes']=youTubeData[['likes','age']].apply(calculateRate,axis=1)
    #Creates a dummy list for the categorical data Categort IDs and adds these columns to the Data Frame
    categoryDummyList = pd.get_dummies(youTubeData['category_id'], prefix='category')
    youTubeData=youTubeData.join(categoryDummyList)
    return youTubeData

def createROCCurve(y_test,y_pred,heading):      #plots ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_pred = label_binarize(y_pred,classes=[0,1])
    y_test = label_binarize(y_test,classes=[0,1])
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(heading)
    plt.legend(loc="lower right")
    plt.show()
    


#Main
#Read the Data
youTubeTrendingData=pd.read_csv("TrendingVideos.csv", encoding = "UTF-8",index_col='video_id')
youTubeNonTrendingData=pd.read_csv("NonTrendingVideos.csv", encoding = "UTF-8",index_col='V_id')
youTubeTrendingData=resample(youTubeTrendingData,replace=False,n_samples=len(youTubeNonTrendingData))   #Resampling of Data for balancing Class Labels
#Pre-processing the Data
youTubeData,youTubeTrendingData,youTubeNonTrendingData=preProcessTheData(youTubeTrendingData,youTubeNonTrendingData)
#Processing the combined Trending and Non Trending Data
youTubeData=processTheData(youTubeData,youTubeTrendingData)
#Drop unused features
youTubeDataForFeatureSelection=youTubeData.drop(['category_id','description','obtained_date','publish_time','thumbnail_link'],axis=1)
#Divide the Data into features and class labels
X = youTubeDataForFeatureSelection.ix[:,(0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)].values
y = youTubeDataForFeatureSelection.ix[:,7].values
#Scaling of features
X=pre.scale(X)

#Selects the most important features from the model to make the model better and reduce the dimensionality
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)

print('\n\n\n------------------------------------LOGISTIC REGRESSION------------------------------------\n\n\n')





print('\n\n------------------------------------Before Applying Linear SVC For Feature Selection------------------------------------\n\n')
featuresUsed=list(youTubeDataForFeatureSelection.columns.values)
print('Features Used\n',featuresUsed)
print('Total',len(featuresUsed))
#Before Feature Selection
#Splits the data into training and testing data. One-third of the data is selected for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=25)
#Logistic Regression is used for traning the model
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
#Confusion matrix is used to calculate the accuracy of the model
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy=((confusion_matrix[0][0]+confusion_matrix[1][1])/len(y_test)*100)
print('\nAccuracy',accuracy,'\n')
print('Classfication Report\n',classification_report(y_test, y_pred))
createROCCurve(y_test,y_pred,'Before Applying Linear SVC For Feature Selection')





#Obtain the Features selected using Linear SVC
print('\n\n------------------------------------After Applying Linear SVC For Feature Selection------------------------------------\n\n')

i=0
featuresSelectedBooleanList=model.get_support()
featuresSelected=[]
temp=youTubeDataForFeatureSelection.drop(['trending'],axis=1)
for headerName in temp.columns.values:
    if(featuresSelectedBooleanList[i]):
        featuresSelected.append(headerName)
    i=i+1
print('Features used\n',featuresSelected)
print('Total',len(featuresSelected))

#After Feature selection
#Same approach is repeated
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3,random_state=25)

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy=((confusion_matrix[0][0]+confusion_matrix[1][1])/len(y_test)*100)
print('\nAccuracy',accuracy,'\n')
print('Classfication Report\n',classification_report(y_test, y_pred))
createROCCurve(y_test,y_pred,'After Applying Linear SVC For Feature Selection')





print('\n\n\n--------------Training The Model After Removing Tags, Channel Title, Age of the Video and Video Title--------------\n\n\n')

X = youTubeDataForFeatureSelection.ix[:,(2,3,4,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)].values
y = youTubeDataForFeatureSelection.ix[:,7].values
X=pre.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy=((confusion_matrix[0][0]+confusion_matrix[1][1])/len(y_test)*100)
print('\nAccuracy',accuracy,'\n')
print('Classfication Report\n',classification_report(y_test, y_pred))
createROCCurve(y_test,y_pred,'Training The Model After Removing Tags, Channel Title, Age of the Video and Video Title')







print('\n\n\n------------------------------------K-NEAREST NEIGHBOUR CLASSIFIER------------------------------------\n\n\n')


X = youTubeDataForFeatureSelection.ix[:,(0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)].values
y = youTubeDataForFeatureSelection.ix[:,7].values

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)



print('\n\n------------------------------------Before Applying Linear SVC For Feature Selection------------------------------------\n\n')
featuresUsed=list(youTubeDataForFeatureSelection.columns.values)
print('Features Used\n',featuresUsed)
print('Total',len(featuresUsed))


l=[5,10,15,25,40,50,100,200,500]
print('Value of K\t   Accuracy')
for i in l:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,)
    #K Neasrest Classifier with different 
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    actual = y_test
    accuracy = (np.sum(predictions == actual)/len(actual))*100
    print(i,'\t\t',accuracy)
    



print('\n\n------------------------------------After Applying Linear SVC For Feature Selection------------------------------------\n\n')

i=0
featuresSelectedBooleanList=model.get_support()
featuresSelected=[]
temp=youTubeDataForFeatureSelection.drop(['trending'],axis=1)
for headerName in temp.columns.values:
    if(featuresSelectedBooleanList[i]):
        featuresSelected.append(headerName)
    i=i+1
print('Features used\n',featuresSelected)
print('Total',len(featuresSelected))


print('Value of K\t   Accuracy')
for i in l:
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3,)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    actual = y_test
    accuracy = (np.sum(predictions == actual)/len(actual))*100
    print(i,'\t\t',accuracy)  
    
    


print('\n\n\n--------------Training The Model After Removing Tags, Channel Title, Age of the Video and Video Title--------------\n\n\n')

X = youTubeDataForFeatureSelection.ix[:,(2,3,4,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)].values
y = youTubeDataForFeatureSelection.ix[:,7].values
print('Value of K\t   Accuracy')
for i in l:
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3,)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    actual = y_test
    accuracy = (np.sum(predictions == actual)/len(actual))*100
    print(i,'\t\t',accuracy)  
    

print('\n\n\n------------------------------------Observations on the Data------------------------------------\n\n\n')
pivoted = pd.pivot_table(youTubeData, values='age', columns='category_id', index='trending')
pivoted.plot(title="Age of the Video vs Trending").legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

pivoted = pd.pivot_table(youTubeData, values='rateOfViews', columns='category_id', index='trending')
pivoted.plot(title="Rate of Views vs Trending").legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

