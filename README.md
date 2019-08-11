# Youtube Videos Popularity Prediction
With the widespread online global access of data and the ease with which an online content can be produced, we often are directed to understand the underlying concept of popularity growth on the internet. It is of utmost relevance to a broad range of services like designing an effective caching model, viral marketing strategies, estimation of costs, advertisement campaigns and for the overall improvement for the future content.

"Trending Videos" are videos that have become popular because they were embedded in the web’s most popular websites and a significant number of people viewed the video externally in addition to on youtube.com

The Video IDs of more than 25000 videos uploaded in the month of November 2017(for the regions United States, Canada, United Kingdom, Germany, and France) was collected by scrapping the YouTube API. Around 8700 unique Video IDs obtained from more than 25000 Video IDs are selected. The meta data of these videos are collected by again querying the YouTube API.

The Trending videos meta data is directly obtained from crowdsourcing platform Kaggle in CSV format. The Trending videos of five regions (United States, Canada, United Kingdom, Germany, and France) are collected for the month of November 2017.

The Title and tags of the trending videos are used to build a ranking system. They are striped and split into individual words after removing the stop words, emoticons and punctuations. The frequency of every word is calculated and used as a ranking system. This ranking system is used to replace the tags and title. Unique words in tags and titles are Predicting the Popularity of YouTube Videos extracted after removing the stop words, punctuations and emoticons, average of the frequency of these words are used in building the model.

The model is trained using Logistic regression after scaling the data.

The model is compared with KNN classifier and also Linear SVC.
