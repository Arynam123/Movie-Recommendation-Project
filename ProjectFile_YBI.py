#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System
# 

# In[ ]:


Recommender System is a system that seeks to predict or filter preference accornding to the user's choices.


# In[ ]:


Collaborative Filtering: Collaborative filtering approaches build a model from the user's past behaviour as well as similar decisioin maded by the  user.


# In[ ]:


Content based filtering: Content based filtering approaches uses a series of discreate characteristics of an item in order to recoommend addditional item withe siimilare propertieos


# # Import library

# In[4]:


import pandas as pd


# # Import dataset

# In[1]:


import pandas as pd
df=pd.read_csv('Movies Recommendation.csv')


# # Describe Data and Data Visualization

# In[3]:


df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.columns


# In[6]:


df.index


# In[7]:


df.shape


# # Define Target Variable(y) and Feature variable(x)

# In[6]:


df_features=df[['Movie_Genre','Movie_Keywords','Movie_Tagline','Movie_Cast','Movie_Director']].fillna('')


# In[7]:


df_features.shape


# 

# In[8]:


x=df_features['Movie_Genre']+' '+df_features['Movie_Keywords']+' '+df_features['Movie_Tagline']+' '+df_features['Movie_Cast']+' '+df_features['Movie_Director']


# In[9]:


x.shape


# # Get Feature Text Conversion to tokens

# In[10]:


#from sklearn.features_extraction.text import Tfidvectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:


tfidf=TfidfVectorizer()


# In[12]:


X=tfidf.fit_transform(x)


# In[13]:


X.shape


# In[14]:


print(X)


# In[15]:


from sklearn.metrics.pairwise import cosine_similarity


# In[16]:


Similarity_score=cosine_similarity(X)


# In[17]:


Similarity_score


# # Model Evaluation : Get movie name input as user and validate and closest spelling

# In[18]:


Fav_movie_name=input("Enter your movie name: ")


# In[19]:


All_movies_title_list=df['Movie_Title'].tolist()


# In[20]:


import difflib


# In[21]:


Movie_Re=difflib.get_close_matches(Fav_movie_name,All_movies_title_list)


# In[22]:


print (Movie_Re)


# In[23]:


Close_Match=Movie_Re[0]


# In[24]:


print(Close_Match)


# In[25]:


Index_of_close_match=df[df.Movie_Title==Close_Match]['Movie_ID'].values[0]


# In[26]:


print(Index_of_close_match)


# In[27]:


Recomm_Score=list(enumerate(Similarity_score[Index_of_close_match]))
print(Recomm_Score)


# In[28]:


len(Recomm_Score)


# # Prediction : Get All movies sort based on Recommendation Score wrt Favoriter Movie

# In[29]:


#Sorting based on similiar movies
Sorted_similar_movie=sorted(Recomm_Score,key=lambda x:x[1],reverse=True)
print(Sorted_similar_movie)


# In[30]:


#Print the name of similare movies based on index
print("Top 30 movies suggested for you")
i=1
for movie in Sorted_similar_movie:
    index=movie[0]
    title_from_index=df[df.index==index]['Movie_Title'].values[0]
    if i<31:
        print(i,'.',title_from_index)
        i+=1


# In[31]:


Top 10 Movie Recommendation System


# In[32]:


Movie_Name=input("Enter the name of your favourite movie: ")


# In[33]:


list_of_all_title=df['Movie_Title'].tolist()


# In[34]:


Find_close_match=difflib.get_close_matches(Movie_Name,list_of_all_title)


# In[35]:


close_match=Find_close_match[0]



# In[36]:


Index_of_Movie=df[df.Movie_Title==close_match]['Movie_ID'].values[0]


# In[37]:


Recomm_Score=list(enumerate(Similarity_score[Index_of_close_match]))


# In[38]:


Sorted_similar_movie=sorted(Recomm_Score,key=lambda x:x[1],reverse=True)


# In[39]:


# print("Top 10 movies suggested for you")
i=1
for movie in Sorted_similar_movie:
    index=movie[0]
    title_from_index=df[df.index==index]['Movie_Title'].values[0]
    if i<11:
        print(i,'.',title_from_index)
        i+=1


# Explanation: This model recommend list of movies based on your choice

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




