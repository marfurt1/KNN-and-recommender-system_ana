from utils import db_connect
# Import libraries


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import dataset

movies = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv')
credits = pd.read_csv('../data/raw/tmdb_5000_credits.csv')

#Merge both dataframes on the 'title' column
movies = movies.merge(credits, on='title')

#Select columns that will be used
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

#As there are only 3 missing values in the 'overview' column, drop them
movies = movies.dropna()

#Working with columns in json format
movies.iloc[0].genres

#Converting these columns using a function to obtain only the genres, without a json format. 
# We are only interested in the values of the 'name' keys
import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)

#The same process for the 'keywords' column
movies['keywords'] = movies['keywords'].apply(convert)

#For the 'cast' column we will create a new but similar function. This time we will limit the number of items to three.
def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L
movies['cast'] = movies['cast'].apply(convert3)

#The only columns left to modify are 'crew' and 'overview'. For the 'crew', we will create a new function that allows to obtain only the values of the 'name' keys for whose 'job' value is 'Director'. 
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
movies['crew'] = movies['crew'].apply(fetch_director)
#For the 'overview' column, we will convert it in a list by using 'split()' methode.
movies['overview'] = movies['overview'].apply(lambda x : x.split())

#For the recommender system to do not get confused, we will remove spaces between words with a function.
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
#Apply our function to the 'genres', 'cast', 'crew' and 'keywords' columns
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

#We will reduce our dataset by combining all our previous converted columns into only one column named 'tags' (which we will create). This column will now have ALL items separated by commas, but we will ignore commas by using lambda x :" ".join(x).
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
new_df = movies[['movie_id','title','tags']]
#new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))

#We will use KNN algorithm to build the recommender system. Before entering the model let's proceed with the text vectorization which you already learned in the NLP lesson.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000 ,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors.shape

#Let's find the cosine_similarity among the movies. Go ahead and run the following code lines in your project to see the results.
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors).shape
similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse =True , key = lambda x:x[1])[1:6]

#Create a recommendation function based on the cosine_similarity. This function should recommend the 5 most similar movies
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate( distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

#MAke a test
recommend('Independence Day')
