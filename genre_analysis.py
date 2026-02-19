import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("artist_data.csv")

def top_ten_per_genre(input_genre):

    df_filtered = df[df['artist_genres'].str.contains(input_genre, case = False, na = False)]
    df_sorted = df_filtered.sort_values(by = 'artist_popularity', ascending=False)
    return df_sorted.head(10)

#print(top_ten_per_genre('reggaeton'))

df_adjusted = df.copy()
df_adjusted['sum_of_genres'] = df_adjusted[['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5']].notna().sum(axis=1)


sb.boxplot(data=df_adjusted, x='sum_of_genres', y='artist_popularity')
plt.title("Distribution of Popularity by Sum of Genres")
plt.xlabel("Sum of Genres")
plt.ylabel("Popularity")
plt.show()

#plt.figure()
#sb.boxplot(data=df_adjusted, x='sum_of_genres', y='followers')
#plt.title("Distribution of Followers by Sum of Genres")
#plt.xlabel("Sum of Genres")
#plt.ylabel("Followers")
#plt.show()

#plt.figure()
#sb.boxplot(data=df_adjusted, x='sum_of_genres', y='followers')
#plt.yscale('log')
#plt.title("Distribution of Followers by Sum of Genres (log scale)")
#plt.xlabel("Sum of Genres")
#plt.ylabel("Followers")
#plt.show()

sb.barplot(data=df_adjusted, x='sum_of_genres', y='artist_popularity', errorbar = None)
plt.title("Average Popularity by Sum of Genres")
plt.xlabel("Sum of Genres")
plt.ylabel("Average Popularity")
plt.show()

sb.barplot(data=df_adjusted, x='sum_of_genres', y='followers', errorbar = None)
plt.title("Average Popularity by Sum of Genres")
plt.xlabel("Sum of Genres")
plt.ylabel("Average Popularity")
plt.show()

correlation_matrix = df_adjusted[['sum_of_genres', 'artist_popularity', 'followers']].corr()
print(correlation_matrix)

