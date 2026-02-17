import pandas as pd
import matplotlib.pyplot as plt

# Import CSV file as dataframe
data = pd.read_csv("artist_data.csv")

no_of_columns = len(data.columns) # Length of columns attribute of dataframe
columns = data.columns # Columns attribute of dataframe
datatypes = data.dtypes # Datatypes attribute of dataframe

print("Number of columns in the dataframe:", no_of_columns)
print(data.head)

# Find unique artists and how many unique artists they are
artist = data["name"].unique()
print(artist)
print("Number of unique artists in the dataframe:", len(artist))

# Finding the Top 10 artists 
top_10_artists = (
    data.groupby("name")["artist_popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    )

print(top_10_artists)

# Create barplot
plt.bar(top_10_artists.index, top_10_artists.values)
plt.title('Top 10 Artists')
plt.xlabel('Artist')
plt.ylabel('Popularity')
plt.tight_layout()
plt.show()