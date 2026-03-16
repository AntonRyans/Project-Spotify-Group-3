# Import libraries
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Connect & Join Tables
# -----------------------------
conn = sqlite3.connect('spotify_database.db')

query = """
SELECT 
    a.track_name,
    a.release_date,
    t.track_popularity
FROM albums_data a
JOIN tracks_data t
ON a.track_id = t.id
"""

df = pd.read_sql_query(query, conn)
conn.close()

# -----------------------------
# STEP 2: Data Cleaning
# -----------------------------
df['track_popularity'] = pd.to_numeric(df['track_popularity'], errors='coerce')
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

df = df.dropna(subset=['track_name', 'track_popularity', 'release_date'])

# -----------------------------
# STEP 3: Create Month Column
# -----------------------------
df['year_month'] = df['release_date'].dt.to_period('M')

# -----------------------------
# STEP 4: Find Top Songs
# -----------------------------
top_songs = (
    df.groupby('track_name')['track_popularity']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_song_names = top_songs.index.tolist()

print("Top 10 Songs by Total Popularity:")
print(top_songs)

# -----------------------------
# STEP 5: Filter Top Songs
# -----------------------------
df_top = df[df['track_name'].isin(top_song_names)]

# -----------------------------
# STEP 6: Aggregate Monthly Data
# -----------------------------
monthly_agg = (
    df_top.groupby(['year_month', 'track_name'])
    .agg({
        'track_popularity': 'mean'
    })
    .reset_index()
)

# Convert for plotting
monthly_agg['year_month'] = monthly_agg['year_month'].astype(str)

# Sort properly
monthly_agg = monthly_agg.sort_values('year_month')

# -----------------------------
# STEP 7: Plot Popularity Over Time
# -----------------------------
plt.figure()

for song in top_song_names:
    song_data = monthly_agg[monthly_agg['track_name'] == song]
    plt.plot(song_data['year_month'], song_data['track_popularity'], label=song)

plt.title('Monthly Popularity Trends for Top Songs')
plt.xlabel('Month')
plt.ylabel('Average Popularity')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()