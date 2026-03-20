import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Connect to database
@st.cache_data
def load_data():
    conn = sqlite3.connect("data/spotify_database.db")

    artists = pd.read_sql("SELECT * FROM artist_data", conn)
    tracks = pd.read_sql("SELECT * FROM tracks_data", conn)
    albums = pd.read_sql("SELECT * FROM albums_data", conn)

    conn.close()
    return artists, tracks, albums

# Main App
def main():
    st.title("Spotify Data Analysis Dashboard")
    st.subheader("Opening Overview")

    # Numerical Summary
    st.header("Key Statistics")

    artists, tracks, albums = load_data()

    total_artists = artists["name"].nunique()
    total_tracks = tracks["id"].nunique()
    total_albums = albums["album_id"].nunique()

    avg_artist_pop = artists["artist_popularity"].mean()
    avg_track_pop = tracks["track_popularity"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Artists", total_artists)
    col2.metric("Total Tracks", total_tracks)
    col3.metric("Total Albums", total_albums)

    col4, col5 = st.columns(2)
    col4.metric("Avg Artist Popularity", round(avg_artist_pop, 2))
    col5.metric("Avg Track Popularity", round(avg_track_pop, 2))

    # Insight Section
    st.header(" Initial Insights")

    st.write("""
    - The dataset contains a diverse set of artists, tracks, and albums.
    - Popularity distribution shows how tracks are spread across low to high popularity ranges.
    - The scatter plot helps identify whether more popular artists tend to produce more popular tracks.
    - Further analysis explores collaborations, audio features, and trends over time.
    """)

# Genre Analysis
    st.header(" Genre Statistics")

    genre_cols = ["genre_1", "genre_2", "genre_3", "genre_4", "genre_5", "genre_6"]
    genre_cols = [col for col in genre_cols if col in artists.columns]

    genre_counts = Counter()

    for row in artists[genre_cols].values:
        genres = [g for g in row if pd.notna(g) and g != ""]
        genre_counts.update(genres)

    # Convert to DataFrame
    genre_df = pd.DataFrame(genre_counts.items(), columns=["Genre", "Count"])
    genre_df = genre_df.sort_values(by="Count", ascending=False)
    genre_df["Genre"] = genre_df["Genre"].str.lower().str.strip()

    # Number of unique genres
    num_unique_genres = genre_df["Genre"].nunique()

    st.metric("Number of Unique Genres", num_unique_genres)

    # Show top 10 genres
    top_genres = genre_df.sort_values(by="Count", ascending=False).head(10)
    top_genres = genre_df.head(10)

    st.subheader("Top 10 Most Common Genres")
    st.dataframe(top_genres)

    fig, ax = plt.subplots()
    ax.bar(top_genres["Genre"], top_genres["Count"])

    ax.set_title("Top 10 Genres")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # Graphical Summary
    st.header(" Popularity Distribution")

    fig, ax = plt.subplots()
    ax.hist(tracks["track_popularity"].dropna(), bins=20)
    ax.set_title("Distribution of Track Popularity")
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)



    # Artist vs Track Popularity
    st.header(" Artist vs Track Popularity")

    merged = tracks.merge(artists, left_on="id", right_on="id", how="inner")

    fig2, ax2 = plt.subplots()
    ax2.scatter(
        merged["artist_popularity"],
        merged["track_popularity"]
    )
    ax2.set_xlabel("Artist Popularity")
    ax2.set_ylabel("Track Popularity")
    ax2.set_title("Artist vs Track Popularity")

    st.pyplot(fig2)


# Run App
if __name__ == "__main__":
    main()