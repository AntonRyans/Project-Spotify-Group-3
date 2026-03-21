import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from analysis import top_10_artists


# Load data
@st.cache_data
def load_data():
    conn = sqlite3.connect("data/spotify_database.db")

    artists = pd.read_sql("SELECT * FROM artist_data", conn)
    tracks = pd.read_sql("SELECT * FROM tracks_data", conn)
    albums = pd.read_sql("SELECT * FROM albums_data", conn)

    # Genres
    genre_cols = [
        col
        for col in ["genre_1", "genre_2", "genre_3", "genre_4", "genre_5", "genre_6"]
        if col in artists.columns
    ]
    genre_counts = Counter()
    for row in artists[genre_cols].values:
        genres = [g for g in row if pd.notna(g) and g != ""]
        genre_counts.update(genres)
    genre_df = pd.DataFrame(
        genre_counts.items(), columns=["Genre", "Count"]
    ).sort_values(by="Count", ascending=False)

    conn.close()
    return artists, tracks, albums, genre_df


def main():
    st.title("Spotify Data Analysis Dashboard")

    # Load data
    artists, tracks, albums, genre_df = load_data()

    # --- Key Metrics ---
    st.header("Key Statistics")
    total_artists = artists["name"].nunique()
    total_tracks = tracks["id"].nunique()
    total_albums = albums["album_id"].nunique()
    num_unique_genres = genre_df["Genre"].nunique()
    avg_artist_pop = round(artists["artist_popularity"].mean(), 2)
    avg_track_pop = round(tracks["track_popularity"].mean(), 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Artists", total_artists)
    c2.metric("Total Tracks", total_tracks)
    c3.metric("Total Albums", total_albums)

    c4, c5, c6 = st.columns(3)
    c4.metric("Unique Genres", num_unique_genres)
    c5.metric("Avg Artist Popularity", avg_artist_pop)
    c6.metric("Avg Track Popularity", avg_track_pop)

    # Insight Section
    st.header("Initial Insights")

    st.write("""
    1. The Spotify dataset contains a diverse set of artists, tracks, and albums.
    2. Popularity distribution shows how tracks are spread across low to high popularity ranges.
    3. The scatter plot helps identify whether more popular artists tend to produce more popular tracks.
    4. Further analysis explores collaborations, audio features, and trends over time.
    """)

    # Tabs for Top 10
    st.header("Top 10 Insights")
    tab1, tab2, tab3 = st.tabs(["Artists", "Albums", "Genres"])

    # Top 10 Artists
    with tab1:
        st.subheader("Top 10 Artists by Popularity")
        top_10_artists(artists)

    # --- Top 10 Albums ---
    with tab2:
        st.subheader("Top 10 Albums by Popularity")
        album_name_col = [c for c in albums.columns if "name" in c.lower()][0]
        album_pop_col = [c for c in albums.columns if "popularity" in c.lower()][0]
        top_albums = (
            albums[[album_name_col, album_pop_col]]
            .dropna()
            .sort_values(by=album_pop_col, ascending=False)
            .head(10)
        )
        st.dataframe(top_albums.reset_index(drop=True))

        # Bar chart
        fig_albums, ax_albums = plt.subplots(figsize=(8, 5))
        ax_albums.bar(
            top_albums[album_name_col], top_albums[album_pop_col], color="green"
        )
        ax_albums.set_xlabel("Album")
        ax_albums.set_ylabel("Popularity")
        ax_albums.set_title("Top 10 Albums")
        plt.xticks(rotation=45)
        st.pyplot(fig_albums)

    # --- Top 10 Genres ---
    with tab3:
        st.subheader("Top 10 Most Common Genres")
        top_genres = genre_df.head(10)
        st.dataframe(top_genres.reset_index(drop=True))

        # Bar chart
        fig_genres, ax_genres = plt.subplots(figsize=(8, 5))
        ax_genres.bar(top_genres["Genre"], top_genres["Count"], color="orange")
        ax_genres.set_xlabel("Genre")
        ax_genres.set_ylabel("Count")
        ax_genres.set_title("Top 10 Genres")
        plt.xticks(rotation=45)
        st.pyplot(fig_genres)


if __name__ == "__main__":
    main()
