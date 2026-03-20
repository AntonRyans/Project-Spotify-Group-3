import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import top_10_artists
import data_wrangling
import database_analysis
from collections import Counter

# Connect to database
@st.cache_data
def load_data():
    conn = sqlite3.connect("data/spotify_database.db")

    artists = pd.read_sql("SELECT * FROM artist_data", conn)
    tracks = pd.read_sql("SELECT * FROM tracks_data", conn)
    albums = pd.read_sql("SELECT * FROM albums_data", conn)

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

    conn.close()
    return artists, tracks, albums, genre_df

# Main App
def main():
    st.title("Spotify Data Analysis Dashboard")
    st.subheader("Opening Overview")

    # Numerical Summary
    st.header("Key Statistics")

    artists, tracks, albums, genre_df = load_data()

    total_artists = artists["name"].nunique()
    total_tracks = tracks["id"].nunique()
    total_albums = albums["album_id"].nunique()

    num_unique_genres = genre_df["Genre"].nunique()

    avg_artist_pop = round(artists["artist_popularity"].mean(),2)
    avg_track_pop = round(tracks["track_popularity"].mean(),2)

    st.markdown(f"""
    <style>
    .stats-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Source Sans Pro', sans-serif;
    }}
    .stats-table th, .stats-table td {{
        border: 1px solid #ddd;
        padding: 12px 15px;
        text-align: center;
    }}
    .stats-table th {{
        background-color: #1DB954;
        color: white;
        font-size: 16px;
    }}
    .stats-table td {{
        font-size: 15px;
        color: #333;
    }}
    .stats-table tr:nth-child(even) {{
        background-color: #f9f9f9;
    }}
    .stats-table tr:hover {{
        background-color: #e0e0e0;
    }}
    </style>

    <table class="stats-table">
        <tr>
            <th>Total Artists</th>
            <th>Total Tracks</th>
            <th>Total Albums</th>
            <th>Number of Unique Genres</th>
            <th>Avg Artist Popularity</th>
            <th>Avg Track Popularity</th>
        </tr>
        <tr>
            <td>{total_artists}</td>
            <td>{total_tracks}</td>
            <td>{total_albums}</td>
            <td>{num_unique_genres}</td>
            <td>{avg_artist_pop}</td>
            <td>{avg_track_pop}</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    # Insight Section
    st.header("Initial Insights")

    st.write("""
    1. The Spotify dataset contains a diverse set of artists, tracks, and albums.
    2. Popularity distribution shows how tracks are spread across low to high popularity ranges.
    3. The scatter plot helps identify whether more popular artists tend to produce more popular tracks.
    4. Further analysis explores collaborations, audio features, and trends over time.
    """)

    st.header("Top 10 Artists by Popularity")

    top_10_artists(artists)

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
    st.header("Artist vs Track Popularity")

    # Merge tables correctly
    merged = albums.merge(tracks, left_on="track_id", right_on="id")
    merged = merged.merge(artists, left_on="artist_0", right_on="name")

    # Drop missing values
    merged = merged.dropna(subset=["artist_popularity", "track_popularity"])

    # Compute correlation
    correlation = merged["artist_popularity"].corr(merged["track_popularity"])
    st.write(f"Correlation: {correlation:.2f}")

    # Create plot
    fig2, ax2 = plt.subplots(figsize=(8,6))

    ax2.scatter(
        merged["artist_popularity"],
        merged["track_popularity"],
        alpha=0.3,
        color="dodgerblue"
    )

    # Regression line
    sns.regplot(
        x="artist_popularity",
        y="track_popularity",
        data=merged,
        scatter=False,
        ax=ax2,
        color="red",
        line_kws={"linewidth":2}
    )

    ax2.set_xlabel("Artist Popularity")
    ax2.set_ylabel("Track Popularity")
    ax2.set_title("Artist vs Track Popularity")
    ax2.grid(True)

    st.pyplot(fig2)

# Run App
if __name__ == "__main__":
    main()