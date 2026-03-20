import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import your custom modules
from analysis import top_10_artists
import data_wrangling as dw

# --- Load Data Function ---
@st.cache_data
def load_and_clean_data():
    # Connect to database (ensure the path matches your environment)
    conn = sqlite3.connect("data/spotify_database.db")
    
    # Load raw data
    df_artists = pd.read_sql("SELECT * FROM artist_data", conn)
    df_tracks = pd.read_sql("SELECT * FROM tracks_data", conn)
    df_albums = pd.read_sql("SELECT * FROM albums_data", conn)
    df_features = pd.read_sql("SELECT * FROM features_data", conn)
    conn.close()

    # Apply your cleaning functions from data_wrangling.py
    df_features = dw.clean_features(df_features)
    df_tracks = dw.clean_tracks(df_tracks)
    df_albums = dw.clean_albums(df_albums)
    df_artists = dw.clean_artists(df_artists)
    
    return df_artists, df_tracks, df_albums, df_features

# --- Basic Dashboard ---
def basic_dashboard(artists, tracks, albums):
    st.title("Spotify Data Analysis Dashboard")

    # --- Compute genre counts ---
    genre_cols = [col for col in ["genre_1","genre_2","genre_3","genre_4","genre_5","genre_6"] if col in artists.columns]
    
    genre_counts = Counter()
    for row in artists[genre_cols].values:
        genres = [g for g in row if pd.notna(g) and g != ""]
        genre_counts.update(genres)
    
    genre_df = pd.DataFrame(genre_counts.items(), columns=["Genre","Count"]).sort_values(by="Count", ascending=False).reset_index(drop=True)
    
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

    # --- Insight Section ---
    st.header("Initial Insights")
    st.write("""
        1. The Spotify dataset contains a diverse set of artists, tracks, and albums.
        2. Popularity distribution shows how tracks are spread across low to high popularity ranges.
        3. Top artists, albums, and genres can be analyzed for popularity trends.
    """)

    # --- Tabs for Top 10 ---
    st.header("Top 10 Insights")
    tab1, tab2, tab3 = st.tabs(["Artists", "Albums", "Genres"])

    # Top 10 Artists
    with tab1:
        st.subheader("Top 10 Artists by Popularity")
        top_10_artists(artists) 

    # Top 10 Albums
    with tab2:
        st.subheader("Top 10 Albums by Popularity")
        album_name_col = [c for c in albums.columns if "name" in c.lower()][0]
        album_pop_col = [c for c in albums.columns if "popularity" in c.lower()][0]
        top_albums = albums[[album_name_col, album_pop_col]].dropna().sort_values(by=album_pop_col, ascending=False).head(10)
        st.dataframe(top_albums.reset_index(drop=True))

        # Bar chart
        fig_albums, ax_albums = plt.subplots(figsize=(8,5))
        ax_albums.bar(top_albums[album_name_col], top_albums[album_pop_col], color='green')
        ax_albums.set_xlabel("Album")
        ax_albums.set_ylabel("Popularity")
        ax_albums.set_title("Top 10 Albums")
        plt.xticks(rotation=45)
        st.pyplot(fig_albums)

    # Top 10 Genres
    with tab3:
        st.subheader("Top 10 Most Common Genres")
        top_genres = genre_df.head(10)
        st.dataframe(top_genres.reset_index(drop=True))

        # Bar chart
        fig_genres, ax_genres = plt.subplots(figsize=(8,5))
        ax_genres.bar(top_genres['Genre'], top_genres['Count'], color='orange')
        ax_genres.set_xlabel("Genre")
        ax_genres.set_ylabel("Count")
        ax_genres.set_title("Top 10 Genres")
        plt.xticks(rotation=45)
        st.pyplot(fig_genres)

# --- Advanced Dashboard ---
def advanced_dashboard(artists, tracks, albums, features):
    st.title("🎵 Spotify Database Statistics Dashboard")
    st.markdown("Explore trends, explicit content ratios, and audio features across eras.")
    
    # --- Ensure 'year' column exists ---
    if 'year' not in albums.columns:
        # Try common alternatives
        if 'release_date' in albums.columns:
            albums['year'] = pd.to_datetime(albums['release_date'], errors='coerce').dt.year
        elif 'album_year' in albums.columns:
            albums['year'] = albums['album_year']
        else:
            st.error("No 'year' or 'release_date' column found in albums data.")
            return  # Stop execution if we can't get a year
    
    # Sidebar info
    st.sidebar.header("Dashboard Controls")
    st.sidebar.write(f"Total Tracks: {len(tracks)}")
    st.sidebar.write(f"Total Albums: {len(albums.drop_duplicates('album_id'))}")

    # --- Album Search ---
    st.header("💿 Album Search")
    col1, col2 = st.columns(2)
    album_input = col1.text_input("Enter Album Name:", value="Black Sand")
    
    if album_input:
        album_data = albums[albums['album_name'].str.lower() == album_input.lower()]
        if not album_data.empty:
            avg_pop = album_data['album_popularity'].mean()
            col1.metric("Average Album Popularity", f"{avg_pop:.2f}/100")

            st.subheader(f"Audio Features for '{album_input}'")
            album_tracks_features = album_data.merge(features, left_on="track_id", right_on="id")
            if not album_tracks_features.empty:
                feature_cols = ["danceability", "energy", "speechiness", 
                                "acousticness", "instrumentalness", "liveness", "valence"]
                stats = album_tracks_features[feature_cols].mean()
                st.bar_chart(stats)
            else:
                st.warning("No audio feature data found for this album.")
        else:
            st.error("Album not found.")

    st.divider()

    # --- Explicit Content Analysis ---
    st.header("🔞 Explicit Content Analysis")

    # Slider for years
    min_year = int(albums['year'].min())
    max_year = int(albums['year'].max())
    year_range = st.slider(
        "Select Year Range for Explicit Analysis:",
        min_year,
        max_year,
        (min_year, max_year)
    )

    filtered_albums = albums[(albums['year'] >= year_range[0]) & (albums['year'] <= year_range[1])]
    filtered_tracks_data = filtered_albums[['track_id', 'artist_0']].merge(
        tracks[['id', 'explicit', 'track_popularity']], 
        left_on='track_id', 
        right_on='id'
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Popularity Comparison")
        if not filtered_tracks_data.empty:
            exp_pop = filtered_tracks_data.groupby("explicit")["track_popularity"].mean().reset_index()
            colors = ['#1DB954' if x == 'false' else '#191414' for x in exp_pop['explicit']]
            fig, ax = plt.subplots()
            ax.bar(exp_pop["explicit"], exp_pop["track_popularity"], color=colors)
            ax.set_ylabel("Avg Popularity")
            ax.set_title(f"Explicit vs Non-Explicit ({year_range[0]}-{year_range[1]})")
            st.pyplot(fig)
        else:
            st.info("No data available for this year range.")

    with c2:
        st.subheader("Most Explicit Artists")
        if not filtered_tracks_data.empty:
            explicit_stats = filtered_tracks_data.groupby('artist_0')['explicit'].apply(
                lambda x: (x == 'true').sum() / len(x)
            ).sort_values(ascending=False).head(10)
            st.table(explicit_stats.rename("Explicit Proportion"))
        else:
            st.info("No artist data for this range.")

    st.divider()

    # --- Musical Evolution by Era ---
    st.header("⏳ Musical Evolution by Era")
    albums['era'] = (albums['year'] // 10 * 10).astype(int).astype(str) + "s"
    era_features = albums.merge(features, left_on="track_id", right_on="id")

    selected_feature = st.selectbox(
        "Select Feature to view over Eras:", 
        ["loudness", "acousticness", "tempo", "energy", "danceability"]
    )

    era_trend = era_features.groupby("era")[selected_feature].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(era_trend["era"], era_trend[selected_feature], color="#1DB954")
    ax2.set_title(f"Average {selected_feature.title()} per Era")
    st.pyplot(fig2)

    st.divider()

    # --- Individual Track Ranking ---
    st.header("🔍 Individual Track Ranking")
    song_input = st.text_input("Enter Track Name:", value="")
    
    if song_input:
        track_row = albums[albums['track_name'].str.lower() == song_input.lower()]
        if not track_row.empty:
            tid = track_row.iloc[0]['track_id']
            rank_features = ["danceability", "energy", "speechiness", "acousticness"]
            labels = ["Very Low", "Low", "Medium", "High", "Very High"]
            ranked_df = dw.add_feature_ranks(features, rank_features, labels)
            track_rank = ranked_df[ranked_df['id'] == tid]
            if not track_rank.empty:
                st.write(f"Scoring for {song_input}:")
                cols = st.columns(len(rank_features))
                for i, feat in enumerate(rank_features):
                    val = track_rank[f"{feat}_rank"].values[0]
                    cols[i].metric(feat.title(), val)
            else:
                st.warning("Feature data for this specific track is missing.")
        else:
            st.info("Start typing a song name to see its rank.")

# --- Main App ---
def main():
    st.set_page_config(page_title="Spotify Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Dashboard Page:", ["Basic Dashboard", "Advanced Dashboard"])

    # Load data once
    artists, tracks, albums, features = load_and_clean_data()

    if page == "Basic Dashboard":
        basic_dashboard(artists, tracks, albums)
    elif page == "Advanced Dashboard":
        advanced_dashboard(artists, tracks, albums, features)

if __name__ == "__main__":
    main()

    """Sidebar for albums, genres and artists (add code)"""