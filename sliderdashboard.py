import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data_wrangling as dw

# Page Config
st.set_page_config(page_title="Spotify Insights Dashboard", layout="wide")

@st.cache_data
def load_and_clean_data():
    # Connect to database
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
    
    # Pre-process dates here so they are available globally in the app
    df_albums['release_date'] = pd.to_datetime(df_albums['release_date'], errors='coerce')
    df_albums['year'] = df_albums['release_date'].dt.year
    
    return df_artists, df_tracks, df_albums, df_features

def main():
    st.title("🎵 Spotify Database Statistics Dashboard")
    st.markdown("Explore trends, explicit content ratios, and audio features across eras.")
    
    artists, tracks, albums, features = load_and_clean_data()

    # Sidebar for Navigation or Global Info
    st.sidebar.header("Dashboard Controls")
    st.sidebar.write(f"Total Tracks: {len(tracks)}")
    st.sidebar.write(f"Total Albums: {len(albums.drop_duplicates('album_id'))}")

    # --- RUBRIC 1 & 4: Album Popularity & Feature Scores ---
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
                feature_cols = ["danceability", "energy", "speechiness", "acousticness", 
                                "instrumentalness", "liveness", "valence"]
                stats = album_tracks_features[feature_cols].mean()
                st.bar_chart(stats)
            else:
                st.warning("No audio feature data found for this album.")
        else:
            st.error("Album not found.")

    st.divider()

    # --- RUBRIC 2 & 3: Explicit Content Analysis (WITH YEAR SLIDER) ---
    st.header("🔞 Explicit Content Analysis")
    
    # Get range for slider
    min_year = int(albums['year'].min())
    max_year = int(albums['year'].max())

    year_range = st.slider(
        "Select Year Range for Explicit Analysis:",
        min_year,
        max_year,
        (min_year, max_year)
    )

    # Filter data based on slider
    filtered_albums = albums[(albums['year'] >= year_range[0]) & (albums['year'] <= year_range[1])]
    
    # Merge filtered albums with tracks to get popularity and explicit status within that timeframe
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
            
            fig, ax = plt.subplots()
            # Ensure order for colors if one category is missing in the range
            colors = ['#1DB954' if x == 'false' else '#191414' for x in exp_pop['explicit']]
            
            ax.bar(exp_pop["explicit"], exp_pop["track_popularity"], color=colors)
            ax.set_ylabel("Avg Popularity")
            ax.set_title(f"Explicit vs Non-Explicit ({year_range[0]}-{year_range[1]})")
            st.pyplot(fig)
        else:
            st.info("No data available for this year range.")

    with c2:
        st.subheader("Most Explicit Artists")
        if not filtered_tracks_data.empty:
            # Calculate proportion
            explicit_stats = filtered_tracks_data.groupby('artist_0')['explicit'].apply(
                lambda x: (x == 'true').sum() / len(x)
            ).sort_values(ascending=False).head(10)
            
            st.table(explicit_stats.rename("Explicit Proportion"))
        else:
            st.info("No artist data for this range.")

    st.divider()

    # --- RUBRIC 5: Era (Decade) Analysis ---
    st.header("⏳ Musical Evolution by Era")
    
    # Logic for era remains global (not affected by the slider above)
    albums['era'] = (albums['year'] // 10 * 10).fillna(0).astype(int).astype(str) + "s"
    
    era_features = albums.merge(features, left_on="track_id", right_on="id")
    
    selected_feature = st.selectbox("Select Feature to view over Eras:", 
                                    ["loudness", "acousticness", "tempo", "energy", "danceability"])
    
    era_trend = era_features.groupby("era")[selected_feature].mean().reset_index()
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(era_trend["era"], era_trend[selected_feature], color="#1DB954")
    ax2.set_title(f"Average {selected_feature.title()} per Era")
    st.pyplot(fig2)

    st.divider()

    # --- RUBRIC 6: Song Feature Scoring (Rankings) ---
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

if __name__ == "__main__":
    main()