import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from collections import Counter

import data_wrangling as dw


st.set_page_config(page_title="Spotify Dashboard", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background-color: #0E1117;
            color: white;
        }
        h1, h2, h3 {
            color: #1DB954;
        }
        [data-testid="stMetricValue"] {
            color: #1DB954;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_clean_data():
    conn = sqlite3.connect("data/spotify_database.db")
    artists = pd.read_sql("SELECT * FROM artist_data", conn)
    tracks = pd.read_sql("SELECT * FROM tracks_data", conn)
    albums = pd.read_sql("SELECT * FROM albums_data", conn)
    features = pd.read_sql("SELECT * FROM features_data", conn)
    conn.close()

    artists = dw.clean_artists(artists)
    tracks = dw.clean_tracks(tracks)
    albums = dw.clean_albums(albums)
    features = dw.clean_features(features)

    albums = albums.copy()
    albums["release_year"] = pd.to_datetime(
        albums["release_date"], errors="coerce"
    ).dt.year

    return artists, tracks, albums, features


def get_filtered_albums(albums, year_range):
    filtered = albums.dropna(subset=["release_year"]).copy()
    filtered["release_year"] = filtered["release_year"].astype(int)
    return filtered[
        filtered["release_year"].between(year_range[0], year_range[1])
    ]


def get_joined_df(tracks, albums, features, year_range):
    filtered_albums = get_filtered_albums(albums, year_range)

    return (
        filtered_albums.merge(
            tracks[["id", "track_popularity", "explicit"]],
            left_on="track_id",
            right_on="id",
            how="left",
        )
        .merge(
            features,
            left_on="track_id",
            right_on="id",
            how="left",
            suffixes=("_track", "_feature"),
        )
    )


def build_genre_df(artists):
    genre_cols = [col for col in artists.columns if col.startswith("genre_")]
    genre_counts = Counter()

    for row in artists[genre_cols].values:
        genre_counts.update(
            genre for genre in row if pd.notna(genre) and str(genre).strip()
        )

    return (
        pd.DataFrame(genre_counts.items(), columns=["Genre", "Count"])
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )


def get_genre_options(artists):
    genre_cols = [col for col in artists.columns if col.startswith("genre_")]
    genres = set()

    for col in genre_cols:
        genres.update(
            genre
            for genre in artists[col].dropna().astype(str).str.strip()
            if genre
        )

    return sorted(genres)


def get_artist_options(artists):
    return sorted(artists["name"].dropna().drop_duplicates().tolist())


def get_track_options(albums, artist_name, year_range):
    if not artist_name:
        return [""]

    artist_albums = get_filtered_albums(albums, year_range)
    artist_albums = artist_albums[artist_albums["artist_0"] == artist_name]

    tracks = (
        artist_albums["track_name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return [""] + tracks


def format_release_date(series):
    dates = pd.to_datetime(series, errors="coerce", utc=True)
    return dates.dt.strftime("%d-%m-%Y").fillna("")


def feature_distribution_chart(features, selected_feature):
    feature_data = features[selected_feature].dropna()

    if feature_data.empty:
        st.info("No feature distribution available.")
        return

    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.hist(feature_data, bins=20, color="#1DB954")
    ax.set_title(f"{selected_feature.title()} Distribution")
    ax.set_xlabel(selected_feature.title())
    ax.set_ylabel("Number of Tracks")
    plt.tight_layout()
    st.pyplot(fig)


def genre_diversity_summary(artists):
    genre_cols = [col for col in artists.columns if col.startswith("genre_")]
    summary = (
        artists[genre_cols]
        .apply(
            lambda row: sum(
                pd.notna(value) and str(value).strip() != "" for value in row
            ),
            axis=1,
        )
        .rename("num_genres")
        .to_frame()
        .join(artists["artist_popularity"])
        .groupby("num_genres", as_index=False)["artist_popularity"]
        .mean()
        .sort_values("num_genres")
    )

    if summary.empty:
        st.info("No genre diversity summary available.")
        return

    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.bar(summary["num_genres"].astype(int).astype(str), summary["artist_popularity"], color="#1DB954")
    ax.set_title("Popularity vs Genre Diversity")
    ax.set_xlabel("Number of Genres")
    ax.set_ylabel("Avg Artist Popularity")
    plt.tight_layout()
    st.pyplot(fig)


def overperforming_artists_table(artists):
    df = artists[["name", "artist_popularity", "followers"]].dropna().copy()
    df = df[df["followers"] > 0]

    if df.empty:
        st.info("No overperforming artists available.")
        return

    df["log_followers"] = np.log(df["followers"])
    X = sm.add_constant(df["log_followers"])
    model = sm.OLS(df["artist_popularity"], X).fit()

    df["overperformance"] = df["artist_popularity"] - model.predict(X)

    table = (
        df.sort_values("overperformance", ascending=False)
        .drop_duplicates(subset=["name"])
        .head(10)
        .assign(
            followers=lambda x: x["followers"].astype(int).map(lambda value: f"{value:,}"),
            artist_popularity=lambda x: x["artist_popularity"].round(2),
            overperformance=lambda x: x["overperformance"].round(2),
        )
        .rename(
            columns={
                "name": "Artist",
                "artist_popularity": "Popularity",
                "followers": "Followers",
                "overperformance": "Overperformance",
            }
        )[["Artist", "Popularity", "Followers", "Overperformance"]]
    )

    st.dataframe(table, width="content", hide_index=True)


def track_rating_section(albums, features, selected_track):
    st.subheader("Track Feature Rating")

    if not selected_track:
        st.info("Select a track from the sidebar to view its feature ratings.")
        return

    track_id = albums.loc[albums["track_name"] == selected_track, "track_id"].iloc[0]
    ranked_df = dw.add_feature_ranks(
        features,
        ["danceability", "energy", "speechiness", "acousticness"],
        ["Very Low", "Low", "Medium", "High", "Very High"],
    )
    track_rank = ranked_df[ranked_df["id"] == track_id]

    if track_rank.empty:
        st.info("No feature data available for this track.")
        return

    cols = st.columns(4)
    for i, feature in enumerate(
        ["danceability", "energy", "speechiness", "acousticness"]
    ):
        cols[i].metric(feature.title(), track_rank[f"{feature}_rank"].iloc[0])


def overview_page(artists, tracks, albums, year_range):
    st.title("Spotify Dashboard")
    st.markdown(
        "This page summarises the dataset with overall statistics and key visual insights."
    )

    filtered_albums = get_filtered_albums(albums, year_range)
    filtered_tracks = tracks[tracks["id"].isin(filtered_albums["track_id"].dropna().unique())]
    genre_df = build_genre_df(artists)

    st.header("General Statistics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Artists", artists["name"].nunique())
    c2.metric("Tracks in selected years", filtered_tracks["id"].nunique())
    c3.metric("Albums in selected years", filtered_albums["album_id"].nunique())

    c4, c5, c6 = st.columns(3)
    c4.metric("Unique Genres", genre_df["Genre"].nunique())
    c5.metric(
        "Avg Track Popularity",
        f'{filtered_tracks["track_popularity"].mean():.2f}'
        if not filtered_tracks.empty
        else "N/A",
    )
    c6.metric(
        "Avg Album Popularity",
        f'{filtered_albums["album_popularity"].mean():.2f}'
        if not filtered_albums.empty
        else "N/A",
    )

    st.divider()
    st.header("Graphical Summary")

    left, right = st.columns(2)

    with left:
        top_genres = genre_df.head(10)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(top_genres["Genre"], top_genres["Count"], color="#1DB954")
        ax.set_title("Top 10 Genres")
        ax.set_xlabel("Genre")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    with right:
        if filtered_albums.empty:
            st.info("No album data available for the selected years.")
        else:
            yearly_counts = (
                filtered_albums.groupby("release_year", as_index=False)["track_id"]
                .nunique()
                .sort_values("release_year")
            )

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(yearly_counts["release_year"], yearly_counts["track_id"])
            ax.set_title("Tracks Released by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of Tracks")
            plt.tight_layout()
            st.pyplot(fig)

    st.divider()
    st.subheader("Genre Diversity & Overperforming Artists")

    left, right = st.columns(2)
    with left:
        genre_diversity_summary(artists)
    with right:
        overperforming_artists_table(artists)


def feature_genre_page(
    artists, tracks, albums, features, selected_feature, selected_genre, year_range
):
    st.title("Feature & Genre Analysis")
    st.markdown(
        "Use the sidebar filters to explore feature patterns, genre summaries, and time-based trends."
    )

    merged = get_joined_df(tracks, albums, features, year_range)
    genre_cols = [col for col in artists.columns if col.startswith("genre_")]

    st.header("Feature Summary")

    if not selected_feature:
        st.info("Select a feature from the sidebar to view feature statistics.")
    else:
        c1, c2, c3 = st.columns(3)
        feature_series = merged[selected_feature].dropna()

        if feature_series.empty:
            c1.metric("Average", "N/A")
            c2.metric("Minimum", "N/A")
            c3.metric("Maximum", "N/A")
        else:
            c1.metric("Average", f"{feature_series.mean():.3f}")
            c2.metric("Minimum", f"{feature_series.min():.3f}")
            c3.metric("Maximum", f"{feature_series.max():.3f}")

        left, right = st.columns(2)

        with left:
            st.subheader(f"{selected_feature.title()} across years")
            yearly_feature = (
                merged.dropna(subset=["release_year", selected_feature])
                .groupby("release_year", as_index=False)[selected_feature]
                .mean()
                .sort_values("release_year")
            )

            if yearly_feature.empty:
                st.info("No feature data available for the selected years.")
            else:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(yearly_feature["release_year"], yearly_feature[selected_feature])
                ax.set_title(f"Average {selected_feature.title()} by Year")
                ax.set_xlabel("Year")
                ax.set_ylabel(selected_feature.title())
                plt.tight_layout()
                st.pyplot(fig)

        with right:
            st.subheader(f"Top tracks for {selected_feature}")
            top_tracks = (
                merged[["track_name", "artist_0", selected_feature]]
                .dropna()
                .sort_values(selected_feature, ascending=False)
                .drop_duplicates(subset=["track_name"])
                .head(10)
                .reset_index(drop=True)
            )

            if top_tracks.empty:
                st.info("No track data available for this feature.")
            else:
                st.dataframe(top_tracks, width="content", hide_index=True)

        st.divider()
        st.subheader(f"{selected_feature.title()} Distribution")
        spacer_left, center, spacer_right = st.columns([1, 2, 1])
        with center:
            feature_distribution_chart(features, selected_feature)

    st.divider()
    st.header("Genre Summary")

    if not selected_genre:
        st.info("Select a genre from the sidebar to view genre statistics.")
        return

    matching_mask = artists[genre_cols].apply(
        lambda col: col.astype(str).str.strip().str.lower() == selected_genre.lower()
    ).any(axis=1)
    matching_artists = artists[matching_mask]

    st.write(
        f"Artists found for genre **{selected_genre}**: {matching_artists['name'].nunique()}"
    )

    genre_names = matching_artists["name"].dropna().unique().tolist()
    genre_albums = get_filtered_albums(albums[albums["artist_0"].isin(genre_names)], year_range)
    genre_tracks = tracks[tracks["id"].isin(genre_albums["track_id"].dropna().unique())]

    c1, c2, c3 = st.columns(3)
    c1.metric("Artists", matching_artists["name"].nunique())
    c2.metric("Albums", genre_albums["album_id"].nunique())
    c3.metric(
        "Avg Track Popularity",
        f'{genre_tracks["track_popularity"].mean():.2f}'
        if not genre_tracks.empty
        else "N/A",
    )

    left, right = st.columns(2)

    with left:
        top_genre_artists = (
            matching_artists[["name", "artist_popularity", "followers"]]
            .drop_duplicates(subset=["name"])
            .sort_values(["artist_popularity", "followers"], ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        st.subheader("Top Artists in Selected Genre")
        st.dataframe(top_genre_artists, width="content", hide_index=True)

    with right:
        if genre_tracks.empty:
            st.info("No track data available for the selected genre and year range.")
        else:
            explicit_stats = (
                genre_tracks.assign(
                    explicit=genre_tracks["explicit"].astype(str).str.lower()
                )
                .groupby("explicit", as_index=False)["track_popularity"]
                .mean()
                .set_index("explicit")
                .reindex(["false", "true"])
                .reset_index()
            )

            explicit_stats["track_popularity"] = explicit_stats["track_popularity"].fillna(0)
            explicit_stats["explicit"] = explicit_stats["explicit"].map(
                {"false": "Non-Explicit", "true": "Explicit"}
            )

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(
                explicit_stats["explicit"],
                explicit_stats["track_popularity"],
                color=["#535353", "#1DB954"],
            )
            ax.set_title("Explicit vs Non-Explicit Popularity")
            ax.set_xlabel("Track Type")
            ax.set_ylabel("Average Popularity")
            plt.tight_layout()
            st.pyplot(fig)


def artist_analysis_page(
    artists, tracks, albums, features, artist_name, selected_feature, selected_track, year_range
):
    st.title("Artist Analysis")
    st.markdown("Select an artist to explore albums, tracks, and feature statistics.")

    if not artist_name:
        st.info("Select an artist from the sidebar.")
        return

    artist_row = artists.loc[artists["name"] == artist_name].drop_duplicates(subset=["name"]).iloc[0]
    artist_albums = get_filtered_albums(albums[albums["artist_0"] == artist_name], year_range)
    artist_tracks = tracks[tracks["id"].isin(artist_albums["track_id"].dropna().unique())]
    artist_merged = artist_albums.merge(features, left_on="track_id", right_on="id", how="left")

    st.header(artist_name)

    c1, c2, c3 = st.columns(3)
    c1.metric("Followers", f"{int(artist_row['followers']):,}")
    c2.metric("Artist Popularity", f"{artist_row['artist_popularity']:.2f}")
    c3.metric("Tracks in selected years", artist_tracks["id"].nunique())

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Albums")
        album_table = (
            artist_albums[["album_name", "release_date", "album_popularity"]]
            .dropna(subset=["album_name"])
            .drop_duplicates()
            .sort_values("album_popularity", ascending=False)
            .reset_index(drop=True)
        )

        if album_table.empty:
            st.info("No album data found for this artist in the selected years.")
        else:
            album_table = album_table.copy()
            album_table["release_date"] = format_release_date(album_table["release_date"])
            st.dataframe(album_table, width="content", hide_index=True)

    with right:
        st.subheader(
            f"Top 10 Albums by Average {selected_feature.title()}"
            if selected_feature
            else "Top 10 Albums"
        )

        if not selected_feature:
            st.info("Select a feature from the sidebar to compare albums.")
        else:
            feature_by_album = (
                artist_merged.dropna(subset=["album_name", selected_feature])
                .groupby("album_name", as_index=False)[selected_feature]
                .mean()
                .sort_values(selected_feature, ascending=False)
                .head(10)
            )

            if feature_by_album.empty:
                st.info("No feature data available for this artist.")
            else:
                feature_by_album[selected_feature] = feature_by_album[selected_feature].round(3)
                ranked_table = feature_by_album.rename(
                    columns={
                        "album_name": "Album",
                        selected_feature: f"Avg {selected_feature.title()}",
                    }
                )
                st.dataframe(ranked_table, width="content", hide_index=True)

    st.divider()
    st.subheader("Most Popular Tracks")

    top_tracks = (
        artist_albums[["track_name", "track_id"]]
        .merge(
            tracks[["id", "track_popularity", "explicit"]],
            left_on="track_id",
            right_on="id",
            how="left",
        )[["track_name", "track_popularity", "explicit"]]
        .dropna(subset=["track_name"])
        .drop_duplicates(subset=["track_name"])
        .sort_values("track_popularity", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    if top_tracks.empty:
        st.info("No track data available for this artist.")
    else:
        st.dataframe(top_tracks, width="content", hide_index=True)

    st.divider()
    track_rating_section(artist_albums, features, selected_track)


def main():
    artists, tracks, albums, features = load_and_clean_data()

    valid_years = albums["release_year"].dropna()
    min_year = int(valid_years.min()) if not valid_years.empty else 1900
    max_year = int(valid_years.max()) if not valid_years.empty else 2025

    st.sidebar.title("Filters")
    page = st.sidebar.radio(
        "Select page",
        ["Overview", "Feature & Genre Explorer", "Artist Search"],
    )

    st.sidebar.markdown("---")
    selected_feature = st.sidebar.selectbox(
        "Select a feature",
        [
            "",
            "danceability",
            "energy",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "loudness",
            "tempo",
        ],
    )
    selected_genre = st.sidebar.selectbox(
        "Select a genre",
        [""] + get_genre_options(artists),
    )
    artist_name = st.sidebar.selectbox(
        "Search for an artist",
        [""] + get_artist_options(artists),
    )

    st.sidebar.markdown("---")
    year_range = st.sidebar.slider(
        "Select year range",
        min_year,
        max_year,
        (min_year, max_year),
    )
    st.sidebar.caption(f"Years: {year_range[0]} to {year_range[1]}")

    selected_track = st.sidebar.selectbox(
        "Select a track",
        get_track_options(albums, artist_name, year_range),
    )

    if page == "Overview":
        overview_page(artists, tracks, albums, year_range)
    elif page == "Feature & Genre Explorer":
        feature_genre_page(
            artists,
            tracks,
            albums,
            features,
            selected_feature,
            selected_genre,
            year_range,
        )
    else:
        artist_analysis_page(
            artists,
            tracks,
            albums,
            features,
            artist_name,
            selected_feature,
            selected_track,
            year_range,
        )


if __name__ == "__main__":
    main()
