import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import data_wrangling as dw

# -----------------------------
# Theme constants
# -----------------------------
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BG = "#121212"
SPOTIFY_PANEL = "#181818"
SPOTIFY_SIDEBAR = "#000000"
SPOTIFY_TEXT = "white"
SPOTIFY_MUTED = "#B3B3B3"

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Spotify Insights Dashboard",
    page_icon="🎵",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {SPOTIFY_BG};
        color: {SPOTIFY_TEXT};
    }}

    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}

    h1, h2, h3, h4 {{
        color: {SPOTIFY_TEXT};
        padding-top: 0.2rem;
    }}

    p, label {{
        color: {SPOTIFY_TEXT} !important;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {SPOTIFY_SIDEBAR};
        border-right: 1px solid #202020;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.2rem;
    }}

    .sidebar-panel {{
        background-color: {SPOTIFY_PANEL};
        border: 1px solid #202020;
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        margin-bottom: 14px;
    }}

    .sidebar-title {{
        color: {SPOTIFY_TEXT};
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}

    .sidebar-subtitle {{
        color: {SPOTIFY_MUTED};
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }}

    .sidebar-section-title {{
        color: {SPOTIFY_GREEN};
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        letter-spacing: 0.3px;
    }}

    .sidebar-note {{
        color: {SPOTIFY_MUTED};
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }}

    .hero-card {{
        background: {SPOTIFY_GREEN};
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 1rem;
    }}

    .hero-card h2 {{
        color: black !important;
        margin: 0;
    }}

    .hero-card p {{
        color: black !important;
        margin: 8px 0 0 0;
        font-weight: 500;
    }}

    div[data-testid="stMetric"] {{
        background-color: {SPOTIFY_PANEL};
        border: 1.5px solid {SPOTIFY_GREEN};
        box-shadow: 0 0 0 1px rgba(29, 185, 84, 0.12);
        padding: 14px;
        border-radius: 12px;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {SPOTIFY_MUTED} !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {SPOTIFY_TEXT} !important;
    }}

    div[data-testid="stInfo"] {{
        background-color: #0f1c2e;
        border: 1px solid #1b3555;
        border-radius: 12px;
    }}

    div[data-testid="stWarning"] {{
        background-color: #1b1b1b;
        border: 1px solid #3a3a3a;
        border-radius: 12px;
    }}

    button[role="tab"] {{
        color: {SPOTIFY_TEXT} !important;
    }}

    button[aria-selected="true"] {{
        border-bottom: 2px solid {SPOTIFY_GREEN} !important;
    }}

    input[type="radio"] {{
        accent-color: {SPOTIFY_GREEN} !important;
    }}

    div[role="radiogroup"] label {{
        color: {SPOTIFY_TEXT} !important;
    }}

    div[data-testid="stSlider"] span {{
        color: {SPOTIFY_TEXT} !important;
    }}

    div[data-testid="stSlider"] [role="slider"] {{
        background-color: {SPOTIFY_GREEN} !important;
        border: 2px solid {SPOTIFY_GREEN} !important;
        box-shadow: 0 0 0 1px {SPOTIFY_GREEN} !important;
    }}

    div[data-baseweb="select"] > div {{
        background: #000000 !important;
        border: 1px solid {SPOTIFY_GREEN} !important;
        border-radius: 8px !important;
        color: {SPOTIFY_GREEN} !important;
    }}

    div[data-baseweb="select"] span {{
        color: {SPOTIFY_GREEN} !important;
    }}

    div[data-baseweb="select"] input {{
        color: {SPOTIFY_GREEN} !important;
        background-color: #000000 !important;
    }}

    table {{
        background-color: {SPOTIFY_PANEL} !important;
        color: {SPOTIFY_TEXT} !important;
        border-collapse: collapse !important;
        width: 100%;
    }}

    thead tr th {{
        background-color: #111111 !important;
        color: {SPOTIFY_GREEN} !important;
    }}

    tbody tr td {{
        background-color: {SPOTIFY_PANEL} !important;
        color: {SPOTIFY_TEXT} !important;
    }}

    div[data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #2b2b2b;
        background-color: {SPOTIFY_PANEL};
    }}

    .section-divider {{
        height: 1px;
        background: linear-gradient(to right, {SPOTIFY_GREEN}, transparent);
        margin: 0.6rem 0 1rem 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_and_clean_data():
    conn = sqlite3.connect("data/spotify_database.db")

    df_artists = pd.read_sql("SELECT * FROM artist_data", conn)
    df_tracks = pd.read_sql("SELECT * FROM tracks_data", conn)
    df_albums = pd.read_sql("SELECT * FROM albums_data", conn)
    df_features = pd.read_sql("SELECT * FROM features_data", conn)
    conn.close()

    df_features = dw.clean_features(df_features)
    df_tracks = dw.clean_tracks(df_tracks)
    df_albums = dw.clean_albums(df_albums)
    df_artists = dw.clean_artists(df_artists)

    return df_artists, df_tracks, df_albums, df_features


# -----------------------------
# Helper functions
# -----------------------------
def prepare_album_year(albums):
    albums = albums.copy()

    if "year" not in albums.columns:
        if "release_date" in albums.columns:
            albums["year"] = pd.to_datetime(
                albums["release_date"], errors="coerce"
            ).dt.year
        elif "album_year" in albums.columns:
            albums["year"] = pd.to_numeric(albums["album_year"], errors="coerce")

    return albums


def build_genre_df(artists):
    genre_cols = [
        col
        for col in ["genre_1", "genre_2", "genre_3", "genre_4", "genre_5", "genre_6"]
        if col in artists.columns
    ]

    genre_counts = Counter()
    for row in artists[genre_cols].fillna("").values:
        genres = [g for g in row if g != ""]
        genre_counts.update(genres)

    genre_df = (
        pd.DataFrame(genre_counts.items(), columns=["Genre", "Count"])
        .sort_values(by="Count", ascending=False)
        .reset_index(drop=True)
    )
    return genre_df


def filter_data_by_year(albums, year_range):
    albums_filtered = albums.copy()

    if "year" in albums_filtered.columns:
        albums_filtered = albums_filtered[
            (albums_filtered["year"] >= year_range[0])
            & (albums_filtered["year"] <= year_range[1])
        ]

    return albums_filtered


def style_figure(fig, ax):
    fig.patch.set_facecolor(SPOTIFY_BG)
    ax.set_facecolor(SPOTIFY_BG)


def style_axis(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, pad=12, color=SPOTIFY_TEXT)
    ax.set_xlabel(xlabel, color=SPOTIFY_TEXT)
    ax.set_ylabel(ylabel, color=SPOTIFY_TEXT)
    ax.tick_params(colors=SPOTIFY_TEXT)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPOTIFY_GREEN)
    ax.spines["bottom"].set_color(SPOTIFY_GREEN)

    ax.grid(axis="y", alpha=0.18, color=SPOTIFY_GREEN)


def create_section_divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def get_artist_row(artists, artist_name):
    if not artist_name or "name" not in artists.columns:
        return pd.DataFrame()

    artist_match = artists[artists["name"].str.lower() == artist_name.lower()].copy()

    if artist_match.empty:
        return pd.DataFrame()

    return artist_match


def get_artist_track_data(albums, tracks, artist_name, year_range=None):
    if not artist_name:
        return pd.DataFrame()

    df = albums.copy()

    if year_range is not None and "year" in df.columns:
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    if "artist_0" not in df.columns or "track_id" not in df.columns:
        return pd.DataFrame()

    artist_tracks = df[df["artist_0"].str.lower() == artist_name.lower()].copy()

    if artist_tracks.empty:
        return pd.DataFrame()

    merged = artist_tracks.merge(tracks, left_on="track_id", right_on="id", how="left")

    return merged


def extract_artist_genres(artist_row):
    genre_cols = [
        col
        for col in ["genre_1", "genre_2", "genre_3", "genre_4", "genre_5", "genre_6"]
        if col in artist_row.columns
    ]

    genres = []
    for col in genre_cols:
        value = artist_row.iloc[0][col]
        if pd.notna(value) and str(value).strip() != "":
            genres.append(str(value))

    return genres


def get_album_filter_columns(albums):
    album_name_candidates = ["album_name", "name", "title"]
    album_name_col = next(
        (c for c in album_name_candidates if c in albums.columns),
        None,
    )
    if not album_name_col:
        album_name_col = next(
            (c for c in albums.columns if "name" in c.lower()),
            None,
        )

    artist_candidates = ["artist_name", "artist", "artist_0"]
    artist_col = next(
        (c for c in artist_candidates if c in albums.columns),
        None,
    )
    if not artist_col:
        artist_col = next(
            (
                c
                for c in albums.columns
                if "artist" in c.lower() and albums[c].dtype == "object"
            ),
            None,
        )

    popularity_col = next(
        (c for c in ["album_popularity", "popularity"] if c in albums.columns),
        None,
    )
    if not popularity_col:
        popularity_col = next(
            (c for c in albums.columns if "popularity" in c.lower()),
            None,
        )

    return album_name_col, artist_col, popularity_col


# -----------------------------
# Page creators
# -----------------------------
def create_overview(artists, tracks, albums, features, genre_df, year_range):
    st.markdown(
        """
        <div class="hero-card">
            <h2>Spotify Analytics Dashboard</h2>
            <p>
                Explore artist performance, genre patterns, album insights, and release trends
                in one interactive dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtered_albums = filter_data_by_year(albums, year_range)

    if "track_id" in filtered_albums.columns:
        filtered_track_ids = filtered_albums["track_id"].dropna().unique()
        filtered_tracks = tracks[tracks["id"].isin(filtered_track_ids)].copy()
    else:
        filtered_tracks = tracks.copy()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Overview")
        st.write(
            f"This page summarizes the Spotify dataset for releases between **{year_range[0]} and {year_range[1]}**. "
            "Use the control panel to switch between pages and apply filters."
        )

    create_section_divider()

    st.markdown("## At a glance")
    st.markdown(
        "These summary indicators give a quick snapshot of the Spotify dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    total_artists = (
        artists["name"].nunique() if "name" in artists.columns else len(artists)
    )
    total_tracks = (
        filtered_tracks["id"].nunique()
        if "id" in filtered_tracks.columns
        else len(filtered_tracks)
    )
    total_albums = (
        filtered_albums["album_id"].nunique()
        if "album_id" in filtered_albums.columns
        else len(filtered_albums)
    )
    avg_artist_pop = (
        round(artists["artist_popularity"].mean(), 1)
        if "artist_popularity" in artists.columns
        else None
    )
    avg_track_pop = (
        round(filtered_tracks["track_popularity"].mean(), 1)
        if "track_popularity" in filtered_tracks.columns
        else None
    )
    unique_genres = genre_df["Genre"].nunique()

    c1.metric("Artists", total_artists)
    c2.metric("Tracks", total_tracks)
    c3.metric("Albums", total_albums)
    c4.metric("Genres", unique_genres)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Avg artist popularity", avg_artist_pop if avg_artist_pop is not None else "N/A"
    )
    c6.metric(
        "Avg track popularity", avg_track_pop if avg_track_pop is not None else "N/A"
    )

    if "year" in albums.columns and albums["year"].notna().any():
        c7.metric("First year", int(albums["year"].min()))
        c8.metric("Last year", int(albums["year"].max()))
    else:
        c7.metric("First year", "N/A")
        c8.metric("Last year", "N/A")

    create_section_divider()

    st.markdown("## Quick visual summaries")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 10 genres")
        top_genres = genre_df.head(10)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        style_figure(fig, ax)
        ax.barh(
            top_genres["Genre"][::-1],
            top_genres["Count"][::-1],
            color=SPOTIFY_GREEN,
        )
        style_axis(ax, "Most common genres", "Artist count", "")
        plt.tight_layout()
        st.pyplot(fig)

    with col_b:
        st.subheader("Track popularity distribution")
        if not filtered_tracks.empty and "track_popularity" in filtered_tracks.columns:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            style_figure(fig, ax)
            ax.hist(
                filtered_tracks["track_popularity"].dropna(),
                bins=20,
                color=SPOTIFY_GREEN,
            )
            style_axis(
                ax,
                "How popularity is distributed",
                "Track popularity",
                "Number of tracks",
            )
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info(
                "No track data is available for the selected year range. Try expanding the filter."
            )

    create_section_divider()

    st.markdown("## Top insights")
    tab1, tab2, tab3 = st.tabs(["Artists", "Albums", "Genres"])

    with tab1:
        st.subheader("Top 10 artists by popularity")

        if "name" in artists.columns and "artist_popularity" in artists.columns:
            top_artists_df = (
                artists[["name", "artist_popularity"]]
                .dropna()
                .drop_duplicates(subset=["name"])
                .sort_values(by="artist_popularity", ascending=False)
                .head(10)
            )

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.table(
                    top_artists_df.rename(
                        columns={"name": "Artist", "artist_popularity": "Popularity"}
                    ).reset_index(drop=True)
                )

            with right_col:
                fig, ax = plt.subplots(figsize=(5.4, 3.4))
                style_figure(fig, ax)
                ax.barh(
                    top_artists_df["name"][::-1],
                    top_artists_df["artist_popularity"][::-1],
                    color=SPOTIFY_GREEN,
                )
                style_axis(ax, "Top artists", "Popularity", "")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Artist name or popularity columns were not found.")

    with tab2:
        st.subheader("Top 10 albums by popularity")
        album_name_col = next(
            (c for c in filtered_albums.columns if "name" in c.lower()), None
        )
        album_pop_col = next(
            (c for c in filtered_albums.columns if "popularity" in c.lower()), None
        )

        if album_name_col and album_pop_col:
            top_albums = (
                filtered_albums[[album_name_col, album_pop_col]]
                .dropna()
                .drop_duplicates()
                .sort_values(by=album_pop_col, ascending=False)
                .head(10)
            )

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.table(top_albums.reset_index(drop=True))

            with right_col:
                fig, ax = plt.subplots(figsize=(5.4, 3.4))
                style_figure(fig, ax)
                ax.barh(
                    top_albums[album_name_col][::-1],
                    top_albums[album_pop_col][::-1],
                    color=SPOTIFY_GREEN,
                )
                style_axis(ax, "Top albums", "Popularity", "")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Album name or popularity columns were not found in the data.")

    with tab3:
        st.subheader("Top 10 most common genres")

        top_genres_table = genre_df.head(10).copy()

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.table(top_genres_table)

        with right_col:
            fig, ax = plt.subplots(figsize=(5.4, 3.4))
            style_figure(fig, ax)
            ax.barh(
                top_genres_table["Genre"][::-1],
                top_genres_table["Count"][::-1],
                color=SPOTIFY_GREEN,
            )
            style_axis(ax, "Top genres", "Artist count", "")
            plt.tight_layout()
            st.pyplot(fig)

    create_section_divider()

    st.markdown("### What this dashboard helps answer")
    st.markdown("""
        - Which artists, albums, and genres stand out most?
        - How is popularity distributed across tracks?
        - How do patterns change when you restrict the year range?
        - Which areas should a Spotify business analyst investigate next?
        """)


def create_artist_explorer(
    artists, tracks, albums, features, artist_search, year_range
):
    st.title("🎤 Artist Explorer")
    st.caption(
        "Use the control panel to choose an artist and explore popularity, genres, track performance, and explicitness."
    )

    if not artist_search:
        st.info(
            "Choose an artist from the control panel to explore track performance, genres, and summary metrics."
        )
        return

    artist_row = get_artist_row(artists, artist_search)

    if artist_row.empty:
        st.warning("That artist was not found in the artist dataset.")
        return

    artist_tracks = get_artist_track_data(albums, tracks, artist_search, year_range)
    genres = extract_artist_genres(artist_row)

    left, right = st.columns([2, 1])

    with left:
        st.subheader(artist_search)

        followers = (
            artist_row["followers"].iloc[0]
            if "followers" in artist_row.columns
            else None
        )
        artist_pop = (
            artist_row["artist_popularity"].iloc[0]
            if "artist_popularity" in artist_row.columns
            else None
        )

        st.write(
            f"Showing artist-level and track-level information for **{artist_search}** "
            f"within the selected years **{year_range[0]}–{year_range[1]}**."
        )

        if genres:
            st.write("**Genres:** " + ", ".join(genres))
        else:
            st.write("**Genres:** No genre information available")

    with right:
        st.markdown("### Artist summary")
        st.metric(
            "Artist popularity",
            round(float(artist_pop), 1) if pd.notna(artist_pop) else "N/A",
        )
        st.metric("Followers", f"{int(followers):,}" if pd.notna(followers) else "N/A")
        st.metric("Tracks in range", len(artist_tracks))

    if artist_tracks.empty:
        st.warning(
            "No matching tracks were found for this artist in the selected year range."
        )
        return

    explicit_rate = None
    avg_track_pop = None

    if "explicit" in artist_tracks.columns:
        explicit_rate = (
            artist_tracks["explicit"].astype(str).str.lower() == "true"
        ).mean()

    if "track_popularity" in artist_tracks.columns:
        avg_track_pop = artist_tracks["track_popularity"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Average track popularity",
        round(float(avg_track_pop), 1) if pd.notna(avg_track_pop) else "N/A",
    )
    c2.metric(
        "Explicit track share",
        f"{explicit_rate * 100:.1f}%" if explicit_rate is not None else "N/A",
    )
    c3.metric(
        "Unique albums",
        (
            artist_tracks["album_id"].nunique()
            if "album_id" in artist_tracks.columns
            else "N/A"
        ),
    )

    create_section_divider()

    st.markdown("### Artist insights")
    tab1, tab2, tab3 = st.tabs(["Top Tracks", "Explicitness", "Audio Features"])

    with tab1:
        st.subheader("Most popular tracks")

        track_name_col = "track_name" if "track_name" in artist_tracks.columns else None

        if track_name_col and "track_popularity" in artist_tracks.columns:
            top_tracks = (
                artist_tracks[[track_name_col, "track_popularity"]]
                .dropna()
                .drop_duplicates()
                .sort_values(by="track_popularity", ascending=False)
                .head(10)
            )

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.table(
                    top_tracks.rename(
                        columns={
                            track_name_col: "Track",
                            "track_popularity": "Popularity",
                        }
                    ).reset_index(drop=True)
                )

            with right_col:
                fig, ax = plt.subplots(figsize=(5.4, 3.4))
                style_figure(fig, ax)
                ax.barh(
                    top_tracks[track_name_col][::-1],
                    top_tracks["track_popularity"][::-1],
                    color=SPOTIFY_GREEN,
                )
                style_axis(ax, f"Top tracks for {artist_search}", "Popularity", "")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Track popularity data is not available for this artist.")

    with tab2:
        st.subheader("Explicit vs non-explicit tracks")

        if "explicit" in artist_tracks.columns:
            explicit_counts = (
                artist_tracks["explicit"]
                .astype(str)
                .str.lower()
                .value_counts()
                .rename_axis("explicit")
                .reset_index(name="count")
            )

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.table(explicit_counts)

            with right_col:
                fig, ax = plt.subplots(figsize=(5.4, 3.4))
                style_figure(fig, ax)
                ax.bar(
                    explicit_counts["explicit"],
                    explicit_counts["count"],
                    color=SPOTIFY_GREEN,
                )
                style_axis(
                    ax, f"Explicitness for {artist_search}", "Explicit", "Track count"
                )
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Explicitness data is not available.")

    with tab3:
        st.subheader("Average audio profile")

        if "track_id" in artist_tracks.columns and "id" in features.columns:
            artist_features = artist_tracks.merge(
                features, left_on="track_id", right_on="id", how="left"
            )

            feature_cols = [
                "danceability",
                "energy",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "liveness",
                "valence",
            ]
            available_feature_cols = [
                c for c in feature_cols if c in artist_features.columns
            ]

            if available_feature_cols:
                feature_means = (
                    artist_features[available_feature_cols]
                    .mean()
                    .sort_values(ascending=False)
                )

                left_col, right_col = st.columns([1, 1])

                with left_col:
                    st.table(
                        feature_means.reset_index().rename(
                            columns={"index": "Feature", 0: "Average score"}
                        )
                    )

                with right_col:
                    fig, ax = plt.subplots(figsize=(5.4, 3.4))
                    style_figure(fig, ax)
                    ax.bar(
                        feature_means.index,
                        feature_means.values,
                        color=SPOTIFY_GREEN,
                    )
                    style_axis(
                        ax,
                        f"Audio features for {artist_search}",
                        "Feature",
                        "Average score",
                    )
                    plt.xticks(rotation=45, color=SPOTIFY_TEXT)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("No audio feature columns were available.")
        else:
            st.info("Feature data could not be linked for this artist.")


def create_genre_explorer(artists, selected_genre):
    st.title("🎧 Genre Explorer")
    st.caption(
        "Use the control panel to explore artist counts, popularity, and top performers by genre."
    )

    if not selected_genre:
        st.info(
            "Choose a genre from the control panel to view artist counts, popularity, and top performers."
        )
        return

    genre_cols = [c for c in artists.columns if "genre_" in c]

    genre_artists = artists[
        artists[genre_cols]
        .fillna("")
        .apply(lambda row: selected_genre in row.values, axis=1)
    ]

    st.write(f"Showing insights for genre: **{selected_genre}**")

    if genre_artists.empty:
        st.warning("No artists found for this genre.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Number of artists", len(genre_artists))

    if "artist_popularity" in genre_artists.columns:
        avg_pop = genre_artists["artist_popularity"].mean()
        c2.metric("Average popularity", round(avg_pop, 1))
    else:
        c2.metric("Average popularity", "N/A")

    create_section_divider()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        if (
            "name" in genre_artists.columns
            and "artist_popularity" in genre_artists.columns
        ):
            top_artists = (
                genre_artists[["name", "artist_popularity"]]
                .dropna()
                .sort_values(by="artist_popularity", ascending=False)
                .head(10)
            )
            st.table(
                top_artists.rename(
                    columns={"name": "Artist", "artist_popularity": "Popularity"}
                ).reset_index(drop=True)
            )

    with right_col:
        if "artist_popularity" in genre_artists.columns:
            fig, ax = plt.subplots(figsize=(5.4, 3.4))
            style_figure(fig, ax)
            ax.hist(
                genre_artists["artist_popularity"].dropna(),
                bins=15,
                color=SPOTIFY_GREEN,
            )
            style_axis(
                ax,
                f"Popularity in {selected_genre}",
                "Popularity",
                "Artist count",
            )
            plt.tight_layout()
            st.pyplot(fig)


def create_feature_explorer(features, tracks, selected_feature):
    st.title("🎚️ Feature Explorer")
    st.caption(
        "Use the control panel to explore the distribution of audio features and their relationship with popularity."
    )

    if not selected_feature or selected_feature not in features.columns:
        st.info(
            "Choose an audio feature from the control panel to explore its distribution and relationship with popularity."
        )
        return

    st.write(f"Analyzing feature: **{selected_feature}**")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        style_figure(fig, ax)
        ax.hist(features[selected_feature].dropna(), bins=20, color=SPOTIFY_GREEN)
        style_axis(
            ax,
            f"Distribution of {selected_feature}",
            selected_feature,
            "Count",
        )
        plt.tight_layout()
        st.pyplot(fig)

    with right_col:
        if "track_popularity" in tracks.columns:
            merged = features.merge(tracks, left_on="id", right_on="id", how="left")

            fig, ax = plt.subplots(figsize=(5.4, 3.4))
            style_figure(fig, ax)
            ax.scatter(
                merged[selected_feature],
                merged["track_popularity"],
                alpha=0.35,
                color=SPOTIFY_GREEN,
            )
            ax.set_title(
                f"{selected_feature} vs popularity",
                fontsize=13,
                pad=12,
                color=SPOTIFY_TEXT,
            )
            ax.set_xlabel(selected_feature, color=SPOTIFY_TEXT)
            ax.set_ylabel("Track popularity", color=SPOTIFY_TEXT)
            ax.tick_params(colors=SPOTIFY_TEXT)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(SPOTIFY_GREEN)
            ax.spines["bottom"].set_color(SPOTIFY_GREEN)
            plt.tight_layout()
            st.pyplot(fig)


def create_time_trends(albums, year_range):
    st.title("⏳ Time Trends")
    st.caption("Explore how release activity and popularity change over time.")

    if "year" not in albums.columns:
        st.warning("No year information is available in the album dataset.")
        return

    filtered_albums = filter_data_by_year(albums, year_range).copy()

    if filtered_albums.empty:
        st.warning("No album data is available for the selected year range.")
        return

    st.write(
        f"Showing time-based trends for releases between **{year_range[0]}** and **{year_range[1]}**."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Albums in range", len(filtered_albums))
    c2.metric("Years covered", filtered_albums["year"].nunique())

    if "album_popularity" in filtered_albums.columns:
        c3.metric(
            "Avg album popularity",
            round(filtered_albums["album_popularity"].mean(), 1),
        )
    else:
        c3.metric("Avg album popularity", "N/A")

    create_section_divider()

    st.markdown("### Releases per year")
    yearly_counts = (
        filtered_albums.groupby("year")
        .size()
        .reset_index(name="album_count")
        .sort_values("year")
    )

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.table(yearly_counts.tail(10).reset_index(drop=True))

    with right_col:
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        style_figure(fig, ax)
        ax.plot(
            yearly_counts["year"],
            yearly_counts["album_count"],
            marker="o",
            color=SPOTIFY_GREEN,
        )
        style_axis(ax, "Releases per year", "Year", "Album count")
        plt.tight_layout()
        st.pyplot(fig)

    create_section_divider()

    st.markdown("### Average album popularity by year")

    popularity_col = None
    if "album_popularity" in filtered_albums.columns:
        popularity_col = "album_popularity"
    else:
        popularity_candidates = [
            c for c in filtered_albums.columns if "popularity" in c.lower()
        ]
        popularity_col = popularity_candidates[0] if popularity_candidates else None

    if popularity_col:
        yearly_popularity = (
            filtered_albums.groupby("year")[popularity_col]
            .mean()
            .reset_index(name="avg_popularity")
            .sort_values("year")
        )

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.table(yearly_popularity.tail(10).reset_index(drop=True))

        with right_col:
            fig, ax = plt.subplots(figsize=(5.8, 3.5))
            style_figure(fig, ax)
            ax.plot(
                yearly_popularity["year"],
                yearly_popularity["avg_popularity"],
                marker="o",
                color=SPOTIFY_GREEN,
            )
            style_axis(ax, "Average album popularity", "Year", "Popularity")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("No album popularity column was found for yearly popularity analysis.")

    create_section_divider()

    st.markdown("### Top years in the dataset")
    top_years = yearly_counts.sort_values("album_count", ascending=False).head(10)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.table(top_years.reset_index(drop=True))

    with right_col:
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        style_figure(fig, ax)
        ax.barh(
            top_years["year"].astype(str)[::-1],
            top_years["album_count"][::-1],
            color=SPOTIFY_GREEN,
        )
        style_axis(ax, "Top years by release count", "Album count", "")
        plt.tight_layout()
        st.pyplot(fig)


def create_album_explorer(
    albums,
    tracks,
    year_range,
    artist_search,
    album_search,
    selected_album,
):
    st.title("💿 Album Explorer")
    st.caption(
        "Use the control panel to filter by artist and album name, then select an album to explore tracks, popularity, and release details."
    )

    filtered_albums = filter_data_by_year(albums, year_range).copy()

    if filtered_albums.empty:
        st.warning("No album data is available for the selected year range.")
        return

    album_name_col, artist_col, popularity_col = get_album_filter_columns(
        filtered_albums
    )

    if not album_name_col:
        st.warning("No album name column was found.")
        return

    album_pool = filtered_albums.copy()

    if artist_col and artist_search:
        album_pool = album_pool[
            album_pool[artist_col].astype(str).str.lower() == artist_search.lower()
        ]

    if album_search:
        album_pool = album_pool[
            album_pool[album_name_col]
            .astype(str)
            .str.contains(album_search, case=False, na=False)
        ]

    if not selected_album:
        st.info(
            "Use the control panel to filter by artist or album name, then choose an album."
        )
        return

    album_df = album_pool[
        album_pool[album_name_col].astype(str).str.lower() == selected_album.lower()
    ].copy()

    if album_df.empty:
        st.warning("That album was not found in the filtered data.")
        return

    album_row = album_df.iloc[0]

    left, right = st.columns([2, 1])

    with left:
        st.subheader(selected_album)

        if artist_col and pd.notna(album_row[artist_col]):
            st.write(f"**Artist:** {album_row[artist_col]}")

        if "year" in album_df.columns and pd.notna(album_row["year"]):
            st.write(f"**Release year:** {int(album_row['year'])}")

        if "album_type" in album_df.columns and pd.notna(album_row["album_type"]):
            st.write(f"**Album type:** {album_row['album_type']}")

    with right:
        st.markdown("### Album summary")
        st.metric("Rows found", len(album_df))

        if popularity_col and pd.notna(album_row[popularity_col]):
            st.metric("Popularity", round(float(album_row[popularity_col]), 1))
        else:
            st.metric("Popularity", "N/A")

        if "track_id" in album_df.columns:
            st.metric("Tracks linked", album_df["track_id"].nunique())
        else:
            st.metric("Tracks linked", "N/A")

    create_section_divider()

    st.markdown("### Linked tracks")

    if "track_id" not in album_df.columns:
        st.info("No track links were found for this album.")
        return

    album_track_ids = album_df["track_id"].dropna().unique()
    album_tracks = tracks[tracks["id"].isin(album_track_ids)].copy()

    if album_tracks.empty:
        st.info("No matching track rows were found for this album.")
        return

    display_cols = []
    for col in ["track_name", "track_popularity", "explicit", "duration_ms"]:
        if col in album_tracks.columns:
            display_cols.append(col)

    if display_cols:
        st.dataframe(
            album_tracks[display_cols].drop_duplicates().reset_index(drop=True),
            width="stretch",
        )

    c1, c2, c3 = st.columns(3)

    if "track_popularity" in album_tracks.columns:
        c1.metric(
            "Avg track popularity",
            round(float(album_tracks["track_popularity"].mean()), 1),
        )
    else:
        c1.metric("Avg track popularity", "N/A")

    if "explicit" in album_tracks.columns:
        explicit_share = (
            album_tracks["explicit"].astype(str).str.lower() == "true"
        ).mean()
        c2.metric("Explicit track share", f"{explicit_share * 100:.1f}%")
    else:
        c2.metric("Explicit track share", "N/A")

    c3.metric(
        "Unique tracks",
        (
            album_tracks["id"].nunique()
            if "id" in album_tracks.columns
            else len(album_tracks)
        ),
    )

    create_section_divider()

    tab1, tab2 = st.tabs(["Track Popularity", "Explicitness"])

    with tab1:
        if (
            "track_name" in album_tracks.columns
            and "track_popularity" in album_tracks.columns
        ):
            top_tracks = (
                album_tracks[["track_name", "track_popularity"]]
                .dropna()
                .drop_duplicates()
                .sort_values(by="track_popularity", ascending=False)
            )

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.table(top_tracks.head(10).reset_index(drop=True))

            with right_col:
                fig, ax = plt.subplots(figsize=(5.4, 3.4))
                style_figure(fig, ax)
                ax.barh(
                    top_tracks["track_name"].head(10)[::-1],
                    top_tracks["track_popularity"].head(10)[::-1],
                    color=SPOTIFY_GREEN,
                )
                style_axis(
                    ax, f"Track popularity for {selected_album}", "Popularity", ""
                )
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Track popularity information is not available.")

    with tab2:
        if "explicit" in album_tracks.columns:
            explicit_counts = (
                album_tracks["explicit"]
                .astype(str)
                .str.lower()
                .value_counts()
                .reset_index()
            )
            explicit_counts.columns = ["explicit", "count"]

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.table(explicit_counts)

            with right_col:
                fig, ax = plt.subplots(figsize=(5.4, 3.4))
                style_figure(fig, ax)
                ax.bar(
                    explicit_counts["explicit"],
                    explicit_counts["count"],
                    color=SPOTIFY_GREEN,
                )
                style_axis(
                    ax, f"Explicitness for {selected_album}", "Explicit", "Track count"
                )
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Explicitness information is not available.")


# -----------------------------
# Main app
# -----------------------------
def main():
    artists, tracks, albums, features = load_and_clean_data()
    albums = prepare_album_year(albums)
    genre_df = build_genre_df(artists)

    st.sidebar.markdown(
        """
        <div class="sidebar-title">Spotify Menu</div>
        <div class="sidebar-subtitle">Navigate your analytics dashboard</div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<div class="sidebar-panel"><div class="sidebar-section-title">♪ Navigation</div></div>',
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Artist Explorer",
            "Genre Explorer",
            "Feature Explorer",
            "Album Explorer",
            "Time Trends",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        '<div class="sidebar-panel"><div class="sidebar-section-title">♫ Filters</div><div class="sidebar-note">Leave a field blank for no filter.</div></div>',
        unsafe_allow_html=True,
    )

    if "year" in albums.columns and albums["year"].notna().any():
        min_year = int(albums["year"].min())
        max_year = int(albums["year"].max())
        year_range = st.sidebar.slider(
            "Select year range",
            min_year,
            max_year,
            (min_year, max_year),
            key="year_range",
        )
    else:
        year_range = (2000, 2023)

    if {"name", "artist_popularity", "followers"}.issubset(artists.columns):
        top_artist_options = (
            artists[["name", "artist_popularity", "followers"]]
            .dropna(subset=["name"])
            .drop_duplicates(subset=["name"])
            .sort_values(
                by=["artist_popularity", "followers"],
                ascending=[False, False],
            )
            .head(200)["name"]
            .tolist()
        )
    else:
        top_artist_options = (
            sorted(artists["name"].dropna().unique().tolist())[:200]
            if "name" in artists.columns
            else []
        )

    artist_search = st.sidebar.selectbox(
        "Artist",
        options=[""] + top_artist_options,
        key="artist_search",
    )

    genre_options = genre_df["Genre"].dropna().tolist() if not genre_df.empty else []
    selected_genre = st.sidebar.selectbox(
        "Genre",
        options=[""] + genre_options,
        key="selected_genre",
    )

    feature_options = [
        "",
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "loudness",
    ]
    selected_feature = st.sidebar.selectbox(
        "Audio feature",
        options=feature_options,
        key="selected_feature",
    )

    album_search = ""
    selected_album = ""

    if page == "Album Explorer":
        st.sidebar.markdown(
            '<div class="sidebar-panel"><div class="sidebar-section-title">♩ Album Tools</div><div class="sidebar-note">Use the artist filter above to narrow the album list, then search and select an album here.</div></div>',
            unsafe_allow_html=True,
        )

        filtered_albums = filter_data_by_year(albums, year_range).copy()
        album_name_col, artist_col, _ = get_album_filter_columns(filtered_albums)

        album_pool = filtered_albums.copy()

        if artist_col and artist_search:
            album_pool = album_pool[
                album_pool[artist_col].astype(str).str.lower() == artist_search.lower()
            ]

        album_search = ""
        album_options = (
            album_pool[album_name_col]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
            if album_name_col
            else []
        )

        selected_album = st.sidebar.selectbox(
            "Album",
            options=[""] + album_options,
            key="album_select",
        )

    st.sidebar.markdown(
        '<div class="sidebar-panel"><div class="sidebar-section-title">♬ Current View</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.write(f"**Page:** {page}")
    st.sidebar.write(f"**Artist:** {artist_search if artist_search else 'None'}")
    st.sidebar.write(f"**Genre:** {selected_genre if selected_genre else 'None'}")
    st.sidebar.write(f"**Feature:** {selected_feature if selected_feature else 'None'}")
    if page == "Album Explorer":
        st.sidebar.write(
            f"**Album search:** {album_search if album_search else 'None'}"
        )
        st.sidebar.write(f"**Album:** {selected_album if selected_album else 'None'}")
    st.sidebar.write(f"**Years:** {year_range[0]}–{year_range[1]}")

    if page == "Overview":
        create_overview(artists, tracks, albums, features, genre_df, year_range)

    elif page == "Artist Explorer":
        create_artist_explorer(
            artists,
            tracks,
            albums,
            features,
            artist_search,
            year_range,
        )

    elif page == "Genre Explorer":
        create_genre_explorer(artists, selected_genre)

    elif page == "Feature Explorer":
        create_feature_explorer(features, tracks, selected_feature)

    elif page == "Album Explorer":
        create_album_explorer(
            albums,
            tracks,
            year_range,
            artist_search,
            album_search,
            selected_album,
        )

    elif page == "Time Trends":
        create_time_trends(albums, year_range)


if __name__ == "__main__":
    main()
