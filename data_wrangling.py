import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from itertools import combinations
from collections import Counter


def clean_features(features_df):
    features_df = features_df.dropna(subset=["id"])

    required_numeric = [
        "danceability", "energy", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "valence", "key", "mode", "loudness", "tempo",
        "duration_ms", "time_signature"
    ]
    features_df = features_df.dropna(subset=required_numeric)

    numeric_cols_0_1 = [
        "danceability", "energy", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence"
    ]

    for col in numeric_cols_0_1:
        features_df = features_df[(features_df[col] >= 0) & (features_df[col] <= 1)]

    features_df = features_df[features_df["key"].isin(range(0, 12))]
    features_df = features_df[features_df["mode"].isin([0, 1])]
    features_df = features_df[features_df["loudness"] <= 0]
    features_df = features_df[features_df["tempo"] >= 0]
    features_df = features_df[features_df["duration_ms"] > 0]
    features_df = features_df[features_df["time_signature"].isin([3, 4, 5])]
    features_df = features_df.drop_duplicates()

    return features_df.reset_index(drop=True)


def clean_tracks(tracks_df):
    tracks_df = tracks_df.dropna(subset=["id"])
    tracks_df = tracks_df[
        (tracks_df["track_popularity"] >= 0) &
        (tracks_df["track_popularity"] <= 100)
    ]
    tracks_df = tracks_df[tracks_df["explicit"].isin(["false", "true"])]
    tracks_df = tracks_df.drop_duplicates()

    return tracks_df.reset_index(drop=True)


def clean_albums(albums_df):
    albums_df = albums_df.dropna(subset=["track_id", "album_id"])
    albums_df = albums_df[albums_df["duration_ms"] > 0]
    albums_df = albums_df[albums_df["duration_sec"] > 0]
    albums_df = albums_df[
        (albums_df["album_popularity"] >= 0) &
        (albums_df["album_popularity"] <= 100)
    ]
    albums_df = albums_df[albums_df["total_tracks"] > 0]
    albums_df = albums_df.dropna(axis=1, how="all")
    albums_df = albums_df.drop_duplicates()

    return albums_df.reset_index(drop=True)


def clean_artists(artists_df):
    artists_df = artists_df.dropna(subset=["id", "name"])
    artists_df = artists_df[
        (artists_df["artist_popularity"] >= 0) &
        (artists_df["artist_popularity"] <= 100)
    ]
    artists_df = artists_df[artists_df["followers"] >= 0]
    artists_df = artists_df.dropna(axis=1, how="all")
    artists_df = artists_df.drop_duplicates()

    return artists_df.reset_index(drop=True)


def resolve_artist_duplicate_names(artists_df):
    artists_df = artists_df.copy()
    artists_df["name_clean"] = artists_df["name"].str.lower().str.strip()

    genre_cols = [col for col in artists_df.columns if col.startswith("genre_")]

    consolidated_rows = []

    for _, group in artists_df.groupby("name_clean"):
        row = {}

        id_values = group["id"].dropna()
        name_values = group["name"].dropna()

        row["id"] = id_values.iloc[0] if not id_values.empty else None
        row["name"] = name_values.iloc[0] if not name_values.empty else None
        row["artist_popularity"] = group["artist_popularity"].max()
        row["followers"] = group["followers"].max()
        row["name_clean"] = group["name_clean"].iloc[0]

        genres = []
        for col in genre_cols:
            genres.extend(group[col].dropna().tolist())

        unique_genres = []
        for genre in genres:
            if genre not in unique_genres:
                unique_genres.append(genre)

        for i, col in enumerate(genre_cols):
            row[col] = unique_genres[i] if i < len(unique_genres) else None

        consolidated_rows.append(row)

    return pd.DataFrame(consolidated_rows).reset_index(drop=True)


def album_feature_summary(album_name, albums_df, features_df):
    albums_features = albums_df.merge(
        features_df,
        left_on="track_id",
        right_on="id",
        how="inner"
    )

    album_tracks = albums_features[
        albums_features["album_name"].str.lower() == album_name.lower()
    ]

    if album_tracks.empty:
        print(f"Album '{album_name}' not found.")
        return None

    feature_columns = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "loudness"
    ]

    summary = album_tracks[feature_columns].mean()
    print(summary)

    return summary


def add_feature_ranks(features_df, rank_features, labels):
    features_ranked = features_df.copy()

    for feature in rank_features:
        if feature in features_ranked.columns:
            features_ranked[f"{feature}_rank"] = pd.qcut(
                features_ranked[feature],
                5,
                labels=labels
            )

    return features_ranked


def main():
    album_name = "Black Sand"
    top_n = 10
    rank_features = ["danceability", "energy"]

    database_connect = sqlite3.connect("data/spotify_database.db")

    artist = pd.read_sql_query("SELECT * FROM artist_data;", database_connect)
    features = pd.read_sql_query("SELECT * FROM features_data;", database_connect)
    tracks = pd.read_sql_query("SELECT * FROM tracks_data;", database_connect)
    albums = pd.read_sql_query("SELECT * FROM albums_data;", database_connect)

    features_clean = clean_features(features)
    tracks_clean = clean_tracks(tracks)
    albums_clean = clean_albums(albums)
    artists_clean = clean_artists(artist)
    artists_clean = resolve_artist_duplicate_names(artists_clean)

    genre_cols = ["genre_1", "genre_2", "genre_3", "genre_4", "genre_5", "genre_6"]
    genre_cols = [col for col in genre_cols if col in artists_clean.columns]

    pair_counts = Counter()

    for row in artists_clean[genre_cols].values:
        genres = [g for g in row if pd.notna(g) and g != ""]
        pair_counts.update(combinations(sorted(genres), 2))

    print(pair_counts.most_common(10))

    numeric_cols = tracks_clean.select_dtypes(include=[np.number]).columns
    df_numeric = tracks_clean[numeric_cols]

    z_scores = np.abs(zscore(df_numeric, nan_policy="omit"))
    outliers_z = z_scores > 3

    print(pd.DataFrame(outliers_z, columns=numeric_cols).sum())

    query = """
    SELECT 
        a.track_name,
        a.release_date,
        t.track_popularity
    FROM albums_data a
    JOIN tracks_data t
        ON a.track_id = t.id
    """

    monthly_df = pd.read_sql_query(query, database_connect)

    monthly_df["track_popularity"] = pd.to_numeric(monthly_df["track_popularity"], errors="coerce")
    monthly_df["release_date"] = pd.to_datetime(monthly_df["release_date"], errors="coerce")
    monthly_df = monthly_df.dropna(subset=["track_name", "track_popularity", "release_date"])

    monthly_df["year_month"] = monthly_df["release_date"].dt.to_period("M")

    top_songs = (
        monthly_df.groupby("track_name")["track_popularity"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    top_song_names = top_songs.index.tolist()

    df_top = monthly_df[monthly_df["track_name"].isin(top_song_names)]

    monthly_agg = (
        df_top.groupby(["year_month", "track_name"])
        .agg({"track_popularity": "mean"})
        .reset_index()
    )

    monthly_agg["year_month"] = monthly_agg["year_month"].astype(str)
    monthly_agg = monthly_agg.sort_values("year_month")

    plt.figure()

    for song in top_song_names:
        song_data = monthly_agg[monthly_agg["track_name"] == song]
        plt.plot(song_data["year_month"], song_data["track_popularity"], label=song)

    plt.title("Monthly Popularity Trends for Top Songs")
    plt.xlabel("Month")
    plt.ylabel("Average Popularity")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    query = """
    SELECT 
        a.release_date,
        AVG(f.loudness) AS avg_loudness,
        AVG(f.acousticness) AS avg_acousticness,
        AVG(f.tempo) AS avg_tempo
    FROM albums_data a
    JOIN features_data f
        ON a.track_id = f.id
    GROUP BY a.release_date
    ORDER BY a.release_date ASC;
    """

    time_df = pd.read_sql_query(query, database_connect)
    time_df["release_date"] = pd.to_datetime(time_df["release_date"], errors="coerce")

    query = """
    SELECT 
        (CAST(CAST(SUBSTR(a.release_date, 1, 4) AS INTEGER) / 10 AS INTEGER) * 10) || 's' AS era,
        AVG(f.loudness) AS avg_loudness,
        AVG(f.acousticness) AS avg_acousticness,
        AVG(f.tempo) AS avg_tempo
    FROM albums_data a
    JOIN features_data f
        ON a.track_id = f.id
    GROUP BY era
    ORDER BY era ASC;
    """

    era_df = pd.read_sql_query(query, database_connect)

    plt.figure(figsize=(10, 6))
    plt.bar(era_df["era"], era_df["avg_loudness"])
    plt.title("Average Loudness per Era")
    plt.xlabel("Era")
    plt.ylabel("Loudness (dB)")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(era_df["era"], era_df["avg_acousticness"])
    plt.title("Average Acousticness per Era")
    plt.xlabel("Era")
    plt.ylabel("Acousticness")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(era_df["era"], era_df["avg_tempo"])
    plt.title("Average Tempo per Era")
    plt.xlabel("Era")
    plt.ylabel("Tempo (BPM)")
    plt.ylim(90, 130)
    plt.show()

    labels = ["very low", "low", "medium", "high", "very high"]
    features_ranked = add_feature_ranks(features_clean, rank_features, labels)

    rank_columns = [feature for feature in rank_features if feature in features_ranked.columns]
    display_columns = []

    for feature in rank_columns:
        display_columns.extend([feature, f"{feature}_rank"])

    print(features_ranked[display_columns].head())

    album_feature_summary(album_name, albums_clean, features_clean)

    database_connect.close()


if __name__ == "__main__":
    main()