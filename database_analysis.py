import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore


def connect_db():
    return sqlite3.connect("data/spotify_database.db")

def outliers(conn):

    cursor = conn.cursor()

    query = "SELECT * FROM tracks_data"
    df = pd.read_sql_query(query, conn)

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    # Compute Z-scores
    z_scores = np.abs(zscore(df_numeric, nan_policy='omit'))

    # Threshold (commonly 3)
    threshold = 3

    # Boolean mask for outliers
    outliers_z = (z_scores > threshold)

    # Count outliers per column
    print("Outliers per column (Z-score):")
    print(pd.DataFrame(outliers_z, columns=numeric_cols).sum())

def analyze_album(conn, album_name):

    cursor = conn.cursor()

    print(f"\nAnalyzing album: {album_name}")

    cursor.execute("""
        SELECT COUNT(track_id)
        FROM albums_data
        WHERE album_name = ?
    """, (album_name,))

    track_count = cursor.fetchone()[0]

    print(f"Number of tracks in album: {track_count}")

    if track_count == 0:
        print("No data found for this album.")
        return

    cursor.execute("""
        SELECT 
            albums_data.track_name,
            features_data.danceability,
            features_data.loudness
        FROM albums_data
        JOIN features_data
            ON albums_data.track_id = features_data.id
        WHERE albums_data.album_name = ?
    """, (album_name,))

    tracks = cursor.fetchall()

    df = pd.DataFrame(
        tracks,
        columns=["track_name", "danceability", "loudness"]
    )

    print("\nTracks + Features:")
    print(df)

    print("\nFeature Statistics:")
    print(df[["danceability", "loudness"]].describe())


def analyze_feature(conn, feature_name):

    cursor = conn.cursor()

    print(f"\nAnalyzing top 10% tracks based on {feature_name}")

    cursor.execute(f"""
        SELECT 
            albums_data.track_name,
            albums_data.artist_0,
            albums_data.artist_1,
            albums_data.artist_2,
            albums_data.artist_3,
            albums_data.artist_4,
            features_data.{feature_name}
        FROM albums_data
        JOIN features_data
            ON albums_data.track_id = features_data.id
    """)

    rows = cursor.fetchall()

    if not rows:
        print("No data found.")
        return

    columns = [
        "track_name","artist_0","artist_1","artist_2",
        "artist_3","artist_4", feature_name
    ]

    df = pd.DataFrame(rows, columns=columns)

    df = df.dropna(subset=[feature_name])

    threshold = df[feature_name].quantile(0.90)

    df_top = df[df[feature_name] >= threshold]

    artist_cols = ["artist_0","artist_1","artist_2","artist_3","artist_4"]

    df_new = df_top.melt(
        id_vars=["track_name", feature_name],
        value_vars=artist_cols,
        value_name="artist"
    )

    df_new = df_new.dropna(subset=["artist"])
    df_new = df_new[df_new["artist"].astype(str).str.strip() != ""]

    print(f"\nArtists appearing most in top 10% tracks based on {feature_name}:")
    print(df_new["artist"].value_counts().head(10))

    print(f"\nAverage {feature_name} score per artist:")
    print(
        df_new.groupby("artist")[feature_name]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )


def plot_feature_distribution(conn, album_name, feature_name):

    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT 
            features_data.{feature_name}
        FROM albums_data
        JOIN features_data
            ON albums_data.track_id = features_data.id
        WHERE albums_data.album_name = ?
    """, (album_name,))

    values = [row[0] for row in cursor.fetchall()]

    if len(values) == 0:
        print("No data found for plotting.")
        return

    plt.figure(figsize=(8,5))

    hist_values, _, _ = plt.hist(
        values,
        bins=10,
        rwidth=0.85,
        edgecolor="black"
    )

    if len(hist_values) > 0:
        y_max = max(hist_values) + 1
        plt.ylim(0, y_max)
        plt.yticks(np.arange(0, y_max + 1, 1))

    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"{feature_name} distribution for {album_name}")

    plt.show()


def explicit_artist_analysis(conn):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            a.artist_0,
            COUNT(*) AS total_tracks,
            SUM(CASE WHEN t.explicit = 'true' THEN 1 ELSE 0 END) AS explicit_tracks,
            ROUND(
                1.0 * SUM(CASE WHEN t.explicit = 'true' THEN 1 ELSE 0 END)
                / COUNT(*),
                3
            ) AS explicit_proportion
        FROM albums_data a
        JOIN tracks_data t
            ON a.track_id = t.id
        GROUP BY a.artist_0
        ORDER BY explicit_proportion DESC, total_tracks DESC
        LIMIT 10;
    """)

    rows = cursor.fetchall()

    column_names = [column[0] for column in cursor.description]

    print("\nExplicit proportion per artist:")
    print(column_names)

    for row in rows:
        print(row)


def analyze_eras(conn):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *,
               (CAST(CAST(SUBSTR(release_date, 1, 4) AS INTEGER) / 10 AS INTEGER) * 10) || "s" AS decade
        FROM albums_data
        ORDER BY decade
        LIMIT 20;
    """)

    column_names = [column[0] for column in cursor.description]

    print("\nEra extraction preview:")
    print(column_names)

def monthly_popularity(conn):

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

    # Data Cleaning
    df['track_popularity'] = pd.to_numeric(df['track_popularity'], errors='coerce')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    df = df.dropna(subset=['track_name', 'track_popularity', 'release_date'])

    # Create Month Column
    df['year_month'] = df['release_date'].dt.to_period('M')

    #  Find Top Songs
    top_songs = (
        df.groupby('track_name')['track_popularity']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    top_song_names = top_songs.index.tolist()

    print("Top 10 Songs by Total Popularity:")
    print(top_songs)

    # Filter Top Songs
    df_top = df[df['track_name'].isin(top_song_names)]

    # Aggregate Monthly Data
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

    # Plot Popularity Over Time
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


def explicit_vs_nonexplicit(conn):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT CAST(AVG(track_popularity) AS INTEGER)
        FROM tracks_data
        WHERE explicit = "false";
    """)

    non_explicit = cursor.fetchone()[0]

    print("The average popularity for non-explicit tracks is:", non_explicit)

    cursor.execute("""
        SELECT CAST(AVG(track_popularity) AS INTEGER)
        FROM tracks_data
        WHERE explicit = "true";
    """)

    explicit = cursor.fetchone()[0]

    print("The average popularity for explicit tracks is:", explicit)

def main():

    conn = connect_db()

    album = "Black Sand"
    feature = "loudness"

    outliers(conn)

    print("-" * 60)

    analyze_album(conn, album)

    print("-" * 60)

    analyze_feature(conn, feature)

    print("-" * 60)

    plot_feature_distribution(conn, album, feature)

    print("-" * 60)

    explicit_artist_analysis(conn)

    print("-" * 60)

    analyze_eras(conn)

    print("-" * 60)

    monthly_popularity(conn)

    print("-" * 60)

    explicit_vs_nonexplicit(conn)

    conn.close()


if __name__ == "__main__":
    main()