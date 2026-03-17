import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm


def load_data():
    return pd.read_csv("data/artist_data.csv")


def basic_inspection(df):
    print("Number of columns:", len(df.columns))
    print("\nColumn names:\n", df.columns)
    print("\nData types:\n", df.dtypes)

    unique_artists = df["name"].nunique()
    print("\nNumber of unique artists:", unique_artists)


def top_10_artists(df):
    top_10 = (
        df.groupby("name")["artist_popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTop 10 Artists:\n", top_10)

    plt.figure()
    plt.bar(top_10.index, top_10.values)
    plt.title('Top 10 Artists by Popularity')
    plt.xlabel('Artist')
    plt.ylabel('Popularity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def top_ten_per_genre(df, genre):
    df_filtered = df[df['artist_genres'].str.contains(genre, case=False, na=False)]
    df_sorted = df_filtered.sort_values(by='artist_popularity', ascending=False)
    return df_sorted.head(10)


def genre_analysis(df):
    df = df.copy()

    df['sum_of_genres'] = df[['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5']].notna().sum(axis=1)

    sb.boxplot(data=df, x='sum_of_genres', y='artist_popularity')
    plt.title("Popularity vs Number of Genres")
    plt.show()

    sb.barplot(data=df, x='sum_of_genres', y='artist_popularity', errorbar=None)
    plt.title("Average Popularity by Number of Genres")
    plt.show()

    sb.barplot(data=df, x='sum_of_genres', y='followers', errorbar=None)
    plt.title("Average Followers by Number of Genres")
    plt.show()

    corr = df[['sum_of_genres', 'artist_popularity', 'followers']].corr()
    print("\nCorrelation matrix:\n", corr)


def regression_analysis(df):
    df = df[['artist_popularity', 'followers']].dropna()
    df = df[df['followers'] > 0]

    correlation = df['artist_popularity'].corr(df['followers'])
    print("\nCorrelation (popularity vs followers):", correlation)

    df['log_followers'] = np.log(df['followers'])

    Y = df['artist_popularity']
    X = sm.add_constant(df['log_followers'])

    model = sm.OLS(Y, X).fit()
    print("\nRegression results:\n", model.summary())

    df['predicted'] = model.predict(X)
    df['residual'] = df['artist_popularity'] - df['predicted']

    over_performers = df.sort_values(by='residual', ascending=False).head(10)
    legacy_artists = df.sort_values(by='residual', ascending=True).head(10)

    print("\nTop 10 Over-performers:\n", over_performers)
    print("\nTop 10 Legacy Artists:\n", legacy_artists)


def main():
    df = load_data()

    basic_inspection(df)
    top_10_artists(df)

    genre = "reggaeton"
    print(f"\nTop 10 {genre} Artists:\n", top_ten_per_genre(df, genre))

    genre_analysis(df)
    regression_analysis(df)


if __name__ == "__main__":
    main()