import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv(r"artist_data.csv")


df = df[['artist_popularity', 'followers']].dropna()

df = df[df['followers'] > 0]


correlation = df['artist_popularity'].corr(df['followers'])


df['log_followers'] = np.log(df['followers'])


Y = df['artist_popularity']
X = sm.add_constant(df['log_followers'])  # adds intercept


model = sm.OLS(Y, X).fit()

print(correlation)
print(model.summary())


df['predicted_popularity'] = model.predict(X)
df['residual'] = df['artist_popularity'] - df['predicted_popularity']

over_performers = df.sort_values(by='residual', ascending=False).head(10)

legacy_artists = df.sort_values(by='residual', ascending=True).head(10)

print("--- Top 10 Over-performers (High Popularity, Low Followers) ---")
print(over_performers)

print("\n--- Top 10 Legacy Artists (Low Popularity, High Followers) ---")
print(legacy_artists)
