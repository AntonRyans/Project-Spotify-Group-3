import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import zscore

# Connect to database
conn = sqlite3.connect('spotify_database.db')

# Load data (adjust table name if needed)
query = "SELECT * FROM tracks_data"
df = pd.read_sql_query(query, conn)

# Close connection
conn.close()

# Display basic info
print(df.info())
print(df.describe())

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