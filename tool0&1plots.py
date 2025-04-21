import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

services_df = pd.read_csv("milwaukee_multiservice_yelp.csv")

#Plot 1: Top 10 Most Common Service Categories

category_counts = services_df['category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']

plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='category', data=category_counts.head(10), palette="viridis")
plt.title("Top 10 Most Common Service Categories in Milwaukee")
plt.xlabel("Number of Services")
plt.ylabel("Service Category")
plt.tight_layout()
plt.show()


# Plot 2: Distribution of Services per Block Group (GEOID)

geo_counts = services_df['GEOID'].value_counts().reset_index()
geo_counts.columns = ['GEOID', 'service_count']

plt.figure(figsize=(10, 6))
sns.histplot(geo_counts['service_count'], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Services Across Block Groups")
plt.xlabel("Number of Services per Block Group")
plt.ylabel("Number of Block Groups")
plt.tight_layout()
plt.show()

