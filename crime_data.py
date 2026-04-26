# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("RUNNING THIS FILE")

# Step 1: Load Dataset
df = pd.read_csv(r"C:\Users\pavit\ML MINI PROJECT\crime_data.csv")
print("Dataset Preview:")
print(df.head())

# Step 2: Data Preprocessing
print("\nMissing Values:")
print(df.isnull().sum())
df = df.dropna()

# Step 3: Feature Engineering
severity_map = {
    'Theft': 1,
    'Burglary': 2,
    'Vandalism': 2,
    'Robbery': 3,
    'Assault': 3,
    'Homicide': 4
}
df['Crime_Severity'] = df['crime_type'].map(severity_map).fillna(1)

# Step 4: Data Analysis
print("\nCrime Count by Type:")
crime_count = df.groupby('crime_type')['crime_id'].count()
print(crime_count)

print("\nCrime Count by District:")
district_crime = df.groupby('district')['crime_id'].count()
print(district_crime)

print("\nDescriptive Statistics:")
print(df[["crime_count", "hour", "Crime_Severity"]].describe())

# Step 5: Data Visualization

# Crime Type Distribution
plt.figure(figsize=(10, 5))
crime_count.plot(kind='bar', color='crimson')
plt.title('Crime Type Distribution')
plt.xlabel('Crime Type')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Crime by Hour
hourly_crime = df.groupby('hour')['crime_id'].count()
plt.figure(figsize=(10, 5))
plt.plot(hourly_crime.index, hourly_crime.values, color='blue', marker='o')
plt.title('Crime Count by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Incidents')
plt.grid(True)
plt.show()

# Hotspot Map
plt.figure(figsize=(10, 6))
plt.scatter(
    df['longitude'],
    df['latitude'],
    c=df['Crime_Severity'],
    cmap='YlOrRd',
    alpha=0.5,
    s=10
)
plt.colorbar(label='Crime Severity')
plt.title('Crime Hotspot Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Correlation Heatmap
corr = df[['hour', 'month', 'Crime_Severity']].corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Correlation Heatmap')
plt.show()

print("\nCrime Pattern Analysis Completed Successfully!")