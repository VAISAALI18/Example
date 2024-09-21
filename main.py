import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from rake_nltk import Rake

# Step 1: Load Data
df = pd.read_csv('Train.csv')

# Data Overview
print("-----Data Overview-----")
print(df.head())  
print(df.info())  
print(df.describe()) 

# Pivot Table for Maximum, Minimum, and Mean ENTITY_LENGTH by CATEGORY_ID
if 'CATEGORY_ID' in df.columns:
    pivot_table = df.groupby('CATEGORY_ID').agg({
        'ENTITY_LENGTH': ['max', 'min', 'mean']
    }).reset_index()

    # Rename columns for clarity
    pivot_table.columns = ['Category_ID', 'Max_Entity_Length', 'Min_Entity_Length', 'Mean_Entity_Length']

    print("-----Pivot Table-----")
    print(pivot_table)
else:
    print("CATEGORY_ID column not found in the DataFrame.")

# Step 5: Univariate Analysis
# Distribution of ENTITY_LENGTH
plt.figure(figsize=(10, 6))
sns.histplot(df['ENTITY_LENGTH'], kde=True, bins=50)
plt.title('Distribution of ENTITY_LENGTH')
plt.show()

# Boxplot to detect outliers in ENTITY_LENGTH
plt.figure(figsize=(10, 6))
sns.boxplot(x='ENTITY_LENGTH', data=df)
plt.title('Boxplot of ENTITY_LENGTH')
plt.show()

# Count plot for CATEGORY_ID (Categorical feature)
plt.figure(figsize=(10, 6))
sns.countplot(x='CATEGORY_ID', data=df, order=df['CATEGORY_ID'].value_counts().index[:10])
plt.title('Top 10 Categories by Frequency')
plt.show()


# Step 2: Text Cleaning Function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Step 3: Apply Cleaning Function to ENTITY_DESCRIPTION
if 'ENTITY_DESCRIPTION' in df.columns:
    df['CLEANED_DESCRIPTION'] = df['ENTITY_DESCRIPTION'].apply(clean_text)

# Step 4: Keyword Extraction using RAKE
def generate_insight(description):
    rake = Rake()  # Initialize RAKE
    rake.extract_keywords_from_text(description)  # Extract keywords
    keyword_scores = rake.get_ranked_phrases_with_scores()  # Get keyword scores
    insights = {phrase: score for score, phrase in keyword_scores}
    
    # Create a summary string
    insight_summary = ', '.join([f"{phrase}: {score}" for phrase, score in insights.items()])
    return insight_summary if insight_summary else 'No significant keywords'

# Apply insight generation function
if 'CLEANED_DESCRIPTION' in df.columns:
    df['INSIGHT'] = df['CLEANED_DESCRIPTION'].apply(generate_insight)


# Step 6: Data Cleaning
print("Missing values before handling:\n", df.isnull().sum())
df['ENTITY_LENGTH'].fillna(df['ENTITY_LENGTH'].mean(), inplace=True)  
df = df.drop_duplicates()  

# Outlier treatment using IQR method
Q1 = df['ENTITY_LENGTH'].quantile(0.25)
Q3 = df['ENTITY_LENGTH'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['ENTITY_LENGTH'] < (Q1 - 1.5 * IQR)) | (df['ENTITY_LENGTH'] > (Q3 + 1.5 * IQR)))]

# Boxplot to detect outliers in ENTITY_LENGTH after cleaning
plt.figure(figsize=(10, 6))
sns.boxplot(x='ENTITY_LENGTH', data=df)
plt.title('Boxplot of ENTITY_LENGTH After Cleaning')
plt.show()

# Step 7: Feature Engineering
df['ENTITY_LENGTH_BINS'] = pd.cut(df['ENTITY_LENGTH'], bins=[0, 50, 100, 150, 200], labels=['Very Small', 'Small', 'Medium', 'Large'])
df = pd.get_dummies(df, columns=['CATEGORY_ID'], drop_first=True)  # One-hot encoding


print("-----Data Cleaning and Feature Engineering Completed-----")
