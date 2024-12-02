import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")

# Define the file path for the dataset
file_path = os.path.join('data', 'bike data.csv')  # Relative path for GitHub

# Load the dataset
data = pd.read_csv(file_path)

# Handle infinite values
data.replace([float('inf'), float('-inf')], float('nan'), inplace=True)

# Optional: Drop rows with missing values
data.dropna(inplace=True)

# Set a style for the plots
sns.set_theme(style="whitegrid")

# 1. Customer Demographics: Distribution by Gender
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='Customer Gender', palette='pastel')
plt.title('Customer Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('script/customer_gender_distribution.png')  # Save the plot in the script/ folder
plt.show()

# 2. Age Group Distribution (Pie Chart)
age_group_counts = data['Age Group'].value_counts()
plt.figure(figsize=(8, 8))
age_group_counts.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=90)
plt.title('Age Group Distribution')
plt.ylabel('')
plt.savefig('script/age_group_distribution.png')  # Save the plot in the script/ folder
plt.show()

# 3. Top 10 Most Purchased Products
top_products = data['Product'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('Top 10 Most Purchased Products')
plt.xlabel('Number of Orders')
plt.ylabel('Product')
plt.savefig('script/top_10_products.png')  # Save the plot in the script/ folder
plt.show()

# 4. Revenue Trend by Year
revenue_by_year = data.groupby('year')['Revenue'].sum()
plt.figure(figsize=(10, 6))
sns.lineplot(x=revenue_by_year.index, y=revenue_by_year.values, marker='o', color='b')
plt.title('Revenue Trend by Year')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.savefig('script/revenue_trend.png')  # Save the plot in the script/ folder
plt.show()

# 5. Profit vs Unit Price (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Unit Price', y='Profit', alpha=0.6, color='purple')
plt.title('Profit vs Unit Price')
plt.xlabel('Unit Price')
plt.ylabel('Profit')
plt.savefig('script/profit_vs_unit_price.png')  # Save the plot in the script/ folder
plt.show()

# 6. Heatmap: Correlation of Numerical Features
plt.figure(figsize=(12, 8))
corr_matrix = data[['Order Quantity', 'Unit Cost', 'Unit Price', 'Cost', 'Revenue', 'Profit']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap')
plt.savefig('script/correlation_heatmap.png')  # Save the plot in the script/ folder
plt.show()
