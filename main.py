import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_customers = 500
data = {
    'Recency': np.random.randint(1, 365, num_customers),  # Days since last purchase
    'Frequency': np.random.randint(1, 13, num_customers),  # Number of purchases in the last year
    'MonetaryValue': np.random.randint(10, 1000, num_customers),  # Total spending in the last year
    'WebsiteVisits': np.random.randint(1, 20, num_customers), # Number of website visits
    'AvgOrderValue': np.random.randint(10, 500, num_customers) #Average value of orders
}
df = pd.DataFrame(data)
# --- 2. Data Preprocessing ---
# Scale the features for KMeans clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_features, columns=df.columns)
# --- 3. Customer Segmentation using KMeans ---
# Determine optimal number of clusters (e.g., using the Elbow method -  simplified here)
kmeans = KMeans(n_clusters=3, random_state=42) # Choosing 3 clusters as an example.  A more robust method would be needed in a real-world scenario.
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_
# --- 4. Analysis and Visualization ---
# Analyze cluster characteristics
cluster_means = df.groupby('Cluster').mean()
print("Cluster Characteristics:")
print(cluster_means)
# Visualize clusters (example using Recency and MonetaryValue)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Recency', y='MonetaryValue', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segmentation based on Recency and Monetary Value')
plt.xlabel('Recency (Days)')
plt.ylabel('Monetary Value')
plt.savefig('customer_segmentation.png')
print("Plot saved to customer_segmentation.png")
#Further analysis could involve exploring other features and relationships, and potentially using more sophisticated clustering techniques.  This example provides a basic framework.
# --- 5. Churn Prediction (Illustrative - Requires additional data and model) ---
# In a real-world scenario, you'd add a 'Churn' column to your data and train a classification model (e.g., Logistic Regression, Random Forest) to predict churn probability for each customer segment.  This step is omitted here due to the synthetic data limitations.