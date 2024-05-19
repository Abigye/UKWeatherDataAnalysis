from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import pandas as pd 
import numpy as np

#Read the aligned data 
aligned_data = pd.read_csv('cleaned_aligned_data.csv')

# get the averages of the weatehr data 
cols = ['tmax(degC)', 'tmin(degC)', 'af(days)', 'rain(mm)', 'sun(hours)']
df_avg = aligned_data.groupby(['station'])[cols].mean()

# Standardize the data to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
data_std = scaler.fit_transform(df_avg)

# perform the clustering
Z = linkage(data_std, method='ward', metric="euclidean")

# Plot dendrogram
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(Z,labels=df_avg.index,ax=ax)

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Weather Stations')
plt.ylabel('Distance')
plt.show()

# Determine the number of clusters you want to obtain
k = 2

# Obtain the cluster labels for each data point
labels = fcluster(Z, k, criterion='maxclust')

# Calculate the cluster centers in the scaled space
cluster_centers_scaled = np.zeros((k, data_std.shape[1]))
for i in range(1, k+1):
    cluster_centers_scaled[i-1] = data_std[labels==i].mean(axis=0)

# Transform the cluster centers back to their original unscaled space
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

# Print the cluster centers in the original unscaled space
print(f"The cluster centers for {k} clusters are: \n {cluster_centers}")




