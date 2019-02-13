from mnist import test_images
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


train_data = test_images().reshape([10000, 28*28])
kmeans = KMeans(n_clusters=10).fit(train_data)
clusters = kmeans.cluster_centers_

N, M = 2, 5
fig, axes = plt.subplots(N, M)
k = 0
for i in range(N):
    for j in range(M):
        axes[i, j].imshow(clusters[k].reshape([28, 28]))
        k += 1
plt.show()
