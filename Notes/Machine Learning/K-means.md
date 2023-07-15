the k refers to the number of clusters we are trying to perform. 
- [ ] use the means to find the closeness between the data points.
- [ ] Simple, Scaleable, and speed. 
the logic behind the k-means is to moving the centers of the clusters till they reflect the data points of the grouping they belong to. 
the big issue with the algorithm is the k is not given for the data set. as the k dependent on the number of natural groups in a particular dataset. the mechanism used to calculate the k is external and can be determined based on the given data, while we keep it this way because in larger dataset it is the most efficient. at some cases K will be intuitive, and the other may not then we tend to use the trial and error procedure or heuristics-based algorithm.

1. Initialization
	In order to group them, the k-means algorithm uses a distance measure to find the similarity or closeness between data points. Before using the k-means algorithm, the most appropriate distance measure needs to be selected. By default, the Euclidean distance measure will be used. Also, if the dataset has outliers, then a mechanism needs to be devised to determine the criteria that are to be identified and remove the outliers of the dataset.
The steps involved in the k-means clustering algorithm are as follows:

Step 1
We choose the number of clusters, _k_.

Step 2
Among the data points, we randomly choose _k_ points as cluster centers.

Step 3
Based on the selected distance measure, we iteratively compute the distance from each point in the problem space to each of the _k_ cluster centers. Based on the size of the dataset, this may be a time-consuming step—for example, if there are 10,000 points in the cluster and _k_ = 3, this means that 30,000 distances need to be calculated.

Step 4
We assign each data point in the problem space to the nearest cluster center.

Step 5
Now each data point in our problem space has an assigned cluster center. But we are not done, as the selection of the initial cluster centers was based on random selection. We need to verify that the current randomly selected cluster centers are actually the center  of each cluster. We recalculate the cluster centers by computing the mean of the constituent data points of each of the _k_ clusters. This step explains why this algorithm is called k-means.

Step 6
If the cluster centers have shifted in step 5, this means that we need to recompute the cluster assignment for each data point. For this, we will go back to step 3 to repeat that compute-intensive step. If the cluster centers have not shifted or if our predetermined stop condition (for example, the number of maximum iterations) has been satisfied, then we are done.


![[Pasted image 20230318170615.png]]

the accuracy is quite large when we reach the level where no shifting to the mean has done but this might be a time consuming resulting
### **Stop Condition**
1. the time has exceed the max time
2. the number of iterations. 



### Limitation of k-means clustering

The k-means algorithm is designed to be a simple and fast algorithm. Because of the intentional simplicity in its design, it comes with the following limitations:

-   The biggest limitation of k-means clustering is that the initial number of clusters has to be predetermined.
    
-   The initial assignment of cluster centers is random. This means that each time the algorithm is run, it may give slightly different clusters.
    
-   Each data point is assigned to only one cluster.
    
-   k-means clustering is sensitive to outliers.