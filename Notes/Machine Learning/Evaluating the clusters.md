the main objective of clustering is to group the data points and should be differentiable. 
1. the data points that belong to the same cluster should be as similar as possible 
2. the data points that belong to separate clusters should be as different as possible. 
Human intuition can be used to evaluate the clusters by visualizing the result. but this maybe not the best case all the time, the math used in such scenario called *Silhouette analysis*, you may watch a video about it but it mainly gives the resulted clusters a range from [0,1] the higher the number the better the cluster. 
We can use this technique to guess the best K as a hyper-parameter 
1. K value the  # of cluster 
2. the distance measurement method 
3. seeds initial values of the centroids. 
