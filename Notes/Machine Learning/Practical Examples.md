we have seen so far how we can use the k-mean to cluster the data. 
now let's take a real life example. 
### Clustering Similar tweets together 
we first takes the set of data and process the data into two phases 
1. **Topic Modeling**: discover the topics of the given tweets. 
	1. tweets tokens 
	2. process the data in the tweet... validate the data inside the tweet
	3. create a TDM that represents the top 200 frequent words 
	4. now we have for example 10 various topics 
2. **Cluster**: we just cluster the data to k-clusters = # of topics. 
	1. after we get the number of topics we then tries to  cluster all the data given the k.