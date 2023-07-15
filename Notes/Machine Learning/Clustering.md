Grouping the data into a cluster with common similarities based on the cluster algorithm the clustered group has more commons and closeness more than any other clusters. 
the clustering accuracy came from the assumption of that we can accurately quantify the similarities and closeness between various data points. the distance measures between the points are done using one of the 3 methods. 
-   Euclidean distance measure
	    the simplest in terms of calculations, as it is the shortest distance between any two points in multidimensional space. 
	    ![[Pasted image 20230318160752.png]]
	    
-   Manhattan distance measure
	    the limitation of the above method in many cases will be the worst to use, where at many scenarios we need to measure the actual closeness and the distance between two points. for example the distance between two points in a map is different when using ground transportation than using a helicopter. 
	    ![[Pasted image 20230318161033.png]]
	    
-   Cosine distance measure
		at large space dimensions above methods are not efficient. The cosine distance measure is calculated by measuring the cosine angle created by two points connected to a reference point. If the data points are close, then the angle will be narrow, irrespective of the dimensions they have. On the other hand, if they are far away, then the angle will be large:
		![[Pasted image 20230318161310.png]]
		*Textual Data can be almost be considered a highly dimension space *. the origin point can be different it is a reference :>
		
### Hierarchical clustering
the previous cluster is mainly using the top down approach where we start with random- most probably small number of clusters then we move forward to find the best fit to the k clusters. while the in other hand this approach can be used from down to top where we start with # cluster = # points--- irrational when big or textual data is given. 
