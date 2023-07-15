the simplest definition is the process of finding a structure to the unstructured data, this happens in case only the data has some features in common or any kind of relations. 
![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781789801217/files/assets/32f60651-b8ff-4ec0-96fa-7b7145366467.png)

the first of all is to understand the data mining process and how it works before we go deeply. 
1. **CRISP-DM**( Cross-Industry  standard Process for Data Mining )
2. **SEMMA**( Sample, Explore, Modify, Model, Access) data-mining process.  
![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781789801217/files/assets/ff5d5601-03c9-465c-92dd-b830b77cd130.png)

- ##### **Phase 1**: **Business Understanding**
  gather the requirements, e.g binary classification problem,  transfer the requirements  to hypothesis that can be proved or rejected. and for other problems it is important to document the min accuracy of the Model. 
	
- ##### **Phase 2**: **Data Understanding**
  1. find the data set for the given problem 
  2. the available dataset is presented.
  3. then measure the quality of the data given. 
  4. the pattern that can be extracted from the given data. 
  5. the right feature that can be used as a label. 
	as we mentioned above phase 1 can be very critical as it can helps
	-   To discover patterns in the dataset

	-   To understand the structure of the dataset by analyzing the discovered pattern.
  
	-   To identify or derive the target variable
	
- ##### **Phase 3**: **Data Preparation**
  the data provided can be classified into 2 portions
  1. Training Data- that can be used to train the model.
  2. Testing Data- that can be used to test the model. 
  the unsupervised machine learning can be used to prepare the data, as they convert the unstructured data to a structured ones adding an additional dimension that can be very helpful in training the model. 
	
- ##### **Phase 4**: **Modeling**
  here we use the supervised model to formulate the patterns that we have discovered.  and prepare the data according to the supervised training model. the features we discovered earlier will be used as labels. this phase is to create a mathematical formula to represent the relations in the patterns of the interest. 
	
- ##### **Phase 5**: **Evaluation**
	this state is where to use the testing data and if the accuracy is what we expected then iterate again over the model. 
	
- ##### **Phase 6**: **Deployment**
  the final stage where the model is published according to the results of the 5th stage. 
  
  