-   The ability to measure the frequency of a pattern
    
-   The ability to establish _cause_-and-_effect_ relationship among the patterns.
    
-   The ability to quantify the usefulness of patterns by comparing their accuracy to random guessing

can be used to determine the selected vars and their relations to the dataset. this could be elaborated more. 
-   What combinations of medicine may lead to complications for patients?

*recommendation  engines* are used to recommend, the data when collected over the time is called **transitional data**. when the ARA is applied to this data streaming it is called as market basket analysis. 
which helps to answer the following questions:

-   What is the optimal placement of items on the shelf?
    
-   How should the items appear in the marketing catalog?
    
-   What should be recommended, based on a user's buying patterns?

the one big sharp edge of the market analysis its self-explanatory and can be understood by the Business. we can express this in a very simple example where we have set of items of the Grocery **PI**={item1, item2, ...}
then @ each time a customer buy set of items these are stored in the **itemset** 
|index | items|
|-- | -------------------------- |
|t1 | wickets, pads|
|t2 | Bat, wickets, pads, helmet|
|t3 | helmet, ball |
##### **PI**= {bat , wickets, pads, helmet, ball} then if we take one of the purchases - t1
here the association rule is used in various transactions to evaluate the relation between the items. 
we need to assume some limitations 
1. each item is already in the **PI** list
2. no overlapping between the items (i.e the bike and 🛞 are considered overlapping)
X is the items, Y is the resulted relation t3 as example X= {helmet, ball } => Y = {bike}
if we could conclude the association rules resulted from the transactions we came with 3 categories
1. Trivial Rule
	   we don't need this as it seems intellectual conclusions, useless.
	-   Anyone who jumps from a high-rise building is likely to die.
	-   Working harder leads to better scores in exams.
	-   The sales of heaters increase as the temperature drops
	-   Driving a car over the speed limit on a highway leads to a higher chance of an accident.
1. inexplicable
	Among the rules that are generated after running the association rules algorithm, the ones that have no obvious explanation are the trickiest to use. Note that a rule can only be useful if it can help us discover and understand a new pattern that is expected to eventually lead toward a certain course of action. If that is not the case, and we cannot explain why event _X_ led to event _Y_, then it is an inexplicable rule, because it's just a mathematical formula that ends up exploring the pointless relationship between two events that are unrelated and independent.
	The following are examples of inexplicable rules:
	-   People who wear red shirts tend to score better in exams.
    
	-   Green bicycles are more likely to be stolen.
    
	-   People who buy pickles end up buying diapers as well.
1. Actionable 
	Actionable rules are the golden rules we are looking for. They are understood by the business and lead to insights. They can help us to discover the possible causes of an event when presented to an audience familiar with the business domain—for example, actionable rules may suggest the best placement in a store for a particular product based on current buying patterns. They may also suggest which items to place together to maximize their chances of selling as users tend to buy them together.

	The following are examples of actionable rules and their corresponding actions:

	-   **Rule 1:** Displaying ads to users' social media accounts results in a higher likelihood of sales.
		**Actionable item:** Suggests alternative ways of advertising a product

	-   **Rule 2:** Creating more price points increases the likelihood of sales.
		**Actionable item:** One item may be advertised in a sale, while the price of another item is raised.
	
#### Ranking Rules 
there are three ways to measure the Association rules. 
1. Support Frequency of items.
	Calculate the avg time each pattern of items. this will result in a number that represent the likely of an item to appear again 
	![[Pasted image 20230328222836.png]]
2. Confidence/ conditional probability 
	 where you assume the X has occurred let's get the conditional 
	 prob = Support(X U Y)/ Support(X) in the above example = 0.5
	 ![][https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781789801217/files/assets/1d0a3ece-4655-4222-b625-51cc092e2c38.png]
	this means that the person with helmet and ball in the basket would also have 50% will also have wickets.  
3. Lift / feedback
	measures how much improvement has been achieved after prediction 
	![][https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781789801217/files/assets/01fae583-e035-4d40-b73d-32212d50c0e2.png]

## Algorithms for association analysis
1. **Apriori algorithm**
	we consider the data in a binary representation where the item is present or absent where the other parameters are not considered. here we set two threshold Support threshold- Confidence Threshold. 
	1. the first phase extract the items that hits the support threshold 
	2. filter the item with high confidence 
	After filtering, the resulting rules are the answer.
	The major bottleneck in the apriori algorithm is the generation of candidate rules in Phase 1—for example, **PI** = {item 1 , item 2 , . . . , item m } can produce 2^m possible itemsets. Because of its multiphase design, it first generates these itemsets and then works toward finding the frequent itemsets. This limitation is a huge performance bottleneck and makes the apriori algorithm unsuitable for larger items.
2. **FP-growth algorithm**
	The **frequent pattern growth** (**FP-growth**) algorithm is an improvement on the apriori algorithm. It starts by showing the frequent transaction FP-tree, which is an ordered tree. It consists of two steps:
	-   Populating the FP-tree
	    we create a sparse matrix that shows the relations between the items. 
	    then calc the freq of the items then sort each transaction based on the freq from lower to higher.
	    we start by creating the FP-tree 
	    1. first insert the first transaction in ascending order of the items. 
	    ![][https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781789801217/files/assets/81575675-cbb3-418b-b2ba-e58240c27b40.png]
	    this tree is ordered tree.  the main evolution is the pattern retrieving will be more effective using the FP-tree. the data structure that helps in this is the pattern tree recognition.
	-   Mining frequent patterns
		 after constructing the above tree we are able to efficiently move to all patterns starting from the leaf to the NULL root. 
