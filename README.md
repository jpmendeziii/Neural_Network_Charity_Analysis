# Neural_Network_Charity_Analysis
## Overview
Using knowledge of Pandas and the Scikit-Learn’s StandardScaler(), the purpose of this analysis was to employ deep learning models in to aid a fictitious company named AlphabetSoup, a philanthropic organization dedicated to donating funds to organizations whose goals it deemed worthwhile, in vetting donation applicants. By training a deep learning model on a retrospective dataset of about 30,000 historical donations for both successful and unsuccessful outcomes, the model served as a binary classifier to predict whether applicants would be successful if given funding. In short, this would help AlphabetSoup know where to place their money.

The deep learning model performed supervised learning by training on the "feature" columns from the dataset to predict a "target" column containing binary values of "0" or "1", as these numbers were indicative of whether the donation of each observation produced an unsuccessful or successful outcome, respectively. To optimize the performance and prepare it for the neural network model, the data required significant preprocessing. Following this, the data was compiled, trained, and the results were evaluated. The details and results of this process are described further below.

![Resources/deliverable1_1.jpg](Resources/deliverable1_1.jpg)
![Resources/deliverable1_2.jpg](Resources/deliverable1_2.jpg)
![Resources/deliverable2_1.jpg](Resources/deliverable2_1.jpg)
![Resources/deliverable2_2.jpg](Resources/deliverable2_2.jpg)
![Resources/deliverable2_3.jpg](Resources/deliverable2_3.jpg)

## Results

* The data was inspected, and a few initial observations were analyzed prior to building the neural network. The preprocessing began by dropping two unnecessary columns, "EIN" and "NAME", which served as neither targets nor features since they were considered unusual for predicting the success of a donation.

* The target variable for this model was the "IS_SUCCESSFUL" column.
* Next, the number of unique values from each column was calculated using the unique() method from the pandas library. This allowed us to visualize which columns had more than 10 unique values and, thus, would need bucketing into the "other" column to reduce the number of dummy columns that would result when converting categorical variables to numerical format later. Note, two categorical variables, "APPLICATION_TYPE" and "CLASSIFICATION" fell into this category. "ASK_AMT" was a numerical column and thus did not need attention.
* To collapse all the infrequent categorical values into a single "other" category for each of the two columns mentioned prior, the value counts were first decided. For instance, the value counts for "APPLICATION_TYPE".
* Next, a filtered list of these value counts was created to only include values from the column which occurred infrequently. In this case, we chose values which occurred less than five-hundred times in the column. Finally, a for loop combined with the replace method allowed us to replace all those values with "other", thus reducing the number of unique values in the column (see image below). The same procedure was used on the other relevant column ("CLASSIFICATION") as well. Again, this reduction in unique values made it less resource-intensive when employing one-hot-encoder later.
* A list was created having only object/string types (categorical types) and the "one hot encoder" was used to create dummy variables. This process was necessary as its use meant all our features were now in numerical format, as needed prior to the neural network implementation.
* Finally, the dataset was separated in the appropriate "y" target and "X" features as well as split into training and testing sets accordingly. Furthermore, the data feature data was scaled which altered each variable to have a mean value of zero and a standard deviation of one. This normalization process prevented variations in the size scaling between columns from over-representing influence on predictions of the target during training.
![Resources/AlphabetSoupCharity_OPTIMIZATION_v2_Accuracy.jpg](Resources/AlphabetSoupCharity_OPTIMIZATION_v2_Accuracy.jpg)
