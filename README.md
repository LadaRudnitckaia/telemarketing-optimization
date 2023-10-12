# telemarketing-optimization

Develoment of a machine learning model optimizing telemarketing through prediction of marketing calls that don't lead to customer conversion.

The dataset used in this project is: https://archive.ics.uci.edu/dataset/222/bank+marketing

## Business problem
Head of marketing in a bank requests data team to optimize the telemarketing by reducing the number of unnecessary calls to existing clients about a product offer.

The current process is: select a subset of customers every week and call them. The cost of each call is 8€. Each new contract will generate ~80€ profits over its lifetime.

Request details:
- Build a Machine Learning pipeline to identify customers that should be called within the weekly cohort and provide recommendations on how to reduce the number of calls.
- Predict "wasted calls" (calls to non-converting customers) and analuyze how many calls can be saved.
- Lose as little business as possible. 


## Solution
### Framing of the data science problem
There are two outcomes of each call: a client bought a product, or in other words, converted, or didn't. This is a binary classification problem: with the target "converted" and two classes: "yes" (1) and "no" (0). These classes can also be interpreted as "worthy call" and "wasted call" to be able to use the model for the call center employees recommendations.

Next, a binary classificaiton model can predict a probability of each class and we have an opportunity to decide how to use these probabilities to predict the class itself by optimizing the threshold according to the business requirements.

Initial analysis of historical data shows that only around 11% of customers convert after a call. This means, we have a dataset with very imbalanced classes: 11% for class 1 and 89% for class 0. 

At the same time, the cost of a wasted call (8$) is much lower than a profit from a worthy call (80$). This means, that the effect of two types of errors the model can make (False Positive (FP) and False Negative (FN) predicitons) is not equal:
- False Positive (FP) prediciton corresponds to a wasted call wrongfully predicted to be worthy. The cost of this error for each call is 8$.
- False Negative (FN) prediciton corresponds to a worthy call wrongfully predicted to be a wasted one. The opportunity cost of this error for each call is 80$.

The fact that classes are imbalanced and the cost of two errors are different imply the following:
- We need to use a metric different to accuracy to evaluate a model and a metric that prioritizes minimizing FN predictions. We, therefore, will use Recall to fine tune and evaluate the model.
- We need to optimize the model threshold (default 0.5) to optimally use the probabilities predicted by the model according to business requirements, in this case according to costs, profits, and opportunity costs from wrongfully predicted calls.

### Solution
#### Development
We start the development of the data pipeline with the **Explorative Data Analysis** (EDA) and explore data types, values, missing values, and correlations. We decide to keep "unknown" category (missing value) as a separate category and keep all the features to train a model.

Next, we **preprocess the data for a model training** as follows: we split the data to features and target, converts target column to numeric data type, split the datasets to training, validation, and test samples (before one-hot encoding to avoid data leakage), and then fit a one-hot encoder (only for categorical columns) on the training sample, and apply it to training, validations, and test samples.  Finally, we save the one-hot encoder object to use be able to use it during the deployment.

We select a **Random Forest algorithm** to train a machine learning model. Ideally, various algorithms should be tried and the best performing one should be selected, e.g., with an AutoML tool. However, for the scope of this project, we only experiment with the Random Forest. This algorithm tends to provide better result than, e.g., a single Decision Tree since it trains a bag of multiple single models and then takes a decision based on single models voting.

Next, we train several Random Forest models with different hyperparameters using Random search and **fine tune the hyperparameters** by evaluating using the cross validation. We use training data sample for this. Once the best hyperparameters are found, we train the final, best model and save it.

For this best model, we **optimize the threshold** by evaluating the "clean gain" (probably, there is a better definition for this metric!) from a model usage for different thresholds on the validation data sample. The optimized threshold is saved. We define the "clean gain" as follows: total gain minus opportunity costs, where:
- Total gain = [correctly predicted as worthy calls] * 80$ - [all predicted as worthy calls] * 8$
- Opportunity costs = [falsely predicted as wasted calls] * (80$ - 8$)

Finally, we **evaluate** the final model with the best threshold on the test set and find the following:
- While the model achieves rather low F1 score and Preicision, it achieves a satisfying Recall of ~0.9.
- Looking at the confusion matrix, we can see that:
    - the model predicts very few False Negatives ~1% meaning we lose very little potential contracts
    - the model still offers to make quite a lot of calls that won't lead to conversion (~51%) but it does reduce the number of wasted calls by correctly preidcting ~38% of wasted calls
- We compare the profit from using a model to the current strategy (calling to all the clients in the weekly cohort) and find out that although using the model does yeild significant opportunity costs (we lose ~10% of opportunties), the profit from reducing the wasted calls is very significant. The profit from using the model is ~13K$ whereas the profit from the current strategy is ~4K$.

Additionally, we analyze how the model makes predicitons, by calculating **global feature importance** available for a random forest model. We find that the model considers the following features as the most important: 'euribor3m', 'nr.employed', 'emp.var.rate', 'cons.conf.idx', 'cons.price.idx', 'pdays', 'poutcome_success', 'age', 'month_may', 'contact_telephone', 'previous', 'campaign', 'contact_cellular'. Surprisingly, top 5 most important features describe social and economic context and not individual charcteristics of the clients. Among the important features are also the features that describe how previous marketing campaigns went for this client. 
- However, these feature importance values mean how often a feature was selected by a decision tree on the high level of the tree because it has good dataset splitting power and can't be considered as genral drivers of conversion. Other methods should be used to analyze the real factors that lead to conversion. In the scope of this project, we only had a look at correlation and logistic regression coefficients to double check whether the features important for a random forest model are also important generally. The features that were also important for logistic regression are: 'cons.price.idx', 'month_may' 'euribor3m', 'emp.var.rate', 'poutcome_failure', 'previous'. This confirms that the features describing social and economic context and previous campaigns results can be important for conversion. At the same the feature 'month_may' might be correlated with some other feature that has real impact on conversion.
- The results of correlation analysis, also confirm that features describing social and economic context and previous campaigns results can be important for conversion since features 'nr.employed', 'pdays', 'euribor3m', and 'emp.var.rate' have negative correlation with the target and features 'previous' and 'poutcome_success' - have positive correlation with the target.

#### Deployment
In the final **ML pipeline** that can be used to predict the worthiness of the call for the new data, we include the data preprocessing and application of the model with the correct threshold. We develop a Python module that can be used to make predictions in Python and, additionally, we deploy the final ML pipeline as a **REST service**. The predictions then can be requested from any application that can send a POST request and transfer JSON data.

Given, that the predictions are needed on a weekly basis, we can precompute the predictions in a **batch**, for example, in the beginning of the week and send the results to the call center.

Besides, considering that all the recommended calls are made during the week, and the ground truth for these data is also received right after the call (whether the call was indeed worthy or not), we can set up a **regular model monitoring**, and when needed, **retraining and redeployment** of the model to ensure the model reflects the latest production data. For that, model training can be also packed into a module or deployed as a service that can be called when retraining is needed.

### Repo structure
The repo consists of four folders:
- data - contains zipped and archived data
- models - contains saved objects needed for the pipeline: the one-hot encoder, the model, and the threshold
- modules - contain functions for data preprocessing, for making predicitons, for deployment as a REST service, and (example, not finalized) for training
- notebooks - contain EDA, a notebook where current model was trained and evaluated, and an example of how the pipeline can be used