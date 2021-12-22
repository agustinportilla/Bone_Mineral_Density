# Bone_Mineral_Density

**Project Domain**: Medicine

**Tools used**: Python

**Type of Algorithms used**: Logistics Regression, RandomForest, K-NearestNeighbours

**Project Summary**:

The objective of this work is to compare the performance of different algorithms when predicting the outcome in one Dataset. 

**Details about the Dataset**:

The Dataset bmd.csv contains 169 records of bone densitometries (measurement of bone mineral density). It contains the following variables:

   - **id**: patient’s number
   - **age** : patient’s age
   - **sex**: patient's gender
   - **fracture** : hip fracture (fracture / no fracture)
   - **weight_kg** : patient's weight in Kg
   - **height_cm** : patient's height in cm
   - **medication**: medication taken by the patient.
   - **waiting_time** : time the patient had to wait for the densitometry (in minutes)
   - **bmd** : patient's measured hip bone mineral density 

## Dataset processing using Logistics Regression ##

   - **Libraries used**: 
     - **For Preprocessing**: Pandas, Numpy and Matplotlib.pyplot
     - **For Model Creation**: Sklearn (datasets, feature_selection, linear_model, feature_selection.RFE) and LogisticRegression.
   
   - **Data Preprocessing details**: 
     - We used **describe** to spot any missing numeric field; 
     - **groupby** and **crosstab** to easily identify potential relations betweem the predicting variables and the target (**fracture/no-fracture**); 
     - **get_dummies**, **join** and **drop** to create a new dataset with a format that where we can apply Logistics Regression, 
     - We then removed the **id** column from the dataset, **dtypes** to check the column type and **astype** to transform string columns to integer.

   - **Model Creation with Sklearn**: 
     - We grouped our predicting variables under the value **X** and defined our target as **Y**
     - We used **RFE** and **LogisticRegression** to rank the impact of every predicting variable on the target variable. 
     - We run the model, obtaining a score of 0.8224852071005917, which means that 82.24...% of the targets were predicted correctly.
     - Based on the ranking previously obtain, wed remove some variables from **X**. We get our best score when we only use the variables **age**, **medication_No medication**, **waiting_time** and **bmd**: 84.02%.
     
   - **Validation**: 
     - **Cross Validation**: We apply cross validation, obtaining a result of 83.38%, which is line with our previous results. 
     - **Confussion Matrix** and **ROC Curve**: We obtain an auc of 81,70% (in line with the previous results).

   - **Logistics Regression Score:** 84.02%. 
     

## Dataset processing using Tree/Forest ##
   - **Libraries used**: 
     - **For Preprocessing**: Pandas, Numpy and Matplotlib.pyplot
     - **For Model Creation**: Sklearn (tree.DecisionTreeClassifier, model_selection.KFold, model_selection.cross_val_score, ensemble.RandomForestClassifier)
     - **For Tree Visualization**: tree.export_graphviz, os, sys, graphviz.Source
   
   - **Data Preprocessing details**: 
     - We used **describe** to spot any missing numeric field; 
     - **groupby** and **crosstab** to easily identify potential relations betweem the predicting variables and the target (**fracture/no-fracture**); 
     - **get_dummies**, **join** and **drop** to create a new dataset with a format that where we can apply Logistics Regression, 
     - We then removed the **id** column from the dataset, **dtypes** to check the column type and **astype** to transform string columns to integer.

   - **Model Creation with DecisionTreeClassifier**: 
     - We grouped our predicting variables under the value **predictors** and defined our target as **target**
     - We divided our dataset into a **training set** (aprox 75%) and a **testing set** (aprox 25%).
     - We used **DecisionTreeClassifier** to create and fit our model (using the **training set**).
     - We calculated our score using the **testing set**: Our score is 83.33%. 
     - We validated these results. They are correct: We have 35 cases in which the model predicted correctly and 7 in which it did not (35/42 = 0.833).
     - One comment regarding this method: If we split our original dataset, and run our model again, we will obtain different results. 
     - Fortunately, the following two methods perform better in terms of variability.
     
   - **Model Creation with Cross Validation**: 
     - We used **KFold** and **cross_val_score** to build our model. We used the complete dataset (no need to divide the dataset into **training** and **testing** here).
     - We calculated our score using a **max_depth** of 5. Our score is 82.2%.
     - We created a loop to understand how we can change **max_depth** to improve the model. We discovered that a **max_depth** of 1 or 2 returns a score of 88.78%.

   - **Model Creation with RandomForestClassifier**: 
     - We build our model using the **training set**.
     - We set the n_estimators parameter to 10000 (that will be the quantity of trees that the model will create).
     - Our score is 85.03%.

   -  **Tree/Forest Score:** 88.78%. 

## Dataset processing using KNN ##
   - **Libraries used**: 
     - **For Preprocessing**: Pandas and Numpy
     - **For Model Creation**: Sklearn (preprocessing, neighbors, model_selection.cross_val_score, model_selection.train_test_split)

   - **Data Preprocessing details**: 
     - We used **describe** to spot any missing numeric field; 
     - **groupby** and **crosstab** to easily identify potential relations betweem the predicting variables and the target (**fracture/no-fracture**); 
     - **get_dummies**, **join** and **drop** to create a new dataset with a format that where we can apply Logistics Regression, 
     - We then removed the **id** column from the dataset, **dtypes** to check the column type and **astype** to transform string columns to integer.

   - **Model Creation with KNeighborsClassifier**: 
     - We grouped our predicting variables under the value **X** and defined our target as **Y**
     - We divided our dataset into a **training set** (aprox 70%) and a **testing set** (aprox 30%).
     - We used **KNeighborsClassifier** to create and fit our model (using the **training set**).
     - We calculated our score using the **testing set**: Our score is 74.51%. 
     - One comment regarding this method: If we split our original dataset, and run our model again, we will obtain different results. 
     - To avoid this, we can run our models multiple times and calculate average accuracy (as we will do next).
     
   - **Model Fine Tuning**
     - We will fine tune our model using two methods. 
     - We first will choose the best combination of columns to be used as predictors.
     - Then we will modify the parameters contained inside KNeighborsClassifier (algorithm, leaf_size, metric, metric_params, n_neighbors, p, weights).
     - In all these cases, we will use 1000 iterations and average the results:
     - **Columns**: The combination of columns that give us the best results when used as predictors are 'age', 'weight_kg', 'height_cm', 'bmd'.
     - **N_Neighbors**: When using the value 25, our model gives the best results.
     - For all the other parameters the same.

   - **KNN Score:** 77.74%. 


## Final Conclusions ##
   - As we have seen, we have analysed our Dataset using three different kind of algorithms: Logistic Regression, Random Tree/Forest and KNN.
   - When we compared their results, we found out that the performance of Random Tree/Forest was the best (88.78%)
   - Random Tree/Forest performed significately higher than Logistics Regression (84.02%) or KNN (77.74%).
