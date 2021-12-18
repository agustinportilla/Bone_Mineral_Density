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

   - Libraries used: Pandas, Numpy and Matplotlib.pyplot (for data preprocessing). Sklearn.datasets, Sklearn.feature_selection, Sklearn.linear_model and Sklearn.linear_model.RFE and import LogisticRegression (for Model Creation).
   - Data Preprocessing steps: We used **describe** to spot any missing numeric field; **groupby** and **crosstab** to easily identify potential relations betweem the predicting variables and the target (**fracture/no-fracture**); **get_dummies**, **join** and **drop** to create a new dataset with a format that where we can apply Logistics Regression, we then removed the **id** column from the dataset and used **astype** to make sure all our fields are numeric.
   - Model Creation with Sklearn: 
     - We grouped our predicting variables under the value **X** and defined our target as **Y**
     - We used **RFE** and **LogisticRegression** to rank the impact of every predicting variable on the target variable. This information will be used later, when we modify the categories included under the value **X**.
     - We run the model, obtaining a score of 0.8224852071005917, which means that 82.24...% of the targets were predicted correctly.
     - Based on the ranking previously obtain, we remove some variables from **X**. We get our best score when we only use the variables **age**, **medication_No medication**, **waiting_time** and **bmd**: 0.8402366863905325. This means that 142 from the 169 targets predicted are correct (84.02...%).
   - Validation: We apply various validation methods (from this point, **X** will only include the variables **age**, **medication_No medication**, **waiting_time** and **bmd**).
     - Cross Validation: We apply cross validation, obtaining a result of 83.38%, which is line with our previous results. 
     - Confussion Matrix and ROC Curve: We obtain an auc of 81,70% (in line with the previous results).
     
