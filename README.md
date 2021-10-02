# Hepatitis C prediction 
The main idea of this small project is to check whether the patient has problems with liver caused by Hepatitis C virus. The data set contains laboratory values of blood donors and Hepatitis C patients with demographic values like age.
The data was obtained from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/HCV+data

# Table of contents:
- [The dataset] (#The_dataset)
- [Project](#Project)
- [Preprocessing](#Preprocessing)
- [Training](#Training)
- [Predicting](#Predicting)
- [Further tasks](#Further_tasks)

## The dataset <a name="The_dataset"></a>
All attributes except Category and Sex are numerical.
Attributes 1 to 4 refer to the data of the patient:
1) X (Patient ID/No.)
2) Category (diagnosis) (values: '0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis')
3) Age (in years)
4) Sex (f,m)
Attributes 5 to 14 refer to laboratory data:
5) ALB
6) ALP
7) ALT
8) AST
9) BIL
10) CHE
11) CHOL
12) CREA
13) GGT
14) PROT

The dataset has got not only HCV but also it includes its progress stages but for this small dataset it is better to just divde them into: healthy and sick patients 

## Preprocessing of the data <a name="Preprocessing"></a>
The first thing to do is dropping the first column and changing columns "Sex" and "Category" to numbers. What was said before there will be only binary predicitons that someone has got HCV problems(1) or has got no problems with liver (0). 
But there are some problems with dataset:
1. Missing data 
Solution: Filling all rows where NANs value occur and replace them with the average of the column
2. There are lots of outliers, what we can see on the plots and the skewness of data. 
Solution: log transformation of the columns that have the most outliers. We will be using Robust Scaler from sklearn.  
3. The most serious problem - imbalanced data. 

Solution: Usage of the ADASYN algorithm to oversample data where patient has got HCV problems. 

#### Correlation heatmap 


## Training of the data <a name="Training"></a>

## Final predictions <a name="Evaluating"></a>

## Future tasks <a name="Future_tasks"></a>
Testing the usage of two models: xgboost and lightgbm 
