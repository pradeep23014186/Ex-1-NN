<H3>NAME: Pradeep kumar G</H3>
<H3>REGISTER NO: 212223230150</H3>
<H3>EX. NO.1</H3>
<H3>DATE:</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
 import pandas as pd
 import io
 from sklearn.preprocessing import StandardScaler
 from sklearn.preprocessing import MinMaxScaler
 from sklearn.model_selection import train_test_split
 df = pd.read_csv("Churn_Modelling.csv")
 df
 df.isnull().sum()
 df.fillna(0)
 df.isnull().sum()
 df.duplicated()
 df['EstimatedSalary'].describe()
 scaler = StandardScaler()
 inc_cols = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary']
 scaled_values = scaler.fit_transform(df[inc_cols])
 df[inc_cols] = pd.DataFrame(scaled_values, columns = inc_cols, index = df.index)
 df
 x = df.iloc[:, :-1]
 y = df.iloc[:, -1]
 print("X Values")
 x
 print("Y Values")
 y
 x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_stat
 print("X Training data")
 x_train
 print("X Testing data")
 x_test
```

## OUTPUT:
# DATASET:
<img width="1388" height="497" alt="image" src="https://github.com/user-attachments/assets/fd421f1b-86be-4a72-92b8-8cb5fb1b8be1" />

# MISSING VALUES:
<img width="228" height="339" alt="image" src="https://github.com/user-attachments/assets/8db399ba-30c2-46ae-bf80-0f652cb21e04" />

# DUPLICATES:
<img width="314" height="271" alt="image" src="https://github.com/user-attachments/assets/4608c4a9-b995-486f-a6df-92a60c6fbce7" />

# OUTLIERS (SALARY):
<img width="343" height="205" alt="image" src="https://github.com/user-attachments/assets/1a346111-dc5c-4a6e-badd-1ae252f78c8f" />

# NORMALIZED DATASET:
<img width="1379" height="518" alt="image" src="https://github.com/user-attachments/assets/83d69416-5792-40b9-86e4-ff0fa02828ba" />

# X_VALUES:



 

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


