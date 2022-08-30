#import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Read Csv
train_df = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("test.csv", dtype={"Age": np.float64}, )

#Print Train Head
print("\n\nTop of the training data:")
print(train_df.head())

#Print Train Statistics
print("\n\nSummary statistics of training data")
print(train_df.describe())

#Combine datasets
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine_df = [train_df, test_df]

#Categorize Name
for dataset in combine_df:
    dataset["Name"]= pd.DataFrame([i.split(",")[1].split(".")[0].strip() for i in dataset['Name']])
    
    dataset['Name'] = dataset['Name'].replace(['Don', 'Rev', 'Dr', 'Mme',
                                                   'Ms','Major', 'Lady', 'Sir',
                                                   'Mlle', 'Col', 'Capt', 'the Countess',
                                                   'Jonkheer', 'Dona'],'Rare')
    dataset["Name"] = dataset['Name'].map({'Master': 0, 'Mr':1, 'Miss':2, 'Mrs':2, 'Rare':3})    
    
#Categorize Age
for dataset in combine_df:
    dataset.Age = dataset.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    dataset.loc[ dataset['Age'] < 0, 'Age'] = -1
    dataset.loc[(dataset['Age'] >=0) & (dataset['Age'] <= 5), 'Age'] = 0
    dataset.loc[(dataset['Age'] > 5) & (dataset['Age'] <= 12), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 18), 'Age'] = 12
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 25), 'Age'] = 18
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 35), 'Age'] = 25
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 50), 'Age'] = 35
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 50
    dataset.loc[(dataset['Age'] > 60) & (dataset['Age'] <= 120), 'Age'] = 60
    dataset.loc[ dataset['Age'] > 120, 'Age'] = 120
    dataset['Age'] = dataset['Age'].astype(int)

#Categorize Sex 
for dataset in combine_df:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine_df:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

#Categorize Embarked 
for dataset in combine_df:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

#Categorize Fare 
for dataset in combine_df:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#Prepare X,y train and test dataframes 
X_train=train_df.drop("Survived",axis=1)
y_train=train_df["Survived"]
X_train  = X_train.drop("PassengerId", axis=1).copy()
X_test  = test_df.drop("PassengerId", axis=1).copy()
#X_train.shape, Y_train.shape, X_test.shape
#print(X_train.head());print(X_test.head())

#Random Forest Function 
random_forest = RandomForestClassifier(random_state = 52, warm_start = True,  n_estimators=250, oob_score=True, n_jobs=6  )
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print("Random Forest" + str(acc_random_forest))

#submission
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission1.csv', index=False)


#Knn
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 
                           weights='uniform')
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
score =knn.score(X_train,y_train)
acc_knn = round(score * 100, 2)
print("Random Forest :" + str(acc_knn))
knn_predict = knn.predict(X_test)

#submission
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": knn_predict
    })
submission.to_csv('knn_submission.csv', index=False)



