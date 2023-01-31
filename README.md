# Classification-of-Glass
This is a dataset from USA Forensic Science Service which has description of 6 types of glass; defined in terms of their oxide content (i.e. Na, Fe, K, etc). Using KNN Algorithm to classify the glasses


#(https://archive.ics.uci.edu/ml/datasets/glass+identification).Load the dataset from here.
# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

# loading the data
df = pd.read_csv('glass.csv')
df.head()
        RI     Na    Mg    Al     Si     K    Ca   Ba   Fe  Type
0  1.52101  13.64  4.49  1.10  71.78  0.06  8.75  0.0  0.0     1
1  1.51761  13.89  3.60  1.36  72.73  0.48  7.83  0.0  0.0     1
2  1.51618  13.53  3.55  1.54  72.99  0.39  7.78  0.0  0.0     1
3  1.51766  13.21  3.69  1.29  72.61  0.57  8.22  0.0  0.0     1
4  1.51742  13.27  3.62  1.24  73.08  0.55  8.07  0.0  0.0     1

# value count for glass types
df.Type.value_counts()
2    76
1    70
7    29
3    17
5    13
6     9
Name: Type, dtype: int64

#Data exploration and visualizaion
#correlation matrix 
cor = df.corr()
sns.heatmap(cor)
<matplotlib.axes._subplots.AxesSubplot at 0x1b1c6b4c248>

![ccorr](https://user-images.githubusercontent.com/114566844/215673250-dd29b057-90c0-4163-9837-1c5635254a7b.png)


sns.scatterplot(df_feat['RI'],df_feat['Na'],hue=df['Type'])
<matplotlib.axes._subplots.AxesSubplot at 0x1b1c6c3cd48>

#pairwise plot of all the features
sns.pairplot(df,hue='Type')
plt.show()

scaler = StandardScaler()

scaler.fit(df.drop('Type',axis=1))
StandardScaler(copy=True, with_mean=True, with_std=True)
#perform transformation
scaled_features = scaler.transform(df.drop('Type',axis=1))
scaled_features
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

	RI	Na	Mg	Al	Si	K	Ca	Ba	Fe
0	0.872868	0.284953	1.254639	-0.692442	-1.127082	-0.671705	-0.145766	-0.352877	-0.586451
1	-0.249333	0.591817	0.636168	-0.170460	0.102319	-0.026213	-0.793734	-0.352877	-0.586451
2	-0.721318	0.149933	0.601422	0.190912	0.438787	-0.164533	-0.828949	-0.352877	-0.586451
3	-0.232831	-0.242853	0.698710	-0.310994	-0.052974	0.112107	-0.519052	-0.352877	-0.586451
4	-0.312045	-0.169205	0.650066	-0.411375	0.555256	0.081369	-0.624699	-0.352877	-0.586451


dff = df_feat.drop(['Ca','K'],axis=1) #Removing features - Ca and K 
X_train,X_test,y_train,y_test  = train_test_split(dff,df['Type'],test_size=0.3,random_state=45) 
knn = KNeighborsClassifier(n_neighbors=4,metric='manhattan')
knn.fit(X_train,y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=4, p=2,
                     weights='uniform')
y_pred = knn.predict(X_test)
print(classification_report(y_test,y_pred))

   precision    recall  f1-score   support

           1       0.69      0.90      0.78        20
           2       0.85      0.65      0.74        26
           3       0.00      0.00      0.00         3
           5       0.25      1.00      0.40         1
           6       0.50      0.50      0.50         2
           7       1.00      0.85      0.92        13

    accuracy                           0.74        65
   macro avg       0.55      0.65      0.56        65
weighted avg       0.77      0.74      0.74        65


accuracy_score(y_test,y_pred)
0.7384615384615385

k_range = range(1,25)
k_scores = []
error_rate =[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #kscores - accuracy
    scores = cross_val_score(knn,dff,df['Type'],cv=5,scoring='accuracy')
    k_scores.append(scores.mean())
    
    #error rate
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred!=y_test))

#plot k vs accuracy
plt.plot(k_range,k_scores)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Cross validated accuracy score')
plt.show()

#plot k vs error rate
plt.plot(k_range,error_rate)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Error rate')
plt.show()




