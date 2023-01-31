# Classification-of-Glass
This is a dataset from USA Forensic Science Service which has description of 6 types of glass; defined in terms of their oxide content (i.e. Na, Fe, K, etc). Using KNN Algorithm to classify the glasses


#(https://archive.ics.uci.edu/ml/datasets/glass+identification).Load the dataset from here.

#import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

#loading the data
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

 "cells": [
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import classification_report\n",
    "from scipy.spatial import distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read train and test sets\n",
    "train = pd.read_csv(\"trainKNN.txt\", header=None)\n",
    "train.columns = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type of glass']\n",
    "train = train.drop('ID', axis=1) # Drop ID since irrelevant to predictions\n",
    "test = pd.read_csv('testKNN.txt', header=None)\n",
    "test.columns=['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type of glass']\n",
    "test = test.drop('ID', axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RI</th>\n",
       "      <th>Na</th>\n",
       "      <th>Mg</th>\n",
       "      <th>Al</th>\n",
       "      <th>Si</th>\n",
       "      <th>K</th>\n",
       "      <th>Ca</th>\n",
       "      <th>Ba</th>\n",
       "      <th>Fe</th>\n",
       "      <th>Type of glass</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.52101</td>\n",
       "      <td>13.64</td>\n",
       "      <td>4.49</td>\n",
       "      <td>1.10</td>\n",
       "      <td>71.78</td>\n",
       "      <td>0.06</td>\n",
       "      <td>8.75</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.51761</td>\n",
       "      <td>13.89</td>\n",
       "      <td>3.60</td>\n",
       "      <td>1.36</td>\n",
       "      <td>72.73</td>\n",
       "      <td>0.48</td>\n",
       "      <td>7.83</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.51618</td>\n",
       "      <td>13.53</td>\n",
       "      <td>3.55</td>\n",
       "      <td>1.54</td>\n",
       "      <td>72.99</td>\n",
       "      <td>0.39</td>\n",
       "      <td>7.78</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.51766</td>\n",
       "      <td>13.21</td>\n",
       "      <td>3.69</td>\n",
       "      <td>1.29</td>\n",
       "      <td>72.61</td>\n",
       "      <td>0.57</td>\n",
       "      <td>8.22</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.51742</td>\n",
       "      <td>13.27</td>\n",
       "      <td>3.62</td>\n",
       "      <td>1.24</td>\n",
       "      <td>73.08</td>\n",
       "      <td>0.55</td>\n",
       "      <td>8.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        RI     Na    Mg    Al     Si     K    Ca   Ba   Fe  Type of glass\n",
       "0  1.52101  13.64  4.49  1.10  71.78  0.06  8.75  0.0  0.0              1\n",
       "1  1.51761  13.89  3.60  1.36  72.73  0.48  7.83  0.0  0.0              1\n",
       "2  1.51618  13.53  3.55  1.54  72.99  0.39  7.78  0.0  0.0              1\n",
       "3  1.51766  13.21  3.69  1.29  72.61  0.57  8.22  0.0  0.0              1\n",
       "4  1.51742  13.27  3.62  1.24  73.08  0.55  8.07  0.0  0.0              1"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RI</th>\n",
       "      <th>Na</th>\n",
       "      <th>Mg</th>\n",
       "      <th>Al</th>\n",
       "      <th>Si</th>\n",
       "      <th>K</th>\n",
       "      <th>Ca</th>\n",
       "      <th>Ba</th>\n",
       "      <th>Fe</th>\n",
       "      <th>Type of glass</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.52152</td>\n",
       "      <td>13.05</td>\n",
       "      <td>3.65</td>\n",
       "      <td>0.87</td>\n",
       "      <td>72.32</td>\n",
       "      <td>0.19</td>\n",
       "      <td>9.85</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.17</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.52152</td>\n",
       "      <td>13.12</td>\n",
       "      <td>3.58</td>\n",
       "      <td>0.90</td>\n",
       "      <td>72.20</td>\n",
       "      <td>0.23</td>\n",
       "      <td>9.82</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.16</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.52300</td>\n",
       "      <td>13.31</td>\n",
       "      <td>3.58</td>\n",
       "      <td>0.82</td>\n",
       "      <td>71.99</td>\n",
       "      <td>0.12</td>\n",
       "      <td>10.17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.03</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.51709</td>\n",
       "      <td>13.00</td>\n",
       "      <td>3.47</td>\n",
       "      <td>1.79</td>\n",
       "      <td>72.72</td>\n",
       "      <td>0.66</td>\n",
       "      <td>8.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.51660</td>\n",
       "      <td>12.99</td>\n",
       "      <td>3.18</td>\n",
       "      <td>1.23</td>\n",
       "      <td>72.97</td>\n",
       "      <td>0.58</td>\n",
       "      <td>8.81</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.24</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        RI     Na    Mg    Al     Si     K     Ca   Ba    Fe  Type of glass\n",
       "0  1.52152  13.05  3.65  0.87  72.32  0.19   9.85  0.0  0.17              1\n",
       "1  1.52152  13.12  3.58  0.90  72.20  0.23   9.82  0.0  0.16              1\n",
       "2  1.52300  13.31  3.58  0.82  71.99  0.12  10.17  0.0  0.03              1\n",
       "3  1.51709  13.00  3.47  1.79  72.72  0.66   8.18  0.0  0.00              2\n",
       "4  1.51660  12.99  3.18  1.23  72.97  0.58   8.81  0.0  0.24              2"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RI</th>\n",
       "      <th>Na</th>\n",
       "      <th>Mg</th>\n",
       "      <th>Al</th>\n",
       "      <th>Si</th>\n",
       "      <th>K</th>\n",
       "      <th>Ca</th>\n",
       "      <th>Ba</th>\n",
       "      <th>Fe</th>\n",
       "      <th>Type of glass</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>196.000000</td>\n",
       "      <td>196.000000</td>\n",
       "      <td>196.000000</td>\n",
       "      <td>196.000000</td>\n",
       "      <td>196.000000</td>\n",
       "      <td>196.000000</td>\n",
       "      <td>196.000000</td>\n",
       "      <td>196.000000</td>\n",
       "      <td>196.000000</td>\n",
       "      <td>196.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.518295</td>\n",
       "      <td>13.375204</td>\n",
       "      <td>2.758980</td>\n",
       "      <td>1.454337</td>\n",
       "      <td>72.635408</td>\n",
       "      <td>0.519388</td>\n",
       "      <td>8.910714</td>\n",
       "      <td>0.164235</td>\n",
       "      <td>0.050255</td>\n",
       "      <td>2.668367</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.003055</td>\n",
       "      <td>0.783145</td>\n",
       "      <td>1.392641</td>\n",
       "      <td>0.491688</td>\n",
       "      <td>0.763578</td>\n",
       "      <td>0.672703</td>\n",
       "      <td>1.421490</td>\n",
       "      <td>0.485198</td>\n",
       "      <td>0.086359</td>\n",
       "      <td>2.062416</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.511310</td>\n",
       "      <td>10.730000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.290000</td>\n",
       "      <td>69.810000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>5.430000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.516458</td>\n",
       "      <td>12.877500</td>\n",
       "      <td>2.362500</td>\n",
       "      <td>1.190000</td>\n",
       "      <td>72.317500</td>\n",
       "      <td>0.140000</td>\n",
       "      <td>8.220000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.517630</td>\n",
       "      <td>13.280000</td>\n",
       "      <td>3.480000</td>\n",
       "      <td>1.360000</td>\n",
       "      <td>72.810000</td>\n",
       "      <td>0.560000</td>\n",
       "      <td>8.575000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.518985</td>\n",
       "      <td>13.792500</td>\n",
       "      <td>3.610000</td>\n",
       "      <td>1.622500</td>\n",
       "      <td>73.080000</td>\n",
       "      <td>0.610000</td>\n",
       "      <td>9.092500</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.090000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.533930</td>\n",
       "      <td>15.790000</td>\n",
       "      <td>4.490000</td>\n",
       "      <td>3.500000</td>\n",
       "      <td>75.180000</td>\n",
       "      <td>6.210000</td>\n",
       "      <td>16.190000</td>\n",
       "      <td>3.150000</td>\n",
       "      <td>0.340000</td>\n",
       "      <td>7.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               RI          Na          Mg          Al          Si           K  \\\n",
       "count  196.000000  196.000000  196.000000  196.000000  196.000000  196.000000   \n",
       "mean     1.518295   13.375204    2.758980    1.454337   72.635408    0.519388   \n",
       "std      0.003055    0.783145    1.392641    0.491688    0.763578    0.672703   \n",
       "min      1.511310   10.730000    0.000000    0.290000   69.810000    0.000000   \n",
       "25%      1.516458   12.877500    2.362500    1.190000   72.317500    0.140000   \n",
       "50%      1.517630   13.280000    3.480000    1.360000   72.810000    0.560000   \n",
       "75%      1.518985   13.792500    3.610000    1.622500   73.080000    0.610000   \n",
       "max      1.533930   15.790000    4.490000    3.500000   75.180000    6.210000   \n",
       "\n",
       "               Ca          Ba          Fe  Type of glass  \n",
       "count  196.000000  196.000000  196.000000     196.000000  \n",
       "mean     8.910714    0.164235    0.050255       2.668367  \n",
       "std      1.421490    0.485198    0.086359       2.062416  \n",
       "min      5.430000    0.000000    0.000000       1.000000  \n",
       "25%      8.220000    0.000000    0.000000       1.000000  \n",
       "50%      8.575000    0.000000    0.000000       2.000000  \n",
       "75%      9.092500    0.000000    0.090000       3.000000  \n",
       "max     16.190000    3.150000    0.340000       7.000000  "
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RI</th>\n",
       "      <th>Na</th>\n",
       "      <th>Mg</th>\n",
       "      <th>Al</th>\n",
       "      <th>Si</th>\n",
       "      <th>K</th>\n",
       "      <th>Ca</th>\n",
       "      <th>Ba</th>\n",
       "      <th>Fe</th>\n",
       "      <th>Type of glass</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.519131</td>\n",
       "      <td>13.763333</td>\n",
       "      <td>1.873889</td>\n",
       "      <td>1.342222</td>\n",
       "      <td>72.820000</td>\n",
       "      <td>0.253889</td>\n",
       "      <td>9.460556</td>\n",
       "      <td>0.292778</td>\n",
       "      <td>0.130556</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.002791</td>\n",
       "      <td>1.083491</td>\n",
       "      <td>1.749753</td>\n",
       "      <td>0.581312</td>\n",
       "      <td>0.892004</td>\n",
       "      <td>0.265133</td>\n",
       "      <td>1.380432</td>\n",
       "      <td>0.617422</td>\n",
       "      <td>0.164798</td>\n",
       "      <td>2.222876</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.511150</td>\n",
       "      <td>12.850000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.340000</td>\n",
       "      <td>71.360000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>6.650000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.517413</td>\n",
       "      <td>13.012500</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.877500</td>\n",
       "      <td>72.212500</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>8.635000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.519515</td>\n",
       "      <td>13.355000</td>\n",
       "      <td>2.395000</td>\n",
       "      <td>1.320000</td>\n",
       "      <td>72.685000</td>\n",
       "      <td>0.175000</td>\n",
       "      <td>9.065000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.015000</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.521055</td>\n",
       "      <td>14.220000</td>\n",
       "      <td>3.570000</td>\n",
       "      <td>1.902500</td>\n",
       "      <td>73.382500</td>\n",
       "      <td>0.502500</td>\n",
       "      <td>10.090000</td>\n",
       "      <td>0.112500</td>\n",
       "      <td>0.240000</td>\n",
       "      <td>6.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.523000</td>\n",
       "      <td>17.380000</td>\n",
       "      <td>3.780000</td>\n",
       "      <td>2.170000</td>\n",
       "      <td>75.410000</td>\n",
       "      <td>0.760000</td>\n",
       "      <td>12.500000</td>\n",
       "      <td>1.670000</td>\n",
       "      <td>0.510000</td>\n",
       "      <td>7.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              RI         Na         Mg         Al         Si          K  \\\n",
       "count  18.000000  18.000000  18.000000  18.000000  18.000000  18.000000   \n",
       "mean    1.519131  13.763333   1.873889   1.342222  72.820000   0.253889   \n",
       "std     0.002791   1.083491   1.749753   0.581312   0.892004   0.265133   \n",
       "min     1.511150  12.850000   0.000000   0.340000  71.360000   0.000000   \n",
       "25%     1.517413  13.012500   0.000000   0.877500  72.212500   0.000000   \n",
       "50%     1.519515  13.355000   2.395000   1.320000  72.685000   0.175000   \n",
       "75%     1.521055  14.220000   3.570000   1.902500  73.382500   0.502500   \n",
       "max     1.523000  17.380000   3.780000   2.170000  75.410000   0.760000   \n",
       "\n",
       "              Ca         Ba         Fe  Type of glass  \n",
       "count  18.000000  18.000000  18.000000      18.000000  \n",
       "mean    9.460556   0.292778   0.130556       4.000000  \n",
       "std     1.380432   0.617422   0.164798       2.222876  \n",
       "min     6.650000   0.000000   0.000000       1.000000  \n",
       "25%     8.635000   0.000000   0.000000       2.000000  \n",
       "50%     9.065000   0.000000   0.015000       4.000000  \n",
       "75%    10.090000   0.112500   0.240000       6.000000  \n",
       "max    12.500000   1.670000   0.510000       7.000000  "
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "def standardize (df):\n",
    "    for col in df.columns:\n",
    "        if col != \"Type of glass\": # Don't standardize the categories\n",
    "            df[col] = (df[col] - df[col].mean())/df[col].std()\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RI</th>\n",
       "      <th>Na</th>\n",
       "      <th>Mg</th>\n",
       "      <th>Al</th>\n",
       "      <th>Si</th>\n",
       "      <th>K</th>\n",
       "      <th>Ca</th>\n",
       "      <th>Ba</th>\n",
       "      <th>Fe</th>\n",
       "      <th>Type of glass</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.888537</td>\n",
       "      <td>0.338119</td>\n",
       "      <td>1.242977</td>\n",
       "      <td>-0.720654</td>\n",
       "      <td>-1.120263</td>\n",
       "      <td>-0.682898</td>\n",
       "      <td>-0.113060</td>\n",
       "      <td>-0.33849</td>\n",
       "      <td>-0.581932</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.224221</td>\n",
       "      <td>0.657345</td>\n",
       "      <td>0.603903</td>\n",
       "      <td>-0.191863</td>\n",
       "      <td>0.123880</td>\n",
       "      <td>-0.058551</td>\n",
       "      <td>-0.760269</td>\n",
       "      <td>-0.33849</td>\n",
       "      <td>-0.581932</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-0.692234</td>\n",
       "      <td>0.197659</td>\n",
       "      <td>0.568000</td>\n",
       "      <td>0.174223</td>\n",
       "      <td>0.464382</td>\n",
       "      <td>-0.192340</td>\n",
       "      <td>-0.795443</td>\n",
       "      <td>-0.33849</td>\n",
       "      <td>-0.581932</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.207857</td>\n",
       "      <td>-0.210950</td>\n",
       "      <td>0.668529</td>\n",
       "      <td>-0.334230</td>\n",
       "      <td>-0.033275</td>\n",
       "      <td>0.075237</td>\n",
       "      <td>-0.485909</td>\n",
       "      <td>-0.33849</td>\n",
       "      <td>-0.581932</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-0.286405</td>\n",
       "      <td>-0.134335</td>\n",
       "      <td>0.618265</td>\n",
       "      <td>-0.435920</td>\n",
       "      <td>0.582248</td>\n",
       "      <td>0.045506</td>\n",
       "      <td>-0.591432</td>\n",
       "      <td>-0.33849</td>\n",
       "      <td>-0.581932</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         RI        Na        Mg        Al        Si         K        Ca  \\\n",
       "0  0.888537  0.338119  1.242977 -0.720654 -1.120263 -0.682898 -0.113060   \n",
       "1 -0.224221  0.657345  0.603903 -0.191863  0.123880 -0.058551 -0.760269   \n",
       "2 -0.692234  0.197659  0.568000  0.174223  0.464382 -0.192340 -0.795443   \n",
       "3 -0.207857 -0.210950  0.668529 -0.334230 -0.033275  0.075237 -0.485909   \n",
       "4 -0.286405 -0.134335  0.618265 -0.435920  0.582248  0.045506 -0.591432   \n",
       "\n",
       "        Ba        Fe  Type of glass  \n",
       "0 -0.33849 -0.581932              1  \n",
       "1 -0.33849 -0.581932              1  \n",
       "2 -0.33849 -0.581932              1  \n",
       "3 -0.33849 -0.581932              1  \n",
       "4 -0.33849 -0.581932              1  "
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Standardize data\n",
    "train = standardize(train)\n",
    "test = standardize(test)\n",
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "# k = 8 neighbors\n",
    "euclid_model = KNeighborsClassifier(n_neighbors=8, metric=distance.sqeuclidean) # Square Euclidean distance model\n",
    "manhattan_model = KNeighborsClassifier(n_neighbors=8, metric=distance.cityblock) # Manhattan distance model\n",
    "x_train = train.drop([\"Type of glass\"], axis=1)\n",
    "y_train = train[\"Type of glass\"]\n",
    "euclid_model.fit(x_train,y_train) # Train models\n",
    "manhattan_model.fit(x_train, y_train)\n",
    "x_test = test.drop(\"Type of glass\", axis=1) \n",
    "y_test = test[\"Type of glass\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>actual</th>\n",
       "      <th>manhattan</th>\n",
       "      <th>euclid</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   actual  manhattan  euclid\n",
       "0       1          1       1\n",
       "1       1          1       1\n",
       "2       1          1       1\n",
       "3       2          2       2\n",
       "4       2          1       1"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Make predictions\n",
    "manhattan_predictions = manhattan_model.predict(x_test)\n",
    "euclid_predictions = euclid_model.predict(x_test) \n",
    "df = pd.DataFrame({'actual': y_test, 'manhattan': manhattan_predictions, 'euclid': euclid_predictions})\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Manhattan Accuracy: 66.67%\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           1       0.50      1.00      0.67         3\n",
      "           2       0.25      0.33      0.29         3\n",
      "           3       0.00      0.00      0.00         3\n",
      "           5       1.00      0.67      0.80         3\n",
      "           6       1.00      1.00      1.00         3\n",
      "           7       1.00      1.00      1.00         3\n",
      "\n",
      "   micro avg       0.67      0.67      0.67        18\n",
      "   macro avg       0.62      0.67      0.63        18\n",
      "weighted avg       0.62      0.67      0.63        18\n",
      "\n",
      "\n",
      "\n",
      "Square Euclidean Accuracy: 61.11%\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           1       0.60      1.00      0.75         3\n",
      "           2       0.33      0.67      0.44         3\n",
      "           3       0.00      0.00      0.00         3\n",
      "           5       1.00      0.67      0.80         3\n",
      "           6       1.00      0.33      0.50         3\n",
      "           7       0.75      1.00      0.86         3\n",
      "\n",
      "   micro avg       0.61      0.61      0.61        18\n",
      "   macro avg       0.61      0.61      0.56        18\n",
      "weighted avg       0.61      0.61      0.56        18\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Jacob/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.\n",
      "  'precision', 'predicted', average, warn_for)\n"
     ]
    }
   ],
   "source": [
    "# Evaluate performance\n",
    "manhattan_count = len(df.loc[df['manhattan'] == df['actual']])\n",
    "euclid_count = len(df.loc[df['euclid'] == df['actual']])\n",
    "print('Manhattan Accuracy: {}%'.format(round(100*manhattan_count/len(df), 2)))\n",
    "print(classification_report(y_test, manhattan_predictions, target_names=df['actual'].astype(str).unique()))\n",
    "print ('\\n')\n",
    "print('Square Euclidean Accuracy: {}%'.format(round(100*euclid_count/len(df), 2)))\n",
    "print(classification_report(y_test, euclid_predictions, target_names=df['actual'].astype(str).unique()))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}




