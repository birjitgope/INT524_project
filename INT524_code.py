#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn.metrics import silhouette_score


'''
Preparing Data
1> Only columns are extracted which are required
2> Only rows are selected where HRDAName == Bachelor of Technology (Mechanical Engineering)
3> Checking for missing values rows, and droping missing value rows in this case
4> Adding Labelencoders, generating uid for identification, droping processed columns
'''
#1
df = pd.read_csv("C:\\Users\\Birjit\\OneDrive\\Documents\\GitHub\\INT524_project\\DATA.csv",usecols=[0,1,2,3,4,5,6,8,9])
print("Shape of data loaded:",df.shape)
print("Datahead:\n",df.head())

#2
df = df[df['MHRDName'] == 'Bachelor of Technology (Mechanical Engineering)']
print("Rows of MHRDName =Bachelor of Technology (Mechanical Engineering):",df.shape[0])
df = df.drop('MHRDName',axis=1)

#3
print("Data with NaNa value:\n",df.isnull().sum())
df = df.dropna(axis=0)
print("Data after removing NaNa value:\n",df.isnull().sum())
print("Rows after droping NaNa rows:",df.shape[0])

#4
#unique student id
df['student_id'] = df["Termid"].astype(str) + "" + df["Regd No"].astype(str) + "" + df["Course"].astype(str)
#Trimimg unwanted columns
df = df.drop('Termid',axis=1)
df = df.drop('Regd No',axis=1)
df = df.drop('Course',axis=1)
#adding label encoder to grades
unique_grades = len(list(df['Grade'].unique()))
gle = LabelEncoder()
df['grade_value'] = gle.fit_transform(df['Grade'].values)
print('After pre-processing Step 4\n',df.head())

#Clustring Techniques
#Kmeans: Unsupervised learning
#in this we does not require target values
k_df = df.copy()
k_df=k_df.drop('student_id',axis=1)
k_df=k_df.drop('grade_value',axis=1)
k_df=k_df.drop('Grade',axis=1)
print("KMeans data input\n",k_df.head())

c = ['red','blue','green','yellow','orange','black','indigo','chocolate','lime','deeppink']
for i in range(2,unique_grades,1):
    km=KMeans(n_clusters=i)
    y_km = km.fit_predict(k_df)
    k_df_arr = np.array(k_df)
    for j in range(i):
    	plt.scatter(k_df_arr[y_km==j,0],k_df_arr[y_km==j,1],marker='*',c=c[j])
    plt.show()

print("Distance score of all data from the center",km.inertia_)
distortion=[]
for i in range(2,unique_grades,1):
    km=KMeans(n_clusters=i)
    Y_km=km.fit_predict(k_df)
    distortion.append(km.inertia_)
    print('Silhouette_score with cluster =',i,silhouette_score(k_df,Y_km))
print('Distortion-->',distortion)
a=np.arange(2,unique_grades)
plt.plot(a,distortion)
plt.grid()
plt.show()

#DBSCAN CLUSTERING
db=DBSCAN(eps=10,min_samples=10,metric='euclidean').fit(k_df)
labels = db.labels_
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(k_df) 
# Normalizing the data so that  
# the data approximately follows a Gaussian distribution 
X_normalized = normalize(X_scaled)  
X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components = 4) 
X_principal = pca.fit_transform(X_normalized) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2','P3','P4'] 
print(X_principal.head()) 


colours = {} 
colours[0] = 'r'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k' 
cvec = [colours[label] for label in labels] 
# For the construction of the legend of the plot 
r = plt.scatter(X_principal['P1'], X_principal['P2'], color ='r'); 
g = plt.scatter(X_principal['P1'], X_principal['P2'], color ='g'); 
b = plt.scatter(X_principal['P1'], X_principal['P2'], color ='b'); 
k = plt.scatter(X_principal['P1'], X_principal['P2'], color ='k'); 

plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec)
plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1')) 
plt.show() 

#Agglomerative Clustering
ac=AgglomerativeClustering(n_clusters=3,affinity='euclidean', linkage='complete')
labels=ac.fit_predict(k_df)
print('Cluster labels:%s'%labels)
import matplotlib.pyplot as plt
plt.scatter(k_df.iloc[labels == 0,0], k_df.iloc[labels == 0,1])
plt.scatter(k_df.iloc[labels == 1,0], k_df.iloc[labels == 1,1])
plt.scatter(k_df.iloc[labels == 0,0], k_df.iloc[labels == 0,1])
plt.scatter(k_df.iloc[labels == 1,0], k_df.iloc[labels == 1,1])
plt.scatter(k_df.iloc[labels == 0,0], k_df.iloc[labels == 0,1])
plt.title('Agglomerative Clustering')
plt.plot()
plt.show()
