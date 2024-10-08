import pandas as pd
import numpy as np
df=pd.read_csv("data.csv",delimiter=";")
pd.set_option('display.max_columns', None)  #it displays all the columns None =>par default
df

"""*Visualisation des donnees*"""

df.describe()

#columns visualisation
col=df.columns.values  #Acces aux colonnes du DATAFRAME
print(col,"\n\n\n")

l=[]
l="[{}]".format(",".join(col))
print(l)

"""#PREPARATION DE DONNEES - DATA PRE-PROCESSING

*1-Verifier si il existe des valeurs manquantes*
"""

df.isnull() #il est difficile de reconnaitre s il existe des valeurs manquantes

#AUTREMENT
df.isnull().sum() #pas de valeur manquante dans notre DATASET

"""*2-Exploration des donnees (identifiez les caractéristiques attributs pertinentes)*"""

df.corr(numeric_only=True).round(2) #retourne une matrice de relation lineaire entre les colonnes

"""#Heatmap"""

#visualisation de la matrice corr
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(30,30))         #40 pouces(hauteur*Largeur)
sns.heatmap(df.corr(numeric_only=True).round(2),annot=True)   #display the values

"""*Identifier les attributs ayant une relation lineaire*"""

def corr_features(dataset):
  corr_mat=df.corr(numeric_only=True)     #matrice
  s=set()
  for i in range(len(corr_mat.columns)):  # !!applicable juste dans le matrice de correlation!!(la liste des noms de colonnes)
    for j in range(i+1,len(corr_mat.columns)):
      if  abs(corr_mat.iloc[i,j])>=0.8:
          col=corr_mat.columns[i]
          s.add(col)
  return s
corr_features(df)
#le principe de ce code

#Se debarasser des elements ayant une realtion lineaire avec autres attributs

df.drop(['hot', 'num_compromised'],axis=1)

"""SPLITIING THE DATASET -DIVISER LA BDD

**1-Diviser les donnes en variable independante X et la variable cible Y**
"""

import numpy as np
y=df["class"]
x=df.drop("class",axis=1)
print(y)
y=np.array(y)

"""**2-Diviser le x et y  bdd en training et testing datasets**"""

#Importons la bibliotheque qui concoit le splitting
from sklearn import model_selection

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=21,shuffle=False)  #random state est fixe a une valeur afin de garder une division fixe
print(x_train.shape)
print(x_test.shape)
print(y_test)
print(y_train)


#lors de la division il faut faire attention que quelques valeurs peuvent exister
#dans des datasets alors elle ne le seront pas dans l autres vue 30% pout le test
#et 70% out le training (verifions la apres encodage)

"""ENCODAGE DES ATTRIBUTS CATEGORIQUES -ENCODING THE CATEGORICAL DATA"""

catego_data=x.select_dtypes(include=["object"])
print(catego_data)

dummies= pd.get_dummies(x_train,columns=["protocol_type","service","flag"])                                        #si on donne seulemt le dataframe comme parametre toute la dataset is going to be encoded
#encoded x_train
dummies1=pd.get_dummies(x_test,columns=["protocol_type","service","flag"])
#encoded x_test

dummies

print(dummies.shape)  #training
print(dummies1.shape)   #testing
dummies1
#le nmbre de colonnes ecodees different

"""#MODELE

#Decision Tree
"""

from sklearn import tree  #modele
from sklearn import metrics #Evaluation du modele

tree_cl=tree.DecisionTreeClassifier()  #crearing an instance of the decision tree
tree_cl1=tree_cl.fit(dummies,y_train) #trained on the model

dummies1=dummies1.reindex(columns=dummies.columns,fill_value=0)
#dummies1==x_test encoded

y_pred=tree_cl1.predict_proba(dummies1) #returns a 2D probability array for each instance
y_pred1=tree_cl1.predict(dummies1) #FOR THE CONFUSION MATRIX
print(y_pred1)

classes = tree_cl.classes_  #Y-pred has the first column as anomaly and 2nd as normal
print(classes)

print(y_test)
print(y_pred)

#remarquons que y_test est sous forme de serie => a transformer en array
import numpy as np

y_test=np.array(y_test)
print(y_test)

#ENCODAGE DE y_test et y_pred

y_test1=np.where(y_test=="normal",1,0)

y_pred1=np.where(y_pred1=="normal",1,0)

print(y_pred1)

"""#EVALUATION DU MODELE

1-Matrice De Confusion
"""

l=[1,0]  #positive(normal)  - negative(anomaly)#les deux sont encodes

#calcul de matrice de confusion
conf_matrix=metrics.confusion_matrix(y_test1,y_pred1,labels=l)
print(conf_matrix ," \n \n")

################
acc0=metrics.accuracy_score(y_test1,y_pred1)
print("--La precision du Logistic Regression model est : ",acc0,"\n \n")
#################




#Tracage de la matrice de confusion
plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix,annot=True,fmt="d") #sous format decimale
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(2), ['Normal', 'Anomaly'])
plt.yticks(np.arange(2), ['Normal', 'Anomaly'])

"""#RANDOM FOREST (Cross-validation)"""

from sklearn import ensemble

#choisissons le nombre d arbres optimal en utilisant la validation croiee(n_estimators)

l=[10,30,50,70] #contient une liste de nombre d arbres a tester
d={}
for i in l:
  modele=ensemble.RandomForestClassifier(n_estimators=i,random_state=0) #random_state ==seed
  scores =model_selection.cross_val_score(modele, dummies, y_train, cv=5, scoring='accuracy') #list of scores for each cv(pli)
  d[i]=np.mean(scores)

print(d)
val_max=max(d,key=d.get)

print(val_max)

#construction du modele

forest =ensemble.RandomForestClassifier(n_estimators=70,random_state=0)
train_for=forest.fit(dummies,y_train)
y_predFor=train_for.predict(dummies1)
print(y_predFor)

y_predFor1=np.where(y_predFor=="normal",1,0) # encoding the array

l=[1,0]

conf_matrix1=metrics.confusion_matrix(y_test1,y_predFor1,labels=l)
print(conf_matrix1,"\n \n")

################
acc1=metrics.accuracy_score(y_test1,y_predFor1)
print("--La precision du Logistic Regression model est : ",acc1,"\n \n")
#################


#Tracage de la matrice de confusion
plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix1,annot=True,fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(2), ['Normal', 'Anomaly'])
plt.yticks(np.arange(2), ['Normal', 'Anomaly'])

"""#KNN - K NEAREST Neighbors"""

from sklearn import neighbors

#Choisir le K le plus approprie

liste=[5,10,20,40,50,100]
l=[]
for i in liste:
  model1=neighbors.KNeighborsClassifier(n_neighbors=i)
  score=model_selection.cross_val_score(model1,dummies,y_train, cv=5, scoring='accuracy')
  l.append(score.mean())
print(l)
a=max(l)
for j in range(len(l)):
  if(l[j]==a):
    print(j)

#L indice de 0 de la liste  correspond a un nombre d instances egale a 5

knn=neighbors.KNeighborsClassifier(n_neighbors=5)
knn_train=knn.fit(dummies,y_train)
y_predKnn=knn_train.predict(dummies1)
print(y_predKnn)
y_predKnn1=np.where(y_predKnn=="normal",1,0)
y_predKnn1

l=[1,0]

conf_matrix2=metrics.confusion_matrix(y_test1,y_predKnn1,labels=l)
print(conf_matrix2,"\n \n")

################
acc2=metrics.accuracy_score(y_test1,y_predKnn1)
print("--La precision du Logistic Regression model est : ",acc2,"\n \n")
#################


#Tracage de la matrice de confusion
plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix2,annot=True,fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(2), ['Normal', 'Anomaly'])
plt.yticks(np.arange(2), ['Normal', 'Anomaly'])

"""#Logistic Regression"""

from sklearn import linear_model

log_Reg=linear_model.LogisticRegression()
logR_train=log_Reg.fit(dummies,y_train)
y_predLogR=logR_train.predict(dummies1)
print(y_predLogR)

#Encoding y_predLogR
y_predLogR=np.where(y_predLogR=="normal",1,0)

l=[1,0]     #donner l ordre aux etiquettes(normal(1)& anomaly(0)) dans la matrice de confusion

conf_matrix3=metrics.confusion_matrix(y_test1,y_predLogR,labels=l)
print(conf_matrix3,"\n \n")

################
acc3=metrics.accuracy_score(y_test1,y_predLogR)
print("--La precision du Logistic Regression model est : ",acc3,"\n \n")
#################

#Tracage de la matrice de confusion
plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix3,annot=True,fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(2), ['Normal', 'Anomaly'])
plt.yticks(np.arange(2), ['Normal', 'Anomaly'])

"""#SUPPORT Vector Machine"""

from sklearn import svm

svm_model=svm.LinearSVC(max_iter=5000)
svm_train=svm_model.fit(dummies,y_train)
y_predSvm=svm_train.predict(dummies1)
print(y_predSvm)

y_predSvm1=np.where(y_predSvm=="narmal",1,0)
print(y_predSvm1)

acc4=metrics.accuracy_score(y_test1,y_predSvm1)
print("La precision du SVM model est : ",acc4)

l=[1,0]     #donner l ordre aux etiquettes(normal(1)& anomaly(0)) dans la matrice de confusion

conf_matrix4=metrics.confusion_matrix(y_test1,y_predSvm1,labels=l)
print(conf_matrix4,"\n \n")

################
acc4=metrics.accuracy_score(y_test1,y_predSvm1)
print("--La precision du SVM model est : ",acc4,"\n \n")
#################

#Tracage de la matrice de confusion
plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix4,annot=True,fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(2), ['Normal', 'Anomaly'])
plt.yticks(np.arange(2), ['Normal', 'Anomaly'])
