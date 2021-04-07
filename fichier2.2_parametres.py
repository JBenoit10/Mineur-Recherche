# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:48:29 2021

@author: jeanb
"""

import pandas as pd
import xlrd
from  matplotlib import pyplot as plt
import math
#import mglearn
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #to plot in 3D
import mpmath as mp




#------------------------------------------------------------------------------
                            #LECTURE DU FICHIER EXCEL

print("PARTIE 1 : Resprésentation graphique des données")
#Lecture du fichier excel
document = xlrd.open_workbook("Data_table_3.xlsx")
fichier = "Data_table_3.xlsx"
sheet = "Sheet1"
df = pd.read_excel(io=fichier, sheet_name=sheet)

#------------------------------------------------------------------------------
                            #CREATION D'UNE BDD OU S'ENTRAINER

ligne_debut = 1
ligne_fin = 6000
ligne = ligne_fin - ligne_debut

df_new = df[['id', 'referenceyear', 'tot_assets', 'capital', 'capital_surplus',
             'paidincapital', 'tot_deposits', 'cash', 'comm_portfolio',
             'securities', 'tot_credit']]

#df_data = df_new.head(ligne)
df_data = df_new[ligne_debut:ligne_fin]

print('------------------------------------')
print("Données à utiliser : \n{}".format(df_data))

df_new = df[['f1']]

#df_resultat = df_new.head(ligne)
df_resultat = df_new[ligne_debut:ligne_fin]

print('------------------------------------')
print("Faillit ou non des banques : \n{}".format(df_resultat))


#------------------------------------------------------------------------------
                            #CREATION DES DONNEES DE TRAINING ET DE TEST

#Choix du nombre de lignes qui vont etre sélectionnée -------------------------
calcul = 0.75*ligne
resultat = int(round(calcul))
#resultat = 75
fin = ligne_fin

#Initialisation des paramètres ------------------------------------------------
capital_ratio_1 = (df_data['capital'])/df_data['tot_assets']
df_data['capital_ratio'] = capital_ratio_1

liquid_ratio_1 = df_data['cash']/df_data['tot_assets']
df_data['liquidity_ratio'] = liquid_ratio_1

#Replace the nan values by 0
df_data.fillna(value=0, inplace = True)
df_resultat.fillna(value=0, inplace = True)



#Training data ----------------------------------------------------------------
Training =  np.array([df_data['referenceyear'][0:resultat],
                      df_data['capital_ratio'][0:resultat],
                      df_data['liquidity_ratio'][0:resultat]])

#plt.plot(Training[0], Training[1], label='Total assets')
plt.scatter(Training[0], Training[1], label='Capital Ratio')
plt.scatter(Training[0], Training[2], label='liquidity ratio')
#plt.plot(Training[0], Training[7], label='real_capital_ratio')
#plt.plot(Training[0], Training[8], label='balence commerciale')
plt.legend()

#Test data --------------------------------------------------------------------
Test = np.array([df_data['referenceyear'][resultat:fin],
                      df_data['capital_ratio'][resultat:fin],
                      df_data['liquidity_ratio'][resultat:fin]])

#Training matrice -------------------------------------------------------------
Xtrain = Training
Xtrain = Xtrain.T
Ytrain = np.array([df_resultat['f1'][0:resultat]])
Ytrain = Ytrain.T

#Test matrice -----------------------------------------------------------------
Xtest = Test
Xtest = Xtest.T
Ytest = np.array([df_resultat['f1'][resultat:fin]])
Ytest = Ytest.T

#Ajout de la colonne de 1 à la matrice Xtrain ---------------------------------
p = np.ones([len(Xtrain),1])
Xtrain = np.append(p, Xtrain, axis = 1)

#Ajout de la colonne de 1 à la matrice Xtest---------------------------------
p = np.ones([len(Xtest),1])
Xtest = np.append(p, Xtest, axis = 1)


#------------------------------------------------------------------------------
#--------------------------------DEFINITION FONCTIONS CFS----------------------

#Fonction hypothèse
def Hypothese(X, Theta):
    return X.dot(Theta)

#Fonction CFS
def CFS(X, Y):
    p1 = np.dot(X.T,X)
    p2 = np.dot(X.T, Y)
    #Theta = np.linalg.pinv(p1).dot(p2)
    Theta =  np.linalg.pinv(p1)
    Theta = Theta.dot(p2)
    #Theta = Theta[~np.isnan(Theta)]
    return Theta

#Fonction Erreur
def Erreur(x,y,theta):
    ypred = np.dot(x,theta)
    I = len(y)
    # boucle for pour parcourir les matrices
    for i in range (I):
        erreur = (1/I) * np.sum(np.square(ypred-y))  #MSE (mean square error)
    return erreur

#------------------------------------------------------------------------------
#------------------------------PARAMETRES OPTIMAUX-----------------------------

Theta_F = CFS(Xtrain, Ytrain)


print(Theta_F)

#Prédiction des données de training--------------------------------------------
Yprediction = Hypothese(Xtrain, Theta_F)
error = Erreur(Xtrain, Ytrain, Theta_F)
print("Erreur : ", error)

#Affichage des données---------------------------------------------------------
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.scatter3D(Xtrain[:,2], Xtrain[:,3], Ytrain, color= 'green')
ax1.scatter3D(Xtrain[:,2], Xtrain[:,3], Yprediction, color= 'red')
ax1.set_xlabel('Capital Ratio', fontweight = 'bold')
ax1.set_ylabel('Liquidity ratio', fontweight = 'bold')
ax1.set_zlabel('Bankruptcy probability', fontweight = 'bold')
plt.show()


#☻Prediction des données de test-----------------------------------------------
#Yprediction2 = Hypothese(Xtest, Theta_F)
#error2 = Erreur(Xtest, Ytest, Theta_F)
#print("Erreur : ", error2)
#------------------------------------------------------------------------------

#Classe de regression logistique-----------------------------------------------
    #Appel permettant la résolution direct pour tout autre type de fichier

class LogisticRegression:
    
    #On instancie les paramètres propres à la régression
        #lr = alpha = précision
        #num_iter = nomre d'iterations !!ATENTION : Possible overfitting!!
        #fit intercept : ici il faut calculer le point d'interception
        #verbose : suite au traitement de beaucoup de données et de nombreuses itérations
   def __init__(self, lr = 0.01, num_iter = 100000, fit_intercept = True, verbose = False):
      self.lr = lr
      self.num_iter = num_iter
      self.fit_intercept = fit_intercept
      self.verbose = verbose
      
      #Ajout de la colonne de 1 à Xpar concatenate
   def __add_intercept(self, X):
      intercept = np.ones((X.shape[0], 1))
      return np.concatenate((intercept, X), axis=1)
  
    #Fonction Sigmoid
   def Sigmoid(self, z):
      return 1 / (1 + np.exp(-z))
    
    #Fonction erreur
   def Erreur(self, h, y):
      return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
  
    #Normalisation des données
   def fit(self, X, y):
      if self.fit_intercept:
        X = self.__add_intercept(X)
        self.theta = np.zeros(X.shape[1])
        for i in range(self.num_iter):
           z = np.dot(X, self.theta)
           h = self.Sigmoid(z)
           gradient = np.dot(X.T, (h - y)) / y.size #gradient : permet de faire des 'boucles' afin de trouver la valeur de Theta pour laquelle la fonction de coût est égale à 0
           self.theta -= self.lr * gradient
           
           z = np.dot(X, self.theta)
           h = self.Sigmoid(z)
           loss = self.Erreur(h, y)
           
           if(self.verbose ==True and i % 10000 == 0):
              print(f'loss: {loss} \t')
      
    #Retourne la valeur exacte de la valeur prédite
   def predict_prob(self, X):
       if self.fit_intercept:
           X = self.__add_intercept(X)
           return self.Sigmoid(np.dot(X, self.theta))
      
        #Arrondie la prédiction
   def predict(self, X):
      return self.predict_prob(X).round()

#Variables --------------------------------------------------------------------
model = LogisticRegression()
model = model.fit(Xtrain, Ytrain)
preds = model.predict(Xtrain)


