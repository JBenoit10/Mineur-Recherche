# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:27:35 2021

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

ligne_debut = 1500
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
capital_ratio = (df_data['capital'] + df_data['capital_surplus'])/df_data['tot_assets']
df_data['capital_ratio'] = capital_ratio

liquidite = df_data['cash']/df_data['tot_assets']
df_data['liquidité'] = liquidite

expo_risque = df_data['tot_credit']/df_data['tot_assets']
df_data['expo_risque'] = expo_risque

risque_depots = df_data['tot_deposits']/df_data['tot_assets']
df_data['risque_depots'] = risque_depots

securite = df_data['securities']/df_data['tot_assets']
df_data['securite'] = securite

real_capital_ratio = (df_data['paidincapital'] + df_data['capital_surplus'])/df_data['tot_assets']
df_data['real_capital_ratio'] = real_capital_ratio

commercial = (df_data['comm_portfolio'] + df_data['cash'])/df_data['tot_assets']
df_data['commercial'] = commercial

#Replace the nan values by 0
df_data.fillna(value=0, inplace = True)
df_resultat.fillna(value=0, inplace = True)



#Training data ----------------------------------------------------------------
Training =  np.array([df_data['referenceyear'][0:resultat],
                      df_data['tot_assets'][0:resultat],
                      df_data['capital_ratio'][0:resultat],
                      df_data['liquidité'][0:resultat],
                      df_data['expo_risque'][0:resultat],
                      df_data['risque_depots'][0:resultat],
                      df_data['securite'][0:resultat],
                      df_data['real_capital_ratio'][0:resultat],
                      df_data['commercial'][0:resultat]])

'''
#plt.plot(Training[0], Training[1], label='Total assets')
plt.scatter(Training[0], Training[2], label='Capital Ratio')
plt.scatter(Training[0], Training[3], label='liquidité')
plt.scatter(Training[0], Training[4], label='exposition aux risques')
plt.scatter(Training[0], Training[5], label='risques liés aux dépots')
plt.scatter(Training[0], Training[6], label='sécurities')
#plt.plot(Training[0], Training[7], label='real_capital_ratio')
#plt.plot(Training[0], Training[8], label='balence commerciale')
plt.legend()
'''

#Test data --------------------------------------------------------------------
Test = np.array([df_data['referenceyear'][resultat:fin],
                      df_data['tot_assets'][resultat: fin],
                      df_data['capital_ratio'][resultat:fin],
                      df_data['liquidité'][resultat:fin],
                      df_data['expo_risque'][resultat:fin],
                      df_data['risque_depots'][resultat:fin],
                      df_data['securite'][resultat:fin],
                      df_data['real_capital_ratio'][resultat:fin],
                      df_data['commercial'][resultat:fin]])

#Training matrice -------------------------------------------------------------
Xtrain = Training
Xtrain = Xtrain.T
Ytrain = np.array([df_resultat['f1'][1:resultat+1]])
Ytrain = Ytrain.T

#Test matrice -----------------------------------------------------------------
Xtest = Test
Xtest = Xtest.T
Ytest = np.array([df_resultat['f1'][resultat+1:fin+1]])
Ytest = Ytest.T

#Ajout de la colonne de 1 à la matrice Xtrain ---------------------------------
p = np.ones([len(Xtrain),1])
Xtrain = np.append(p, Xtrain, axis = 1)

#Ajout de la colonne de 1 à la matrice Xtest---------------------------------
p = np.ones([len(Xtest),1])
Xtest = np.append(p, Xtest, axis = 1)


#------------------------------------------------------------------------------
#--------------------------------DEFINITION FONCTIONS CFS----------------------

def Sigmoid(z):
	result = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return result 

#Hypothèse --------------------------------------------------------------------
def Hypothese(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)


def costFunction(X,Y,theta,m):
	sumOfErrors = 0
	for i in range(m):
		xi = X[i]
		hi = Hypothese(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-hi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	print ('cost is ', J )
	return J

def costFunctionDerivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypothese(theta,X[i])
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

def gradientDescent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in range(len(theta)):
		CFDerivative = costFunctionDerivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta

#Fonction Erreur
def Erreur(x,y,theta):
    ypred = np.dot(x,theta)
    I = len(y)
    # boucle for pour parcourir les matrices
    for i in range (I):
        erreur = (1/I) * np.sum(np.square(ypred-y))  #MSE (mean square error)
    return erreur

def CFS(X, Y):
    p1 = np.dot(X.T,X)
    p2 = np.dot(X.T, Y)
    #Theta = np.linalg.pinv(p1).dot(p2)
    Theta =  np.linalg.pinv(p1)
    Theta = Theta.dot(p2)
    #Theta = Theta[~np.isnan(Theta)]
    return Theta
#------------------------------------------------------------------------------
#------------------------------PARAMETRES OPTIMAUX-----------------------------

initial_theta = [0,0]
alpha = 0.1
iterations = 6000

#ori_data.dropna(inplace=True)
Theta_F = CFS(Xtrain, Ytrain)
#Theta_F = Theta_F[~np.isnan(Theta_F)]
#x = x[~numpy.isnan(x)]


print(Theta_F)

#Prédiction des données de training--------------------------------------------
Yprediction = Hypothese(Xtrain, Theta_F)
error = Erreur(Xtrain, Ytrain, Theta_F)
print("Erreur : ", error)

#Affichage des données---------------------------------------------------------
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.scatter3D(Xtrain[:,1], Xtrain[:,2], Ytrain, color= 'green')
ax1.scatter3D(Xtrain[:,1], Xtrain[:,2], Yprediction, color= 'red')
ax1.set_xlabel('x1', fontweight = 'bold')
ax1.set_ylabel('x2', fontweight = 'bold')
ax1.set_zlabel('y', fontweight = 'bold')
plt.show()

plt.scatter(Xtrain[:,2], Ytrain, color= 'green')
plt.scatter(Xtrain[:,2], Yprediction, color= 'red')
plt.xlabel('Total value in dollars')
plt.ylabel('Banckruptcy risk')
plt.show()

#☻Prediction des données de test-----------------------------------------------
#Yprediction2 = Hypothese(Xtest, Theta_F)
#error2 = Erreur(Xtest, Ytest, Theta_F)
#print("Erreur : ", error2)
#------------------------------------------------------------------------------

#Plot Data---------------------------------------------------------------------
'''fig2 = plt.figure()
ax1 = fig2.gca(projection='3d')
ax1.scatter3D(Xtest[:,1], Xtest[:,2], Ytest, color= 'green')
ax1.scatter3D(Xtest[:,1], Xtest[:,2], Yprediction2, color= 'red')
ax1.set_xlabel('x1', fontweight = 'bold')
ax1.set_ylabel('x2', fontweight = 'bold')
ax1.set_zlabel('y', fontweight = 'bold')
plt.show()
'''

#Variables --------------------------------------------------------------------
initial_theta = [0,0]
alpha = 0.1
iterations = 1000

