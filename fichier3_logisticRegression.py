# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:25:11 2021

@author: jeanb
"""

import xlrd
import math
import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt
from scipy import misc

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

df_new = df[['id', 'referenceyear', 'tot_assets', 'capital', 'capital_surplus', 
             'paidincapital', 'tot_deposits', 'cash', 'comm_portfolio', 
             'securities', 'tot_credit']]
df_data = df_new.head(20)
print('------------------------------------') 
print("Données à utiliser : \n{}".format(df_data))

df_new = df[['f1']]
df_resultat = df_new.head(20)
print('------------------------------------')
print("Faillit ou non des banques : \n{}".format(df_resultat))

#------------------------------------------------------------------------------
                            #CREATION DES DONNEES DE TRAINING ET DE TEST 

#Choix du nombre de lignes qui vont etre sélectionnée -------------------------
calcul = 0.75*200
resultat = int(round(calcul))
#resultat = 75
fin = 200

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

#plt.plot(Training[0], Training[1], label='Total assets')
plt.plot(Training[0], Training[2], label='Capital Ratio')
plt.plot(Training[0], Training[3], label='liquidité')
plt.plot(Training[0], Training[4], label='exposition aux risques')
plt.plot(Training[0], Training[5], label='risques liés aux dépots')
plt.plot(Training[0], Training[6], label='sécurities')
#plt.plot(Training[0], Training[7], label='real_capital_ratio')
#plt.plot(Training[0], Training[8], label='balence commerciale')
plt.legend()

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
                                    #FONCTIONS 
initial_theta = [0,0]
alpha = 0.1
iterations = 1000
#Sigmoid ----------------------------------------------------------------------
def Sigmoid(z):
	result = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return result 

#Hypothèse --------------------------------------------------------------------
def Hypothese(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)

#Fonction de Coût -------------------------------------------------------------
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

#Gradient descent --------------------------------------------------------------
def gradientDescent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in range(len(theta)): 
		CFDerivative = misc.derivative(costFunction, 2.0) #Dérivée de la fonction de coût
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta

#Fonction logistique ----------------------------------------------------------
def Logistic_Regression(X,Y,alpha,theta,num_iters):
	m = len(Y)
	for x in range(num_iters):
		new_theta = gradientDescent(X,Y,theta,m,alpha)
		theta = new_theta
		if x % 100 == 0:
			costFunction(X,Y,theta,m)
			print ('theta ', theta)	
			print ('cost is ', costFunction(X,Y,theta,m))

