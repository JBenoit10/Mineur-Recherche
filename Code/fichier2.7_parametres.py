# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:03:24 2021

@author: jeanb

"""
import pandas as pd
import xlrd
from  matplotlib import pyplot as plt
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

ligne_debut = 1
ligne_fin = 6000
ligne = ligne_fin - ligne_debut

df_new = df[['id', 'referenceyear', 'tot_assets', 'capital', 'on_time_deposits', 'capital_surplus',
             'paidincapital', 'tot_deposits', 'cash', 'tot_credit', 'on_demand_deposits']]

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
capital_ratio_2 = (df_data['capital'] + df_data['capital_surplus'])/df_data['tot_assets']
df_data['capital_ratio'] = capital_ratio_2


deposits_risks_1 = (df_data['on_demand_deposits'] + df_data['on_time_deposits'])/df_data['tot_assets']
df_data['deposits_risks'] = deposits_risks_1

liquid_ratio_2 = df_data['cash']/df_data['tot_assets']
df_data['liquidity_ratio'] = liquid_ratio_2


#Replace the nan values by 0
df_data.fillna(value=0, inplace = True)
df_resultat.fillna(value=0, inplace = True)



#Training data ----------------------------------------------------------------
Training =  np.array([df_data['referenceyear'][0:resultat],
                      df_data['capital_ratio'][0:resultat],                                            
                      df_data['deposits_risks'][0:resultat],
                      df_data['liquidity_ratio'][0:resultat]])


plt.scatter(Training[0], Training[1], label='Capital Ratio')
plt.scatter(Training[0], Training[2], label='risques liés aux dépots')
plt.scatter(Training[0], Training[3], label='Liquidity ratio')
plt.legend()

#Test data --------------------------------------------------------------------
Test =  np.array([df_data['referenceyear'][resultat:fin],
                      df_data['capital_ratio'][resultat:fin],                                            
                      df_data['deposits_risks'][resultat:fin],
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
plt.scatter(Xtrain[:,2], Ytrain, color= 'green')
plt.scatter(Xtrain[:,2], Yprediction, color= 'pink')
plt.scatter(Xtrain[:,3], Ytrain, color= 'blue')
plt.scatter(Xtrain[:,3], Yprediction, color= 'red')
plt.scatter(Xtrain[:,4], Ytrain, color= 'black')
plt.scatter(Xtrain[:,4], Yprediction, color= 'red')
plt.xlabel('Result of the equations')
plt.ylabel('Banckruptcy risk')
plt.legend()
plt.show()

'''
plt.scatter(Xtrain[:,2], Ytrain, color= 'green')
plt.scatter(Xtrain[:,2], Yprediction, color= 'red')
plt.xlabel('Total value in dollars')
plt.ylabel('Banckruptcy risk')
plt.show()
'''

'''
plt.scatter(Xtrain[:,1], Ytrain, color= 'green')
plt.scatter(Xtrain[:,1], Yprediction, color= 'red')
plt.xlabel('Year')
plt.ylabel('Banckruptcy risk')
plt.show()
'''

#Variables --------------------------------------------------------------------
initial_theta = [0,0]
alpha = 0.1
iterations = 1000

