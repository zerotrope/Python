# CLUSTERING - apprentissage non supervise
# K-Means

# 1. Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importer le dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values #Equivlent a 'Cells(rows, columns)' avec ':' = all
# Il n'y a pas de Y (variable dependante) c'est le propre du Clustering

# En visualisant le dataset la colonne "spending score" prepare auparavant est un score indiquant la
# proportion du client du supermarche a†depenser de 0 a 100. Le but va etre de creer des cluster afin
# d'identifier des groupes, segments de clients organises de maniere logique.

# 3. Gerer les donnees manquantes
#Il n'y en a pas

# 4. Gerer les variables categoriques
#Il n'y en a pas

# 5. Diviser le dataset entre le Training set (80% du dataset) et le Test set (20%)
# Pas necessaire de diviser le dataset : UNSUPERVISED

# 6. Feature Scaling
# Pas necessaire, les donnees du dataset etant deja†sur une echelle comparable de 0 a 100
 
# 7. Utiliser la methode Elbow pour toruver le nombre optimal de clusters
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11): #la borne superieure d'un range de Python est exclue
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11), WCSS) #On est hors de la courbe
plt.title("La m√©thode Elbow")
plt.xlabel("Nombre de clusters")
plt.ylabel("WCSS")
plt.show() #Nombre optimal de clusters obtenu par lecture graphique a 5 "au niveau du coude"

# 8. Construction du modele
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = "k-means++", random_state = 0)
y_kmeans = kmeans.fit_predict(X) #Nous avons creer notre variable dependante 

# 9. Visualiser les resultats
plt.scatter(X[y_kmeans == 1, 0],X[y_kmeans == 1, 1], c = 'red', label = 'Cluster 1' ) #Je prends ainsi tous 
# les points d'observation de mon dataset qui appartiennent au cluster numero 1
plt.scatter(X[y_kmeans == 2, 0],X[y_kmeans == 2, 1], c = 'blue', label = 'Cluster 2' )
plt.scatter(X[y_kmeans == 3, 0],X[y_kmeans == 3, 1], c = 'green', label = 'Cluster 3' )
plt.scatter(X[y_kmeans == 4, 0],X[y_kmeans == 4, 1], c = 'cyan', label = 'Cluster 4' )
plt.scatter(X[y_kmeans == 0, 0],X[y_kmeans == 0, 1], c = 'magenta', label = 'Cluster 5' )
plt.title("Cluster de clients")
plt.xlabel("Salaire annuel")
plt.ylabel("Spending score")
plt.legend()
plt.show()