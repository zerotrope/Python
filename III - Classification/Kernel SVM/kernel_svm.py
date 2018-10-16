# CLASSIFICATION
# Kernel SVM

# I - Data Preprocessing

# 1. Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importer le dataset
dataset = pd.read_csv('Social_Network_ADs.csv')
X = dataset.iloc[:, [2, 3]].values #On selectionne uniquement les ages et salaires
Y = dataset.iloc[:, -1].values #-1 désigne la dernière colonne du dataset

# 3. Gérer les données manquantes
#Il n'y en a pas

# 4. Gérer les variables catégoriques
#Il n'y en a pas

# 5. Diviser le dataset entre le Training set (généralement 80% du dataset, ici 75% des 400 observations) 
# et le Test set (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# 6. Feature Scaling
# NOTES:
#Consiste à mettre toutes nos variables sous la même échelle en soustrayant la valeur observée à la moyenne
#et à diviser par l'ecart-type
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# II - Construction du modèle

# 7. Importation de l'objet du modèle
from sklearn.svm import SVC
classifier = SVC(random_state = 0) #par défaut c'est le "rbf" qui est sélectionné càd le Gaussien

# 8. Lier l'objet au Training Set par la méthode fit
classifier.fit(X_train, Y_train)

# 9. Nouvelles prédictions
y_pred = classifier.predict(X_test)

# 10. Construire la matrice de confusion pour compter le nombre de prédictions correctes/incorrectes
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

# 11. Visualiser les résultats
# on pourrait imaginer que chaque pixel sur le graph pourrait être un utilisateur. A partir de là on peut
# utiliser notre modèle pour prédire le résultat d'un pixel utilisateur (acheteur ou non), l'idée étant 
# qu'un positif sera coloré en vert et en rouge inversement. On construit donc un 'grid' avec des intervalles
# en x et en y et on les colorie. 
# Ensuite grâce à 'contourf qui permet de tracer la droite.
from matplotlib.colors import ListedColormap 
X_set, y_set = X_train, Y_train # à remplacer par X_test, Y_test pour comparaison
# On construit le grid
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# On trace la frontière entre les deux régions
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# On place les points d'observations sur le graph
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
# On rajoute les labels
plt.title('Résultats du Training set')
plt.xlabel('Age')
plt.ylabel('Salaire Estimé')
plt.legend()
plt.show()
# affiche une "courbe limite de prédiction" ou "prediction boundary" non linéaire qui classe bien mieux
# les utilisateurs que les modèles linéaires.
#Classification encore meilleur avec les Test Sets qu'avec les Training Sets.