#Problema: alteração de pH em agua
#Identificação: através do sensor de pH, onde irá fornecer o pH atual do aquario
#Inputs: 
# {
#  "peixe": ["pimelodus pictus", "Rasbora Harlequim",  "Tetra Cosmops"]
#  "ph_atual": 7.0 
# }
#Output: 
# {
#   "status": "necessaria a limpeza e troca de agua, pois o ph está alterado"
#   "ph_essential": 6.5
# }
# modelos de treinos: 
# {
#   "familia": "pimelodus pictus"
#   "ph_med": 6.5
#   "ph_min": 6.0
#   "ph_max": 7.0
# }

# visualize the data
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#dataset = read_csv(url, names=names)
dataset = pd.read_json(r'C:\Users\Andre\Documents\eye-fish-ia\src\data-set.json')

print(dataset.head(20))
print(dataset.shape)
print(dataset.describe())
print(dataset.groupby('familia').size())

array = dataset.values
X = array[:,0:3]
y = array[:,3]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.30, random_state=1, shuffle=True)
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
