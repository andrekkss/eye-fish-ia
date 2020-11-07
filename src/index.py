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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
# Load dataset
dataset = pd.read_json(r'C:\Users\Andre\Documents\eye-fish-ia\src\data-set.json')
filename = 'finalized_model.sav'

array = dataset.values
X = array[:,0:3]
y = array[:,3]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.30, random_state=1, shuffle=True)
# Spot Check Algorithms

# Make predictions on validation dataset
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

joblib.dump(model, filename)
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))