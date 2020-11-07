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
#   "peixe": "pimelodus pictus"
#   "ph_med": 6.5
#   "ph_min": 6.0
#   "ph_max": 7.0
# }

# visualize the data
import pandas
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

dataset = pd.read_json(r'C:\Users\Andre\Documents\eye-fish-ia\src\data-set.json')

array = dataset.values
X = array[:,0:3]
y = array[:,3]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=1, shuffle=True)

filename = r'C:\Users\Andre\Documents\eye-fish-ia\src\test\finalized_model.sav'
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)

print(loaded_model.predict([[8.0, 8.8, 8.4]]))