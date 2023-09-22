import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv('Iris.csv')
iris = iris.drop('Id', axis=1)

x = np.array(iris.drop(columns='Species'))
y = np.array(iris['Species'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Definimos una lista de modelos para evaluar
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("SVC", SVC()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("KNeighbors", KNeighborsClassifier())
]

best_models = []
best_score = 0

# Entrenamos y evaluamos cada modelo
for name, model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test) * 100
    print(f"Precisión del modelo {name}: {score:.2f}%")
    
    # Verificamos si el modelo actual tiene una mejor precisión que el mejor registrado
    if score > best_score:
        best_score = score
        best_models = [name]
    elif score == best_score:
        best_models.append(name)

# Imprimimos el modelo o modelos con la mayor precisión
if len(best_models) == len(models):
    print("\nTodos los modelos tuvieron el mismo porcentaje de precisión.")
else:
    models_str = ", ".join(best_models)
    print(f"\nEl modelo más preciso es {models_str} con una precisión de {best_score:.2f}%.")

# Si deseas hacer una predicción con un modelo específico, puedes hacerlo de la siguiente manera:
model_to_predict = KNeighborsClassifier()
model_to_predict.fit(x_train, y_train)
prediction = model_to_predict.predict([x_test[0]])
print(f"\nPredicción: {prediction[0]}")
