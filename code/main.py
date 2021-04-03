import logging
import sklearn.datasets as ds
import pandas as pd
import numpy as np
import math as mt


from sklearn.metrics import mean_squared_error
from mulearn import FuzzyInductor
from mulearn.optimization import TensorFlowSolver
from mulearn.kernel import GaussianKernel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def create_dataset(name):

    #carico il dataset.
    iris_X, iris_y = ds.load_iris(return_X_y=True)  # In iris_X matrice con i quattro valori per oggetto
                                                    # In iris_y etichette rispettive (0,1,2)

    labels = ("Setosa", "Versicolor", "Virginica") # Creo le etichette Setosa, Versicolor e Virginica

    #creo dataframe con le etichette per i valori in iris_X 
    df = pd.DataFrame(iris_X, columns=["Sepal length", "Sepal width",
                                       "Petal length", "Petal width"])

    #per i valori di iris_y associo le rispettive etichette
    df['Class'] = iris_y
    df['Class'] = df['Class'].map(lambda c: labels[c])

    iris_dataset = iris_y.copy()

    if(name == "Setosa"):
        #creo dataset iris_setosa dove le etichette 0 diventano 1 e le altre a 0        
        iris_dataset[iris_dataset != 0] = 2
        iris_dataset[iris_dataset == 0] = 1
        iris_dataset[iris_dataset == 2] = 0

    if(name == "Versicolor"):
        iris_dataset[iris_dataset==2] = 0
    
    if(name == "Virginica"):
        iris_dataset[iris_dataset != 2] = 0
        iris_dataset[iris_dataset == 2] = 1


    return iris_X, iris_dataset


def create_logger(path):   
    
    
    fhandler = logging.FileHandler(filename = path)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    
    logger.addHandler(fhandler)  
    
    return fhandler




handler = create_logger('./log/Setosa_c005_sigma01_penalization01.log')


    
c = 0.05
sigma = 0.1
    
penalization = 0.1
n_iter = 100
    
iris_X, iris_dataset = create_dataset("Setosa")

# Gurobi prova
fi = FuzzyInductor(c = c, k=GaussianKernel(sigma = sigma))
fi.fit(iris_X, iris_dataset)

#rmse gurobi
rmse_gurobi = mean_squared_error(iris_dataset, fi.predict(iris_X), squared=False)
logger.debug("RMSE GUROBI: " + str(rmse_gurobi))

 
    
while n_iter <= 1500:
        
    #TensoFlow prova

    try:
        fi = FuzzyInductor(solver=TensorFlowSolver(n_iter = n_iter, penalization = penalization), c = c, k=GaussianKernel(sigma = sigma))
    except (ModuleNotFoundError, ValueError):
        print('Tensorflow not available')

    fi.fit(iris_X, iris_dataset) #fi impara iris_X con il confronto con le etichette vere

    rmse_tensorflow = mean_squared_error(iris_dataset, fi.predict(iris_X), squared=False)
    logger.debug("RMSE TENSORFLOW: "+ str(rmse_tensorflow))

    #salvare distanza tra (rmse gurobi) e (rmse tensorFlow)
    distance = mt.fabs((rmse_tensorflow) - (rmse_gurobi))
    logger.debug("DISTANCE RMSE: "+ str(distance))

    #salvare coppia num iterazioni - distanza
    couples = [(n_iter, distance)]
    logger.debug("COUPLE(DISTANCE RMSE): "+ str(couples))

    n_iter += 100 


logger.removeHandler(handler)