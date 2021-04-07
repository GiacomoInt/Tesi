import logging
import sklearn.datasets as ds
import pandas as pd
import numpy as np
import math as mt

from itertools import product
from sklearn.metrics import mean_squared_error
from mulearn import FuzzyInductor
from mulearn.optimization import TensorFlowSolver
from mulearn.kernel import GaussianKernel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def create_handler(path):   
    
    
    fhandler = logging.FileHandler(filename = path)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    
    logger.addHandler(fhandler)  
    
    return fhandler

def create_dataset(name):

    #load dataset, in iris_X values, in iris_y labels 0 1 2
    iris_X, iris_y = ds.load_iris(return_X_y=True)  

    labels = ("Setosa", "Versicolor", "Virginica") 

    #dataframe with correct labels for respective values
    df = pd.DataFrame(iris_X, columns=["Sepal length", "Sepal width",
                                       "Petal length", "Petal width"])

    #associating 
    df['Class'] = iris_y
    df['Class'] = df['Class'].map(lambda c: labels[c])

    #dataset copy for labels 0 1
    selected_iris_dataset = iris_y.copy()

    #dataset selected with labels
    if(name == "Setosa"):        
        selected_iris_dataset[selected_iris_dataset != 0] = 2
        selected_iris_dataset[selected_iris_dataset == 0] = 1
        selected_iris_dataset[selected_iris_dataset == 2] = 0
    elif(name == "Versicolor"):
        selected_iris_dataset[selected_iris_dataset==2] = 0    
    elif(name == "Virginica"):
        selected_iris_dataset[selected_iris_dataset != 2] = 0
        selected_iris_dataset[selected_iris_dataset == 2] = 1


    return iris_X, selected_iris_dataset

#parametri possibile funzione (path_log, iris_type, c, sigma, penalization)
def shot(nome, c, sigma, penalization):  
    
    handler = create_handler(f"../log/{nome}-Chi/c{str(c).replace('.','')}_sigma{str(sigma).replace('.','')}_penalization{str(penalization).replace('.','')}.log")

    iris_X, selected_iris_dataset = create_dataset("Setosa")

    #parametri     
    #c = 0.05
    #sigma = 0.1
    #penalization = 0.1 

    n_iter = 100

    # Gurobi solver & fit
    fi = FuzzyInductor(c=c, k=GaussianKernel(sigma=sigma))
    fi.fit(iris_X, selected_iris_dataset)

    # rmse gurobi
    rmse_gurobi = mean_squared_error(selected_iris_dataset, fi.predict(iris_X), squared=False)
    logger.info("RMSE GUROBI: " + str(rmse_gurobi))

    # TensorFlow solver
    fi = FuzzyInductor(solver=TensorFlowSolver(n_iter=n_iter, penalization=penalization),
                         c=c,k=GaussianKernel(sigma=sigma))
    try:
        fi.fit(iris_X, selected_iris_dataset)
    except (ModuleNotFoundError, ValueError):
        print('Tensorflow not available')


    # rmse TensorFlow
    rmse_tensorflow = mean_squared_error(selected_iris_dataset, fi.predict(iris_X), squared=False)
    logger.info("RMSE TENSORFLOW: " + str(rmse_tensorflow))

    # calcolo distanza
    distance = abs(rmse_tensorflow - rmse_gurobi)
    logger.info("DISTANCE RMSE: " + str(distance))

    #coppia n_iter, distance
    couple = [(n_iter, distance)]
    logger.info("COUPLE(N_ITER,DISTANCE RMSE): " + str(couple))

    # salvo i chi
    chi_ = fi.chis_
    logger.info("CHI: " + str(chi_))

    #incremento n
    n = 200

    # faccio ciclo fino a 10000
    while n <= 10000:

        # TensorFlow solver
        fi = FuzzyInductor(solver=TensorFlowSolver(initial_values=chi_, n_iter=n_iter, penalization=penalization),
                            c=c,k=GaussianKernel(sigma=sigma))
        try:
            fi.fit(iris_X, selected_iris_dataset)
        except (ModuleNotFoundError, ValueError):
            print('Tensorflow not available')


        # rmse TensorFlow
        rmse_tensorflow = mean_squared_error(selected_iris_dataset, fi.predict(iris_X), squared=False)
        logger.info("RMSE TENSORFLOW: " + str(rmse_tensorflow))

        # calcolo distanza
        distance = abs(rmse_tensorflow - rmse_gurobi)
        logger.info("DISTANCE RMSE: " + str(distance))

        #coppia n_iter, distance
        couple = [(n, distance)]
        logger.info("COUPLE(N_ITER,DISTANCE RMSE): " + str(couple))

        # salvo i chi
        chi_ = fi.chis_
        logger.info("CHI: " + str(chi_))

        #incremento n_iter
        n += 100

    logger.removeHandler(handler)


def main():

    for i in list(product(["Setosa","Versicolor","Virginica"],[0.05,1,75,200],[0.1,0.25,0.5],[0.1,10,100], repeat=1)):
        shot(*i)



if __name__ == "__main__":
    main()