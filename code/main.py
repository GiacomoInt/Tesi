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

#NEED TO CREATE A REAL LOGGER
def create_handler(path):   
    
    
    fhandler = logging.FileHandler(filename = path)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    
    logger.addHandler(fhandler)  
    
    return fhandler


#parametri possibile funzione (path_log, iris_type, c, sigma, penalization)
def main():

    handler = create_handler('./log/prova6.log')

    iris_X, selected_iris_dataset = create_dataset("Setosa")

    #parametri     
    c = 75
    sigma = 0.25

    penalization = 0.1 

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

    # faccio ciclo fino a 1500
    while n <= 1500:

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


if __name__ == "__main__":
    main()