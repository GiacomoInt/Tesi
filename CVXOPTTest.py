import sklearn.datasets as ds
import pandas as pd

from sklearn.metrics import mean_squared_error
from mulearn import FuzzyInductor
from mulearn.kernel import GaussianKernel
from mulearn.optimization import CVXOPTSolver



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


def main(nome):  
    
    '''
    handler = create_handler(f"../../../log/Prove-6/Different-Optimizer/{nome}_c{str(c).replace('.','')}"
                             f"_sigma{str(sigma).replace('.','')}"
                             f"_penalization{str(penalization).replace('.','')}"
                             f"_{str(optimizer)}.log")
    
    #salvo i parametri
    logger.info(f"PARAMETRI DI PARTENZA: nome={nome}, c={c}, sigma={sigma}, penalization={penalization},"
                f"optimizer={optimizer}")'''
    

    iris_X, selected_iris_dataset = create_dataset(nome)

    # Gurobi solver & fit
    fi = FuzzyInductor()
    
    
    '''start = time.time()'''
    fi.fit(iris_X, selected_iris_dataset)
    '''end = time.time()'''
    
    # chis gurobi
    gurobi_chis = fi.chis_
    '''logger.info(f"GUROBI_CHIS: {gurobi_chis}")
    logger.info(f"GUROBI_START: {start}")
    logger.info(f"GUROBI_END: {end}")
    logger.info(f"TEMPO_ESECUZIONE GUROBI(IN EPOCH): {(end-start)}")'''
    fi = FuzzyInductor(solver=CVXOPTSolver())
    
    fi.fit(iris_X, selected_iris_dataset)
    
    cvxopt_chis = fi.chis_
    
    rmse_distance = abs(mean_squared_error(gurobi_chis, cvxopt_chis, squared=False))

    '''logger.removeHandler(handler)'''
    
    return rmse_distance



x = main("Versicolor")
print(f'Rmse: {str(x)}')