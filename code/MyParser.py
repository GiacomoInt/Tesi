
import pandas as pd
import time
import datetime as dt

def epoch_to_millis(epoch):
    
    datetime_time = dt.datetime.fromtimestamp(epoch)
    s = (datetime_time - dt.datetime(1970, 1, 1)).total_seconds()-3600
               
    return s

class MyParser:

    def __init__(self, path):
        self.path =  path
        self.file = open(self.path, 'r')
        self.lines = (self.file).readlines()
    

    def find_distance_rmse(self):

        coppie = []

        for line in (self.lines):
            if 'COUPLE(N_ITER,DISTANCE RMSE)' in line:
                split = line.split(':')
                s = split[3].replace('[','')
                s = s.replace('(','')
                s = s.replace(')','')
                s = s.replace(']','')
                s = s.split(',')
                a = int(s[0])
                b = float(s[1])
                coppie.append((a,b))

        return coppie
    

    def tensorflow_chis(self):

        lista = []
        chis = []
        
        for line in self.lines:
            
            if 'TENSORFLOW_CHIS' in line:
                split = line.split(':')
                s = split[3].replace('[','')
                s = s.replace(']','')
                lista = [float(n) for n in s.split(',')]
                chis.append(lista)

        return chis


    def gurobi_chis(self):
        
        chis = []
        
        for line in self.lines:
            
            if 'GUROBI_CHIS' in line:
                split = line.split(':')
                s = split[3].replace('[','')
                s = s.replace(']','')
                chis = [float(n) for n in s.split(',')]

        return chis


    def execution_time_gurobi(self):
        
        for line in self.lines:
            
            if 'TEMPO_ESECUZIONE GUROBI' in line:
                split = line.split(':')
                
        return epoch_to_millis(float(split[3]))
        
        
    def execution_time_tensorflow(self):
        
        for line in self.lines:
            if 'TEMPO_ESECUZIONE TENSORFLOW' in line:
                split = line.split(':')
                
        return epoch_to_millis(float(split[3]))

