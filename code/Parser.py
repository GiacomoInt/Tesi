class Parser:

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