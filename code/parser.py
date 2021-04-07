def find_distance_rmse(path):

    file = open(path, 'r')
    Lines = file.readlines()

    coppie = []

    for line in Lines:
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