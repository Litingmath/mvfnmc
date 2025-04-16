import os
import numpy as np

def load_data_from_file(type, folder):

    with open(os.path.join(folder, "luo_simmat_drugs_"+type+".txt"), "r") as inf:
        next(inf)
        int_array = [line.strip("\n").split()[1:] for line in inf]
        intMat = np.array(int_array, dtype=np.float64).T    # drug-target interaction matrix
    return intMat

folder= 'datasets_mv/luo/Dsim/'
type="se"
SD1=load_data_from_file(type,folder)



print(SD1.shape)

print(SD1)