import csv
import numpy as np
import os
from timebudget import timebudget
from multiprocessing import Pool
import math
import time

from DataFileManagement import *
from ParameterObject import *
from SimCompModifiedFHN import *

if __name__ == "__main__":
    #filename = '[insertname].csv'
    #[u, v] = LoadAndProcessCsv(filename)
    u = np.zeros((512,512))
    v = np.zeros((512,512))

    # Set initial conditions for a planar wave
    for i in range(50):
        u[i,:] = 1


    # Initialize the Params object
    params = Params(
        D_u=0.001,  # Diffusion coefficient for u
        alpha=0.152,     # Parameter a
        beta=1.327,     # Parameter b
        epsilon=0.006,
        mu=1.183,
        gamma=0.14,
        theta=-0.004,
        delta=1.254,
        dt=0.1,  # Time step
        dx=12/512,    # Spatial resolution
        nx=512,    # Grid size in x direction
        ny=512,    # Grid size in y direction
        last_step=2000,  # Time step to stop simulation
    )
    
    new_u, new_v = update(u, v, params)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    np.savetxt('u_matrix_'+timestr+'.txt',new_u,fmt='%.32f')
    np.savetxt('v_matrix_'+timestr+'.txt',new_v,fmt='%.32f')
