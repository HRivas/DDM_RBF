#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:35:26 2018

@author: Ricardo Rivas

Mean elapsed time to solve linear system.
------------------------------------------

Mean of 10 executions with different number of nodes.
"""
import numpy as np
import matplotlib.pyplot as plt

# Averages
nnodes =  [90, 360, 1440, 1875, 3000, 4500, 5250]
direct = [0.0011277, 0.011631, 0.406818, 0.9121465, 4.451053, 13.07872, 25.8539363636]
shur = [0.0023996, 0.00922, 0.2192392, 0.4504903, 1.562957, 4.796953, 6.0807709091]
pshur = [0.1769814, 0.787959, 2.321872, 2.254407, 2.660694, 3.678235, 4.7503245455]

# Max - Min Data
mind = [0.001072,0.011551,0.390566,0.846418,3.86322,10.9877,24.3666]
maxd = [0.00128,0.0118,0.426079,0.985208,4.64653,13.3884,26.5235]
minsh = [0.002309,0.009085,0.212316,0.384243,1.26259,4.1252,5.61865]
maxsh = [0.002679,0.009359,0.226796,0.506423,1.6503,4.90305,6.36526]
minpsh = [0.086187,0.541127,1.91274,1.45284,1.6003,2.81599,3.94634]
maxpsh = [0.3676,1.02634,2.68298,3.20616,3.0921,4.27569,5.37765]
mind = -np.array(mind)+np.array(direct)
maxd = np.array(maxd)-np.array(direct)
minsh = -np.array(minsh)+np.array(shur)
maxsh = np.array(maxsh)-np.array(shur)
minpsh = -np.array(minpsh)+np.array(pshur)
maxpsh = np.array(maxpsh)-np.array(pshur)

plt.close('all')
plt.figure()
plt.errorbar(nnodes, direct, yerr = [mind, maxd], fmt='-', marker = '.')
plt.errorbar(nnodes, shur, yerr = [minsh, maxsh], fmt='-', marker = '.')
plt.errorbar(nnodes, pshur, yerr = [minpsh, maxpsh], fmt='-', marker = '.')
plt.grid('on')
plt.title('Mean elapsed time to solve linear system')
plt.xlabel('Number of nodes')
plt.ylabel('Time [s]')
plt.legend(['Direct solution','Shur Complement','Parallelized Shur'])
plt.savefig('performance.pdf')