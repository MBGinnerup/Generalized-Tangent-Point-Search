#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
from TPS_2D.maps.random_maps import random_map
from TPS_2D.planners.tps import TPS
from TPS_2D.planners.tps_bidirectional import TPS_Bidirectional
from TPS_2D.planners.tps_theta import TPS_Theta
from TPS_2D.planners.core.pre_processing import TPS_preprocess
from TPS_2D.utils.plot_utils import visualize

#%% Generate map
size = 100
map = random_map(size=size, dense=0.3, distribution=0.5, relative_size=2, small=(1,5), large=(10,20), connectivity_guarantee=True)
start = (0,0)
goal = (size-1, size-1)

#%% Pre-Processing
pre = TPS_preprocess(map)

#%% Planning
path = TPS(map, start, goal, pre=pre)

#%% Visualization
visualize(map, path=path, start=start, goal=goal)

# %%
