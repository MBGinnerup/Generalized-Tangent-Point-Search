#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
from TPS_3D.maps.random_maps import random_map
from TPS_3D.planners.tps_3d import TPS
from TPS_3D.planners.tps_bidirectional_3d import TPS_Bidirectional
from TPS_3D.planners.tps_theta_3d import TPS_Theta
from TPS_3D.planners.core.pre_processing import TPS_preprocess
from TPS_3D.utils.plot_utils import visualize

#%% Generate map
size = 25
map = random_map(size=size, dense=0.2, distribution=0.5, relative_size=1, small=(10,50), large=(60,100), clustered=True, connectivity_guarantee=True)
start = (0,0,0)
goal = (size-1, size-1, size-1)

#%% Pre-Processing
pre = TPS_preprocess(map)

#%% Planning
path = TPS(map, start, goal, pre=pre)

#%% Visualization
visualize(map, path=path, start=start, goal=goal, alpha_fig=0.25)

# %%