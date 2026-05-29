#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
from TPS_2D.maps.real_world_maps import real_world_map
from TPS_2D.planners.tps import TPS
from TPS_2D.planners.tps_bidirectional import TPS_Bidirectional
from TPS_2D.planners.tps_theta import TPS_Theta
from TPS_2D.planners.core.pre_processing import TPS_preprocess
from TPS_2D.utils.plot_utils import visualize
from TPS_2D.utils import path_queries as pq

#%% Generate map
map = real_world_map(0.75, map_name='university')

#%% Pre-Processing
pre = TPS_preprocess(map)

#%% Planning
start = pq.random_start(map)
goal = pq.random_goal(map, start, min_dist=100)
path = TPS(map, start, goal, pre=pre)

#%% Visualization
visualize(map, path=path, start=start, goal=goal)

# %%
