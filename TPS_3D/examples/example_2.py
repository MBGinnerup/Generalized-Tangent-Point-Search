#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
from TPS_3D.maps.real_world_maps import real_world_map
from TPS_3D.planners.tps_3d import TPS
from TPS_3D.planners.tps_bidirectional_3d import TPS_Bidirectional
from TPS_3D.planners.tps_theta_3d import TPS_Theta
from TPS_3D.planners.core.pre_processing import TPS_preprocess
from TPS_3D.utils.plot_utils import visualize
from TPS_3D.utils import path_queries as pq

#%% Generate map
map, terrain = real_world_map(0.70, map_name='jørpeland')

#%% Pre-Processing
pre = TPS_preprocess(map, terrain=terrain)

#%% Planning
start = pq.random_start_environment(map, terrain=terrain)
goal = pq.random_goal_environment(map, terrain=terrain, start=start, min_dist=150)
path = TPS(map, start, goal, pre=pre)

#%% Visualization (plotting may take some time)
visualize(map, path=path, start=start, goal=goal)

# %%
