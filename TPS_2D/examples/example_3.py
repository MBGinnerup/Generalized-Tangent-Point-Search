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
from TPS_2D.planners.core.pre_processing import graph_builder
from TPS_2D.utils import path_queries as pq

# %% Generate map
size = 100
map = random_map(size=size, dense=0.4, distribution=0.5, relative_size=1, small=(1,5), large=(10,20))

# %% Pre-Processing
pre = TPS_preprocess(map)

# %% Planning and build graph
graph = {}
start = pq.random_start(map)
goal = pq.random_goal(map, start, min_dist=75)
path = TPS(map, start, goal, pre=pre, graph=graph, build_graph=True)

# %% Visualize
visualize(map, path=path, start=start, goal=goal, graph=graph)

# %% Build larger graph:
graph = graph_builder(map, pre, build_method=TPS, min_dist=75, iterations=1000, existing_graph=graph)

# %% Visualize
visualize(map, graph=graph)

# %%
