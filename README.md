# Generalized-Tangent-Point-Search
A generalized Tangent Point Search (TPS) algorithm for 2D and 3D path planning.

## Overview

This repository contains the implementation of a generalised Tangent 
Point Search (TPS) algorithm for path planning in both 2D and 3D 
environments, proposed as part of a bachelor's thesis at Aarhus 
University. The method extends the improved TPS algorithm of 
[Tai et al. (2024)](https://doi.org/10.1109/CoDIT62066.2024.10708452) 
and [Tai et al. (2025)](https://doi.org/10.1002/rob.22570) with the 
following contributions:

- **Generalised TPS** — handles overlapping convex hulls and start/goal positions inside convex hulls without entering infinite loops
- **Fallback strategy** — guarantees path completeness in all environments
- **3D extension** — extends TPS to three-dimensional environments with terrain handling for real-world elevation maps
- **Graph caching mechanism** — reduces online planning time in repeated-query scenarios
- **Random map generator** — configurable 2D and 3D map generator supporting multiple environment types

---

## Examples

### 2D

- **[`example_1.py`](TPS_2D/examples/example_1.py)** — path planning on a randomly generated map
- **[`example_2.py`](TPS_2D/examples/example_2.py)** — path planning on a real-world map
- **[`example_3.py`](TPS_2D/examples/example_3.py)** — graph caching for repeated path queries

### 3D

- **[`example_1.py`](TPS_3D/examples/example_1.py)** — path planning on a randomly generated 3D map
- **[`example_2.py`](TPS_3D/examples/example_2.py)** — path planning on a real-world 3D elevation map

---

## Real-World Maps

The real-world environments used in this project are obtained using 
[Map2Map](https://map2map.io), which converts OBJ files of real-world 
terrain and building data into occupancy grids suitable for path 
planning. The environments included are:

- **University Park, Aarhus** — characterised by large university buildings
- **Åbyhøj, Aarhus** — characterised by a dense arrangement of smaller residential buildings
- **Jørpeland, Norway** — mountainous terrain with significant elevation variation (3D only)

---

## Fallback Interface

All fallback methods follow a common interface compatible with most 
standard path planners:

```python
fallback_method(grid, start, goal) -> path
```

This makes it straightforward to replace the default fallback with 
any alternative planner that accepts a grid, start position, and 
goal position as input and returns a path.

---

## Installation

```bash
git clone https://github.com/MBGinnerup/Generalized-Tangent-Point-Search.git
cd Generalized-Tangent-Point-Search
pip install -r requirements.txt
```

### Requirements

- Python 3.11+
- numpy
- scipy
- matplotlib
- scikit-image
- tqdm

---

## References

- Tai et al. (2024) — *Tangent Point Search*, CoDIT 2024. [DOI](https://doi.org/10.1109/CoDIT62066.2024.10708452)
- Tai et al. (2025) — *Improved TPS*, Journal of Field Robotics. [DOI](https://doi.org/10.1002/rob.22570)
- Map2Map — *Real-world map conversion*. [map2map.io](https://map2map.io)

