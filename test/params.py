import os
import numpy as np
from matscipy.fracture_mechanics.clusters import diamond, set_groups, set_regions
import ase.io
from quippy.potential import Potential

# Interaction potential
calc = Potential("IP SW", param_str="""
<SW_params n_types="1">
<per_type_data type="1" atomic_num="14" />
<per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
      p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
<per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14"
      lambda="42.0" gamma="1.20" eps="2.1675" />
</SW_params>
""")

# Fundamental material properties
el = 'Si'
a0 = 5.43094
surface_energy = 1.3595 * 10  # GPa*A = 0.1 J/m^2

elastic_symmetry = 'cubic'

# Crack system
crack_surface = [1, 1, 1]
crack_front = [1, -1, 0]

vacuum = 6.0

# Simulation control
ds = 0.005
nsteps = 10000
continuation = True

k0 = 1.0
k1_range = [1.0]
dk = 1e-5
dalpha = 0.05
alpha_range = np.linspace(1.0, 8.0, 100)

fmax = 1e-5
max_steps = 1000
flexible = True
prerelax = True

extended_far_field = False

r_III = 32.0
cutoff = 3.77118
r_I = r_III - 3 * cutoff
print(f'r_I = {r_I}, r_III = {r_III}')

n = [2 * int((r_III + cutoff)/ a0), 2 * int((r_III + cutoff)/ a0) - 1, 1]
print('n=', n)

# Setup crack system and regions I, II, III and IV
cryst = diamond(el, a0, n, crack_surface, crack_front)
cluster = set_regions(cryst, r_I, cutoff, r_III)  # carve circular cluster

ase.io.write('cryst.cfg', cryst)
ase.io.write('cluster.cfg', cluster)


