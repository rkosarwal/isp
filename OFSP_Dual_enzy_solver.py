import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import subplots_adjust
from pyme import timing
import numpy as np
from pyme.model import Model
import time

# propensities
reaction_1 = lambda *x : 4.0 * np.maximum(x[0],0.0) * np.maximum(x[1],0.0)   # R1
reaction_2 = lambda *x : 5.0 * np.maximum(x[2],0.0)   # R2
reaction_3 = lambda *x : 1.0 * np.maximum(x[2],0.0)   # R3
reaction_4 = lambda *x : 4.0 * np.maximum(x[3],0.0) * np.maximum(x[4],0.0)   # R4
reaction_5 = lambda *x : 5.0 * np.maximum(x[5],0.0)   # R5
reaction_6 = lambda *x : 1.0 * np.maximum(x[5],0.0)   # R6

"""
reaction_1 = lambda *x : 1.0 * np.maximum(x[0],0.0) * np.maximum(x[1],0.0)   # R1
reaction_2 = lambda *x : 1.0 * np.maximum(x[2],0.0)   # R2
reaction_3 = lambda *x : 0.1 * np.maximum(x[2],0.0)   # R3
reaction_4 = lambda *x : 1.0 * np.maximum(x[3],0.0) * np.maximum(x[4],0.0)   # R4
reaction_5 = lambda *x : 1.0 * np.maximum(x[5],0.0)   # R5
reaction_6 = lambda *x : 0.5 * np.maximum(x[5],0.0)   # R6
"""

# transitions (S, E1, C1,  P, E2, C2)
            # x0  x1  x2  x3  x4  x5
v_1 = (-1, -1, 1, 0, 0, 0)
v_2 = (1, 1, -1, 0, 0, 0)
v_3 = (0, 1, -1, 1, 0, 0)
v_4 = (0, 0, 0, -1, -1, 1)
v_5 = (0, 0, 0, 1, 1, -1)
v_6 = (1, 0, 0, 0, 1, -1)

# starting state  (S, E1, C1,  P, E2, C2)
x_0 = (50, 20, 0,  0, 10, 0) # initial populations of the six species.

species_names = ('S','E1','C1','P','E2','C2')

dual_enzy_model = Model(propensities = [reaction_1,reaction_2,reaction_3,reaction_4,reaction_5,reaction_6], transitions = [v_1,v_2,v_3,v_4,v_5,v_6], initial_state = x_0, species = species_names)
#print(dual_enzy_model.propensities)
print(dual_enzy_model.transitions)
print(dual_enzy_model.initial_state)
print(dual_enzy_model.species)


from pyme.OFSP import OFSP_Solver
# OFSP
start_time = time.time()
OFSP_dual_enzy_model = OFSP_Solver(dual_enzy_model,1,1e-6)

T = np.arange(0.0,2.0,0.01)

output_data = []
for isp_position in T:
    OFSP_dual_enzy_model.step(isp_position)
    OFSP_dual_enzy_model.print_stats
    output_data.append(OFSP_dual_enzy_model.print_stats)

    """ Check Point"""
    OFSP_dual_enzy_model.check_point()

    """ Probing """
    X = np.zeros((6,2))
    X[:,0] = [48,18,2,0,10,0]
    X[:,1] = [48,19,1,1,10,0]

    OFSP_dual_enzy_model.probe_states(X)
elapsed_time = time.time() - start_time
print "Time elapsed:" +' '+ str( elapsed_time) +' '+ "seconds"
OFSP_dual_enzy_model.plot()
#OFSP_dual_enzy_model.plot_contour()
OFSP_dual_enzy_model.plot_checked()
np.savetxt('OFSPData_dual_enzyOFSP.csv', np.column_stack(output_data), delimiter=',')
