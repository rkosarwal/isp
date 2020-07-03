"""
Author: Rahul Kosarwal
For ISP LOLAS
Model Catalytic
"""


"""
s_1(*x) = x[0]  # A
s_2(*x) = x[1]  # B
s_3(*x) = x[2]  # C
s_4(*x) = x[3]  # D

    A --k_1--> B
    B --k_2--> C
    B + D --k_3--> B + E

"""
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import subplots_adjust
from ispy.routines import timing
import numpy as np
from ispy.routines.bioModel import Model
import time

# propensities
reaction_1 = lambda *x : 1.0 * x[0]
reaction_2 = lambda *x : 1000.0 * x[1]
reaction_3 = lambda *x : 100.0 * x[3] * x[1]

# transitions (S, B, C, P, E)
v_1 = (-1, 1, 0, 0, 0)
v_2 = (0, -1, 1, 0, 0)
v_3 = (0, -1, 0, -1, 1)

# starting state
x_0 = (50,0,0,80,0) # initial populations of the three species.

species_names = ('S','B','C','P','E')

catalytic_model = Model(propensities = [reaction_1,reaction_2,reaction_3], transitions = [v_1,v_2,v_3], initial_state = x_0, species = species_names)

#print(catalytic_model.propensities)
print(catalytic_model.transitions)
print(catalytic_model.initial_state)
print(catalytic_model.species)

# inside a propensity
print(catalytic_model.propensities[0](*catalytic_model.initial_state))

from ispy.ISPalgo import ISP_Method
# ISPLAS
start_time = time.time()
ISP_catalytic = ISP_Method(catalytic_model,10,1e-6,Expander="ISPLAS")

T = np.arange(0.0,0.5,0.01)

output_data = []
for isp_position in T:
	ISP_catalytic.step(isp_position)
	ISP_catalytic.isp_output 		# Prints some information of where the solver is.
	output_data.append(ISP_catalytic.isp_output)
	
	""" Runtime plotting"""
	#OFSP_ABC.plot(inter=True)  # For interactive plotting

	""" Check Point"""
	ISP_catalytic.bechmark()

	""" Probing """
	#X = np.zeros((3,2))
	#X[:,0] = [8,2,0]
	#X[:,1] = [7,2,1]

	X = np.zeros((5,2))
	X[:,0] = [48,2,0,80,0]
	X[:,1] = [47,2,1,80,0]
	#X[:,2] = [48,1,0,79,1]

	ISP_catalytic.checked_states(X)
elapsed_time = time.time() - start_time
print "Time elapsed:" +' '+ str( elapsed_time) +' '+ "seconds"
ISP_catalytic.plotting()
#ISP_catalytic.plot_contour()
ISP_catalytic.plot_checked()
np.savetxt('ISPData_Catalytic.csv', np.column_stack(output_data), delimiter=',')
