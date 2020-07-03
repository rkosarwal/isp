"""
First ISPLOLAS solver using the ABC model.
"""

import numpy as np

# Importing the model class
from ABC_model import ABC_model

# calling the class
from ispy.ISPalgo import ISP_Method

# ISPLAS
ISP_CAT = ISP_Method(ABC_model, 1, 1e-6, Expander="ISPLOLAS", validity_test=None)

# Example with Validity function
"""
def validity_func(X):
  return np.sum(np.abs(X),axis=0) == 10 # since we started with 10 states.
# The solver initialisation would look like

ISP_ABC = ISP_Method(ABC_model,10,1e-6,validity_test = validity_func)
"""



T = np.arange(0.01,1.0,0.01)

plot_data = []
for t in T:
	ISP_CAT.step(t)
	ISP_CAT.isp_output 		# Prints some information of where the solver is.
	plot_data.append(ISP_CAT.isp_output)
	
	""" Runtime plotting"""
	#OFSP_ABC.plot(inter=True)  # For interactive plotting

	""" Check Point"""
	ISP_CAT.bechmark()

	""" Probing """
	#X = np.zeros((3,2))
	#X[:,0] = [8,2,0]
	#X[:,1] = [7,2,1]

	X = np.zeros((3,2))
	X[:,0] = [3,7,0]
	X[:,1] = [2,7,1]

	ISP_CAT.checked_states(X)
ISP_CAT.plotting()
ISP_CAT.plot_checked()
np.savetxt('ISPData_ToyModel.csv', np.column_stack(plot_data), delimiter=',')
