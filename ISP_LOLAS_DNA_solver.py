"""
Author: Rahul Kosarwal
For ISP LOLAS
Model DNA Damage Signal
"""

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import subplots_adjust
from ispy.routines import timing
import numpy as np
from ispy.routines.bioModel import Model
import time

# Initial values
x5 = 2
x11 = 1*10**-3
x21 = 0
x23 = 5*0.13533
x19 = 1.95
x15 = 0
x0 = 3*10**-2
x3 = 5
x14 = 0
x10 = 6.3
x18 = 1*10**-3
x20 = 1*10**-3
x1 = 1*10**-3
x7 = 1*10**-3
x9 = 1*10**-4
x26 = 1*10**-4
x22 = 1*10**-2
x16 = 0
x17 = 0
x12 = 1
x13 = 1*10**-4
x4 = 13.5
x6 = 1*10**-3
x24 = 2.65*10**-2
x8 = 4*10**-4

# propensities
reaction_1 = lambda *x : (5*10**-3)*np.maximum(x[6],0.0)*np.maximum(x[7],0.0) # R3
reaction_2 = lambda *x : (2.5*10**-3)*np.maximum(x[0],0.0) # R4
reaction_3 = lambda *x : (5*10**-4)*np.maximum(x[0],0.0) # R13
reaction_4 = lambda *x : (5*10**-4)*np.maximum(x[0],0.0)*np.maximum(x[8],0.0) # R18
reaction_5 = lambda *x : (5*10**-3)*np.maximum(x[5],0.0) # R19
reaction_6 = lambda *x : (5*10**-4)*np.maximum(x[0],0.0)*np.maximum(x[9],0.0) # R20
reaction_7 = lambda *x : (5*10**-5)*np.maximum(x[1],0.0) # R21
reaction_8 = lambda *x : (5*10**-4)*np.maximum(x[0],0.0)*np.maximum(x[10],0.0) # R44
reaction_9 = lambda *x : (5*10**-5)*np.maximum(x[2],0.0)*np.maximum(x[3],0.0) # R45
reaction_10 = lambda *x : (4*10**-2)*np.maximum(x[13],0.0)*np.maximum(x[11],0.0) # R49
reaction_11 = lambda *x : (2.5*10**-3)*np.maximum(x[14],0.0)*np.maximum(x[11],0.0) # R50
reaction_12 = lambda *x : (5*10**-8)*np.maximum(x[2],0.0) # R51
reaction_13 = lambda *x : (5*10**-7) # R52
reaction_14 = lambda *x : (5*10**-5)*np.maximum(x[2],0.0) # R53
reaction_15 = lambda *x : (1*10**-2)*np.maximum(x[2],0.0)*np.maximum(x[14],0.0) # R54
reaction_16 = lambda *x : (5*10**-8)*np.maximum(x[16],0.0) # R55
reaction_17 = lambda *x : (5*10**-5) # R56
reaction_18 = lambda *x : (5*10**-3)*np.maximum(x[3],0.0) # R57
reaction_19 = lambda *x : (5*10**-4)*np.maximum(x[10],0.0) # R59
reaction_20 = lambda *x : (2.5*10**-3)*np.maximum(x[4],0.0)*np.maximum(x[0],0.0) # R46
reaction_21 = lambda *x : (2.5*10**-3)*np.maximum(x[1],0.0)*np.maximum(x[4],0.0) # R47
reaction_22 = lambda *x : (2.5*10**-3)*np.maximum(x[4],0.0)*np.maximum(x[5],0.0) # R48
reaction_23 = lambda *x : (5*10**-3) # R1
reaction_24 = lambda *x : (5*10**-4)*np.maximum(x[6],0.0) # R2
reaction_25 = lambda *x : (5*10**-5) # R37
reaction_26 = lambda *x : (5*10**-3)*np.maximum(x[8],0.0) # R39
reaction_27 = lambda *x : (1.75*10**-4)*np.maximum(x[17],0.0) # R27
reaction_28 = lambda *x : (2.25*10**-2)*np.maximum(x[13],0.0)*np.maximum(x[8],0.0) # R26
reaction_29 = lambda *x : (1.75*10**-4)*np.maximum(x[18],0.0) # R33
reaction_30 = lambda *x : (2.5*10**-3)*np.maximum(x[8],0.0)*np.maximum(x[14],0.0) # R32
reaction_31 = lambda *x : (1.0*10**-2)*np.maximum(x[23],0.0) # R38
reaction_32 = lambda *x : (5*10**-8) # R34
reaction_33 = lambda *x : (2.25*10**-2)*np.maximum(x[13],0.0)*np.maximum(x[9],0.0) # R24
reaction_34 = lambda *x : (1.75*10**-4)*np.maximum(x[19],0.0) # R25
reaction_35 = lambda *x : (2.5*10**-3)*np.maximum(x[14],0.0)*np.maximum(x[9],0.0) # R30
reaction_36 = lambda *x : (1.75*10**-4)*np.maximum(x[20],0.0) # R31
reaction_37 = lambda *x : (1.0*10**-2)*np.maximum(x[13],0.0)*np.maximum(x[9],0.0) # R35
reaction_38 = lambda *x : (1.5*10**-3)*np.maximum(x[14],0.0)*np.maximum(x[9],0.0) # R36
reaction_39 = lambda *x : (2.0*10**-3) # R40
reaction_40 = lambda *x : (1.0*10**-4)*np.maximum(x[3],0.0) # R42
reaction_41 = lambda *x : (5.0*10**-4)*np.maximum(x[10],0.0) # R43
reaction_42 = lambda *x : (7.5*10**-2)*np.maximum(x[2],0.0) # R5
reaction_43 = lambda *x : (2.5*10**-3)*np.maximum(x[12],0.0) # R6
reaction_44 = lambda *x : (1.25*10**-3)*np.maximum(x[12],0.0)*np.maximum(x[21],0.0) # R7
reaction_45 = lambda *x : (2.5*10**-4)*np.maximum(x[22],0.0) # R8
reaction_46 = lambda *x : (2.5*10**-2)*np.maximum(x[22],0.0)*np.maximum(x[13],0.0) # R22
reaction_47 = lambda *x : (1.75*10**-3)*np.maximum(x[13],0.0) # R23
reaction_48 = lambda *x : (1.75*10**-4)*np.maximum(x[17],0.0) # R27
reaction_49 = lambda *x : (2.25*10**-2)*np.maximum(x[13],0.0)*np.maximum(x[21],0.0) # R26
reaction_50 = lambda *x : (2.0*10**-3)*np.maximum(x[13],0.0)*np.maximum(x[13],0.0) # R17
reaction_51 = lambda *x : (1.9*10**-2)*np.maximum(x[24],0.0)*np.maximum(x[14],0.0) # R28
reaction_52 = lambda *x : (5.0*10**-4)*np.maximum(x[14],0.0) # R14
reaction_53 = lambda *x : (5.0*10**-4)*np.maximum(x[14],0.0) # R29
reaction_54 = lambda *x : (5.0*10**-2)*np.maximum(x[2],0.0) # R68
reaction_55 = lambda *x : (8.0*10**-4)*np.maximum(x[15],0.0) # R69

#reaction_54 = lambda *x : (2.5*10**-3)*np.maximum(x[24],0.0)*np.maximum(x[14],0.0) # R28
#reaction_55 = lambda *x : (2.5*10**-3)*np.maximum(x[15],0.0) # R9


#             (x0, x1,  x2,  x3,  x4,  x5,  x6, x7, x8,  x9,  x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24)
# transitions (x5, x11, x21, x23, x19, x15, x0, x3, x14, x10, x18, x20, x1,  x7,  x9,  x26, x22, x16, x17, x12, x13, x4,  x6,  x24, x8)
v_1 = (1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R3
v_2 = (-1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R4
v_3 = (-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R13
v_4 = (-1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R18
v_5 = (1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R19
v_6 = (-1, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R20
v_7 = (1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R21
v_8 = (-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R44
v_9 = (0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R45
v_10 = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R49
v_11 = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R50
v_12 = (0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R51
v_13 = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R52
v_14 = (0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R53
v_15 = (0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R54
v_16 = (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0) # R55
v_17 = (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R56
v_18 = (0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0) # R57
v_19 = (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R59
v_20 = (-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R46
v_21 = (0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R47
v_22 = (0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R48
v_23 = (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R1
v_24 = (0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # R2
v_25 = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R37
v_26 = (0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R39
v_27 = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  1,  0,  0, 0, -1, 0, 0, 0, 0,  0,  0, 0) # R27
v_28 = (0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,  -1,  0,  0, 0, 1, 0, 0, 0, 0,  0,  0, 0) # R26
v_29 = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  1,  0, 0, 0, -1, 0, 0, 0,  0,  0, 0) # R33
v_30 = (0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,  0,  -1,  0, 0, 0, 1, 0, 0, 0,  0,  0, 0) # R32
v_31 = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  -1, 0) # R38
v_32 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R34
v_33 = (0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,  -1,  0,  0, 0, 0, 0, 1, 0, 0,  0,  0, 0) # R24
v_34 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  1,  0,  0, 0, 0, 0, -1, 0, 0,  0,  0, 0) # R25
v_35 = (0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,  0,  -1,  0, 0, 0, 0, 0, 1, 0,  0,  0, 0) # R30
v_36 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  1,  0, 0, 0, 0, 0, -1, 0,  0,  0, 0) # R31
v_37 = (0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,  -1,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R35
v_38 = (0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,  0,  -1,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R36
v_39 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R40
v_40 = (0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R42
v_41 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R43
v_42 = (0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R5
v_43 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,  0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R6
v_44 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,  0,  0,  0, 0, 0, 0, 0, 0, -1,  1,  0, 0) # R7
v_45 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0, 0, 0, 0, 0, 0, 1,  -1,  0, 0) # R8
v_46 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0, 0, 0, 0, 0, 0, 0,  -1,  0, 0) # R22
v_47 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  -1,  0,  0, 0, 0, 0, 0, 0, 0,  1,  0, 0) # R23
v_48 = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  1,  0,  0, 0, -1, 0, 0, 0, 0,  0,  0, 0) # R27
v_49 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  -1,  0,  0, 0, 1, 0, 0, 0, -1,  0,  0, 0) # R26
v_50 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  -1,  0,  0, 0, 0, 0, 0, 0, 1,  0,  0, 0) # R17
v_51 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0, 0, 0, 0, 0, 0, 0,  0,  0, -1) # R28
v_52 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  -1,  0, 0, 0, 0, 0, 0, 1,  0,  0, 0) # R14
v_53 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  -1,  0, 0, 0, 0, 0, 0, 0,  0,  0, 1) # R29
v_54 = (0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R68
v_55 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  -1, 0, 0, 0, 0, 0, 0,  0,  0, 0) # R69

# starting state (x5, x11, x21, x23, x19, x15, x0, x3, x14, x10, x18, x20, x1,  x7,  x9,  x26, x22, x16, x17, x12, x13, x4,  x6,  x24, x8)
#x_0 = (2, 10**-3, 0, 5*0.13533, 1.95, 0, 3*10**-2, 5, 0, 6.3, 10**-3, 10**-3, 10**-3, 10**-3, 10**-4, 10**-4, 10**-2) # initial populations of the three species.
      #(x5, x11, x21, x23, x19, x15, x0, x3, x14, x10, x18, x20, x1,  x7,  x9,  x26, x22, x16, x17, x12, x13, x4,  x6,  x24, x8)
x_0_ratio = (2,1,0,1,2,0,3,5,0,6,1,1,1,1,1,1,1,0,0,1,1,13,1,2,1)

#species_names = ('CycD_CDK4_6','p27_CycD_CDK4_6', 'E2F', 'Rb',  'Rb_E2F',  'p21_CycD_CDK4_6',  'CycD',  'CDK4_6',  'p21',  'p27',  'p16',  'Rb-PP_E2F',  'CycE', 'CycE_CDK2-P','CycA_CDK2-P',  'X',   'Rb-PPPP',  'p21_CycE_CDK2-P',  'p21_CycA_CDK2-P',  'p27_CycE_CDK2-P',  'p27_CycA_CDK2-P',   'CDK2',   'CycE_CDK2',   'p53',  'CycA_CDK2')
#                   ('x5',           'x11',       'x21', 'x23',  'x19',          'x15',         'x0',     'x3',    'x14',  'x10',  'x18',     'x20',      'x1',      'x7',         'x9',      'x26',    'x22',          'x16',              'x17',              'x12',              'x13',          'x4',      ' x6',       'x24',     'x8')
species_names = ('x5','x11','x21','x23','x19','x15','x0','x3','x14','x10','x18','x20','x1','x7',' x9','x26','x22','x16','x17','x12','x13','x4','x6','x24','x8')

damage_dna = Model(
    propensities = [reaction_1,reaction_2,reaction_3,reaction_4,reaction_5,reaction_6,reaction_7,reaction_8,reaction_9,reaction_10,reaction_11,reaction_12,reaction_13,reaction_14,reaction_15,reaction_16,reaction_17,reaction_18,reaction_19,reaction_20,reaction_21,reaction_22,reaction_23,reaction_24,reaction_25,reaction_26,reaction_27,reaction_28,reaction_29,reaction_30,reaction_31,reaction_32,reaction_33,reaction_34,reaction_35,reaction_36,reaction_37,reaction_38,reaction_39,reaction_40,reaction_41,reaction_42,reaction_43,reaction_44,reaction_45,reaction_46,reaction_47,reaction_48,reaction_49,reaction_50,reaction_51,reaction_52,reaction_53,reaction_54,reaction_55],
    transitions = [v_1,v_2,v_3,v_4,v_5,v_6,v_7,v_8,v_9,v_10,v_11,v_12,v_13,v_14,v_15,v_16,v_17,v_18,v_19,v_20,v_21,v_22,v_23,v_24,v_25,v_26,v_27,v_28,v_29,v_30,v_31,v_32,v_33,v_34,v_35,v_36,v_37,v_38,v_39,v_40,v_41,v_42,v_43,v_44,v_45,v_46,v_47,v_48,v_49,v_50,v_51,v_52,v_53,v_54,v_55],
    initial_state = x_0_ratio,
    species = species_names
    )

T = np.arange(0.0,1.0,0.1)
#T = np.arange(0.0,1.5,0.1)

from ispy.ISPalgo import ISP_Method
"""
def validity_function(X):
	on_off = X[0,:] < 2
	neg_states = np.logical_and.reduce(X >= 0, axis=0)
	return np.multiply(on_off,neg_states)
"""
start_time = time.time()
ISP_obj = ISP_Method(damage_dna,10,1e-6,Expander="ISPLOLASBLNP",validity_test=None)

"""
for isp_position in range(10):
	ISP_obj.step(0.1*(isp_position+1))
"""

output_data = []
for isp_position in T:
	ISP_obj.step(isp_position)
	ISP_obj.isp_output 		# Prints some information of where the solver is.
	output_data.append(ISP_obj.isp_output)

	""" Runtime plotting"""
	#OFSP_ABC.plot(inter=True)  # For interactive plotting

	""" Check Point"""
	ISP_obj.bechmark()

	""" Probing """
	X = np.zeros((25,1))
	X[:,0] = [2,1,0,1,2,0,3,5,0,6,1,1,1,1,1,1,1,0,0,1,1,13,1,2,1]
	#X[:,1] = [7,2,1]

	ISP_obj.checked_states(X)
elapsed_time = time.time() - start_time
print "Time elapsed:" +' '+ str( elapsed_time) +' '+ "seconds"
ISP_obj.plotting()
ISP_obj.plot_checked()
np.savetxt('ISPData_DNA.csv', np.column_stack(output_data), delimiter=',')