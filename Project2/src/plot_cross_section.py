import matplotlib.pyplot as plt
import numpy as np
from Stellar import Star

R_sun 	= 6.96e8		# [m] Solar radius
L_sun 	= 3.846e26		# [W] Solar luminosity
M_sun 	= 1.989e30		# [kg] Solar mass

R0 		= R_sun    		# [m] Initial radius
L0 		= L_sun     	# [W]  Initial luminocity
M0 		= M_sun       	# [kg] Initial mass
rho0 	= 1.42e-7*1.408e3   # [kg m^-3] Initial density
T0 		= 15770         	# [K]  Initial temperature

"""Generating plottable arrays"""
SanityStar = Star(R0, L0, T0, rho0, M0, sanity_debug=False)
m, y = SanityStar.solve((M0, 0), 0.01, 1e-4, variable_step=True, debug=False)
R_values, P, L_values, T = y	# Function values
rho = SanityStar.rho(P, T)
n = len(m)						# Length of arrays

F_C_list, F_R_list = SanityStar.FC_list # Normalized convective and radiative flux arrays

R_values /= 6.96e8			# Normalizing radial distance
L_values /= L0				# Normalizng luminosity
R0 /= 6.96e8				# Normalizing initial radius

show_every = 5				
core_limit = 0.995

"""Plotting cros section"""

plt.figure(figsize=(8, 5))
plt.grid()
fig = plt.gcf()  # get current figure
ax = plt.gca()  # get current axis
rmax = 1.2*R0
ax.set_xlim(-rmax, rmax)
ax.set_ylim(-rmax, rmax)
ax.set_aspect('equal')  # make the plot circular
j = show_every
for k in range(0, n-1):
	j += 1
	if j >= show_every:  # don't show every step - it slows things down
		if(L_values[k] > core_limit):  # outside core
			if(F_C_list[k] > 0.0):		# convection
				circR = plt.Circle((0, 0), R_values[k], color='red', fill=False)
				ax.add_artist(circR)
			else:				# radiation
				circY = plt.Circle((0, 0), R_values[k], color='yellow', fill=False)
				ax.add_artist(circY)
		else:				# inside core
			if(F_C_list[k] > 0.0):		# convection
				circB = plt.Circle((0, 0), R_values[k], color='blue', fill=False)
				ax.add_artist(circB)
			else:				# radiation
				circC = plt.Circle((0, 0), R_values[k], color='cyan', fill=False)
				ax.add_artist(circC)
		j = 0
# These are for the legend (drawn outside the main plot)
circR = plt.Circle((2*rmax, 2*rmax), 0.1*rmax, color='red', fill=True)
circY = plt.Circle((2*rmax, 2*rmax), 0.1*rmax, color='yellow', fill=True)
circC = plt.Circle((2*rmax, 2*rmax), 0.1*rmax, color='cyan', fill=True)
circB = plt.Circle((2*rmax, 2*rmax), 0.1*rmax, color='blue', fill=True)
ax.legend([circR, circY, circC, circB], ['Convection outside core', 'Radiation outside core', 'Radiation inside core',
                                         'Convection inside core'], loc=0)  # only add one (the last) circle of each colour to legend
plt.xlabel(r"$R/R_\odot$", fontsize = 14)
plt.ylabel(r"$R/R_\odot$", fontsize = 14)
#plt.xlim(-0.9, 0.9)
#plt.ylim(-0.9, 2)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title(rf"$R_0/R_\odot = ${R0:.2g}, $\rho_0 = {rho0*1e-3:.2g}$ gcm$^{-3}$, $T_0 = {T0}$ K", fontsize = 14)

# Show all plots
#plt.savefig("CrossSec8.jpg")
plt.show()

