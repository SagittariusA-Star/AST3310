import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from Stellar import Star

R_sun 	= 6.96e8		# [m] Solar radius
L_sun 	= 3.846e26		# [W] Solar luminosity
M_sun 	= 1.989e30		# [kg] Solar mass

R0 		= 1.1*R_sun    # [m] Initial radius
L0 		= L_sun        # [W]  Initial luminocity
M0 		= M_sun        # [kg] Initial mass
rho0 	= 60e-3        # [kg m^-3] Initial density
T0 		= 5770         # [K]  Initial temperature

"""Generating plottable arrays of function values"""
Sun 						= Star(R0, L0, T0, rho0, M0, sanity_debug=False)
m, y 						= Sun.solve((M0, 0), 0.01, 1e-4, variable_step=True, debug=False)
R_values, P, L_values, T	= y				# Function values
Rho 						= Sun.rho(P, T)	# [kgm^-3] Density
n 							= len(m)		# Array length
F_C_list, F_R_list 			= Sun.FC_list	# Normalized relative convective and radiative energy flux

"""Creating empty array to fill"""
epsI 		= np.zeros_like(m)	# PPI efficiency
epsII 		= np.zeros_like(m)	# PPII efficiency
epsIII 		= np.zeros_like(m)	# PPIII efficiency
epsTOT 		= np.zeros_like(m)	# Total energy production
eps_array 	= np.zeros_like(m)	# Energy produced per mass unit
nabla_stab	= np.zeros_like(m)	# Radiatiative temperature gradient
nabla_star	= np.zeros_like(m)	# Stars actual temperature gradient
nabla_ad	= np.ones_like(m)*Sun.nabla_ad	# Adiabatic temperature gradient

for i in range(len(m)):
	"""Filling arrays"""
	T9 				= T*1e-9	#[GK] Temperatures
	eI, eII, eIII 	= Sun.PPchain(T9[i], Rho[i])	
	epsI[i] 		= eI
	epsII[i] 		= eII
	epsIII[i] 		= eIII
	epsTOT[i] 		= eI + eII + eIII
	eps_array[i] 	= Sun.epsilon(T9[i], Rho[i])
	nabla_stab[i]		= Sun.nabla_stable(Rho[i], T[i], R_values[i], m[i], L_values[i])
	nabla_star[i]		= Sun.nabla_star(Rho[i], T[i], R_values[i], m[i], L_values[i])

R_values 	/= R_sun	# Normalizing radial distance
L_values 	/= L_sun	# Normalizing luminosity
R0 			/= R_sun	# Normalizing initial radius

CoreRadius = R_values[np.where(L_values <= 0.995)]  
CoreRadius = CoreRadius[0]									# [R_sun] Core radius
ConvRadius = R_values[np.where(nabla_stab>Sun.nabla_ad)]
ConvRadius = ConvRadius[-1]									# [R_sun] Inner convection zone radius

"""Printing out how close R, L and m approach zero, 
   as well as core and inner convection zone radii"""
print("Errors:")
print(f"R_last: {100*(1 - abs(R_values[-1] - R0)/R0)} %")
print(f"L_last: {100*(1 - abs(L_values[-1] - L0)/L0)} %")
print(f"m_last: {100*(1 - abs(m[-1] - M0)/M0)} %")
print(f"R_core: {CoreRadius} R_sun")
print(f"R_conv_inner: {ConvRadius} R_sun")

"""Ploting cross-section parameters, and cross-section of star"""

plt.subplots(2, 2, figsize = (9, 7))
plt.tight_layout(pad = 2.6)

grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.3)

plt.subplot(grid[0, 0])
plt.plot(R_values, F_C_list, label=r"$F_C$")
plt.plot(R_values, F_R_list, label=r"$F_R$")
plt.plot(CoreRadius*np.ones(10), np.linspace(0, np.max(F_C_list/np.max(F_C_list)), 10), "r--")
plt.plot(ConvRadius*np.ones(10), np.linspace(0, np.max(F_C_list/np.max(F_C_list)), 10), "m--")
plt.xlabel(r"$R/R_\odot$", fontsize = 12)
plt.ylabel(r"Energy flux $\frac{F}{F_C + F_R}$", fontsize = 12)
plt.legend(loc=0)
plt.grid()

plt.subplot(grid[0, 1])
plt.plot(R_values, epsI/epsTOT, "r", label=r"$PP-I$")
plt.plot(R_values, epsII/epsTOT, "g", label=r"$PP-II$")
plt.plot(R_values, epsIII/epsTOT, "b", label=r"$PP-III$")
plt.plot(R_values, eps_array/np.max(eps_array), "m", label=r"$\epsilon/\epsilon_{max}$")
plt.plot(CoreRadius*np.ones(10), np.linspace(0, np.max(eps_array/np.max(eps_array)), 10), "r--")
plt.plot(ConvRadius*np.ones(10), np.linspace(0, np.max(eps_array/np.max(eps_array)), 10), "m--")
plt.ylabel(r"Relative energy", fontsize=12)
plt.xlabel(r"$R/R_\odot$", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc = 9, labelspacing = 0.1)


plt.subplot(grid[1, 0])
plt.plot(R_values[np.where(nabla_star>1e-3)], nabla_stab[np.where(nabla_star>1e-3)], label=r"$\nabla_{stable}$", color = "b")
plt.plot(R_values[np.where(nabla_star>1e-3)], nabla_star[np.where(nabla_star>1e-3)], linestyle="--", label=r"$\nabla^*$", color = "orange")
plt.plot(R_values[np.where(nabla_star>1e-3)], nabla_ad[np.where(nabla_star>1e-3)], label=r"$\nabla_{ad}$", color = "k")
plt.plot(CoreRadius*np.ones(10), np.linspace(0, np.max(nabla_stab), 10), "r--")
plt.plot(ConvRadius*np.ones(10), np.linspace(0, np.max(nabla_stab), 10), "m--")
plt.xlabel(r"$R/R_\odot$", fontsize = 12)
plt.ylabel(r"$\nabla$", fontsize = 12)
plt.yscale("log")
plt.legend(loc=0)
plt.grid()

plt.subplot(grid[1, 1])
plt.plot(R_values, nabla_stab, label=r"$\nabla_{stable}$", zorder = 1, color = "b")
plt.plot(R_values, nabla_star, linestyle="--", label=r"$\nabla^*$", zorder = 3, color = "orange")
plt.plot(R_values, nabla_ad, label=r"$\nabla_{ad}$", zorder = 2, color = "k")
plt.plot(ConvRadius*np.ones(10), np.linspace(0, np.max(nabla_stab), 10), "m--")
plt.xlabel(r"$R/R_\odot$", fontsize = 12)
plt.ylabel(r"$\nabla$", fontsize = 12)
#plt.yscale("symlog")
plt.xlim(0.68, 1.12)
plt.ylim(0.39, 0.42)
plt.legend(loc=9)
plt.grid()
#plt.savefig("fluxplot.jpg")

plt.subplots(3, 2, figsize=(10, 7))
plt.tight_layout(pad = 3.1)
grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.3)

plt.subplot(grid[0, 0])
plt.plot(R_values, m/1.989e30, label=r"$M(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(m/1.989e30), np.max(m/1.989e30), 10), "r--")
plt.plot(ConvRadius*np.ones(10), np.linspace(np.min(m/1.989e30), np.max(m/1.989e30), 10), "m--")
#plt.xlabel(r"$R/R_\odot$", fontsize=12)
plt.ylabel(r"$M/M_\odot$", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.legend(loc=0)

plt.subplot(grid[0, 1])
plt.semilogy(R_values, Rho*1e-3, label=r"$\rho(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(Rho*1e-3), np.max(Rho*1e-3), 10), "r--")
plt.plot(ConvRadius*np.ones(10), np.linspace(np.min(Rho*1e-3), np.max(Rho*1e-3), 10), "m--")
#plt.xlabel(r"$R/R_\odot$", fontsize=12)
plt.ylabel(r"$\rho [g cm^{-3}]$", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.legend(loc=0)

plt.subplot(grid[1, 0])
plt.semilogy(R_values, P*1e-15, label=r"$P(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(P*1e-15), np.max(P*1e-15), 10), "r--")
plt.plot(ConvRadius*np.ones(10), np.linspace(np.min(P*1e-15), np.max(P*1e-15), 10), "m--")
#plt.xlabel(r"$R/R_\odot$", fontsize=12)
plt.ylabel(r"$P [PPa]$", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.legend(loc=0)

plt.subplot(grid[1, 1])
plt.plot(R_values, L_values, label=r"$L(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(L_values), np.max(L_values), 10), "r--")
plt.plot(ConvRadius*np.ones(10), np.linspace(np.min(L_values), np.max(L_values), 10), "m--")
#plt.xlabel(r"$R/R_\odot$", fontsize=12)
plt.ylabel(r"$L/L_\odot$", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.legend(loc=0)

plt.subplot(grid[2, 0])
plt.plot(R_values, T*1e-6, label=r"$T(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(T*1e-6), np.max(T*1e-6), 10), "r--")
plt.plot(ConvRadius*np.ones(10), np.linspace(np.min(T*1e-6), np.max(T*1e-6), 10), "m--")
plt.xlabel(r"$R/R_\odot$", fontsize=12)
plt.ylabel(r"$T$[MK]", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.legend(loc=0)

show_every = 5
core_limit = 0.995

ax = plt.subplot(grid[2, 1])
plt.grid()

rmax = 1.2*R0
#ax.set_xlim(-rmax,rmax)
#ax.set_ylim(-rmax,rmax)
ax.set_aspect('equal')	# make the plot circular
j = show_every
for k in range(0, n-1):
	j += 1
	if j >= show_every:	# don't show every step - it slows things down
		if(L_values[k] > core_limit):	# outside core
			if(F_C_list[k] > 0.0):		# convection
				circR = plt.Circle((0,0),R_values[k],color='red',fill=False)
				ax.add_artist(circR)
			else:				# radiation
				circY = plt.Circle((0,0),R_values[k],color='yellow',fill=False)
				ax.add_artist(circY)
		else:				# inside core
			if(F_C_list[k] > 0.0):		# convection
				circB = plt.Circle((0,0),R_values[k],color='blue',fill = False)
				ax.add_artist(circB)
			else:				# radiation
				circC = plt.Circle((0,0),R_values[k],color='cyan',fill = False)
				ax.add_artist(circC)
		j = 0
circR = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='red',fill=True)		# These are for the legend (drawn outside the main plot)
circY = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='yellow',fill=True)
circC = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='cyan',fill=True)
circB = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='blue',fill=True)
ax.legend([circR, circY, circC, circB],
         ['Convection outside core', 'Radiation outside core', 'Radiation inside core', 'Convection inside core'],
		  loc = 0, fontsize = 10) # only add one (the last) circle of each colour to legend
plt.xlabel(r"$R/R_\odot$", fontsize = 12)
plt.ylabel(r"$R/R_\odot$", fontsize = 12)
plt.xlim(-1.2, 4.5)
plt.ylim(-1.15, 1.15)
#plt.savefig("modelbest.jpg")
plt.show()

