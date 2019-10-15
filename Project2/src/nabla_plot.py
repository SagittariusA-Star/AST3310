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

Sun 						= Star(R0, L0, T0, rho0, M0, sanity_debug=False)
m, y 						= Sun.solve((M0, 0), 0.01, 1e-4, variable_step=True, debug=False)
R_values, P, L_values, T	= y
Rho 						= Sun.rho(P, T)
n 							= len(m)
F_C_list, F_R_list 			= Sun.FC_list

epsI 		= np.zeros_like(m)
epsII 		= np.zeros_like(m)
epsIII 		= np.zeros_like(m)
epsTOT 		= np.zeros_like(m)
eps_array 	= np.zeros_like(m)
nabla_stab	= np.zeros_like(m)
nabla_star	= np.zeros_like(m)
nabla_ad	= np.ones_like(m)*Sun.nabla_ad
kap         = np.zeros_like(m)

for i in range(len(m)):
    # Energy production for best star
    T9 				= T*1e-9
    eI, eII, eIII 	= Sun.PPchain(T9[i], Rho[i])
    epsI[i] 		= eI
    epsII[i] 		= eII
    epsIII[i] 		= eIII
    epsTOT[i] 		= eI + eII + eIII
    eps_array[i] 	= Sun.epsilon(T9[i], Rho[i])
    nabla_stab[i]		= Sun.nabla_stable(Rho[i], T[i], R_values[i], m[i], L_values[i])
    nabla_star[i]		= Sun.nabla_star(Rho[i], T[i], R_values[i], m[i], L_values[i])
    T6 = T[i]*1e-6
    logR = np.log10(Rho[i]) - 3 - 3*np.log10(T6)   
    logT = np.log10(T[i])
    kap[i] = 10**(Sun.kappa(logR, logT))*1e-1

A = Rho*L_values*kap/(R_values**2*T**3*Sun.g(R_values, m))
B = 128*np.pi*Sun.sigma*Sun.my*Sun.mu/(15*Sun.kB)*np.ones_like(m)

plt.figure()
plt.plot(R_values/R_sun, A, label = r"$\frac{\rho}{r^2T^3}$")
plt.plot(R_values/R_sun, B, label = r"$\frac{128\pi\sigma}{15L\kappa}$")
plt.xlabel(r"$R/R_\odot$")
plt.yscale("log")
plt.legend(loc=0)
plt.grid()
plt.show()
