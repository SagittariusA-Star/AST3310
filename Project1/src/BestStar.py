import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Stellar_Core import StellarCore as SC

R0          = 0.72*6.96e8                       # [m] Initial radius
L0          = 3.846e26                          # [W]  Initial luminocity
T0          = 5.7e6                             # [K]  Initial temperature
rho0        = (5.1*(1.989e30
               /(4*np.pi/3*(6.96e8)**3)))       # [kg m^-3] Initial density
M0          = 0.8*1.989e30                      # [kg] Initial mass
N           = 50                                # Loop lengt
change      = np.linspace( 0.2, 2, N)           # Percentwise change from:
R_array     = np.ones_like(change)*change*R0    # - R_0
rho_array   = np.ones_like(change)*change*rho0  # - rho_0
L_grid_vals = np.zeros(shape = (N, N))          # Luminocity values in R-rho grid
R_grid_vals = np.zeros_like(L_grid_vals)        # Radius values in R-rho grid
m_grid_vals = np.zeros_like(L_grid_vals)        # Mass values in R-rho grid

"""Testing different initial values for radius and density
   and given inital temperature in a double for loop. Saving all 
   last values of generated luminocity, mass and 
   radius arrays in their respective matrix. Adding together these three
   matrices in dimensionless form to generate color- and 3D plots,
   as well as finding the initial values for R and rho that make
   L, R and m approach zero the closest."""

"""
for i, r in enumerate(R_array):
        for j, rho in enumerate(rho_array):
            CoreBest = SC(r, L0, T0, rho, M0)
            m, y = CoreBest.solve((M0, 0), 0.001, 1e-4,
                                    variable_step=True, debug=False)
            R, P, L, T = y
            L_grid_vals[i, j] = L[-1]/L0
            R_grid_vals[i, j] = R[-1]/r
            m_grid_vals[i, j] = m[-1]/M0
combined_grid_vals = np.add(L_grid_vals, R_grid_vals)
combined_grid_vals = np.add(combined_grid_vals, m_grid_vals)
"""
"""Saving matrices so double loop
   only needs to be run one time"""
#np.save("Lgrid.npy", L_grid_vals)           
#np.save("Rgiid.npy", R_grid_vals)
#np.save("mgrid.npy", m_grid_vals)
#np.save("combgrid.npy", combined_grid_vals)
"""Loading saved matrices"""

L_grid_vals        = np.load("Lgrid.npy")       # Luminocity values in R-rho grid
R_grid_vals        = np.load("Rgrid.npy")       # Radius values in R-rho grid
m_grid_vals        = np.load("mgrid.npy")       # Mass values in R-rho grid
combined_grid_vals = np.load("combgrid.npy")    # L + R + m values in R-rho grid (unitless)

best_pos           = np.where(combined_grid_vals == np.min(combined_grid_vals))  # Position of best initial conditions
R0_best            = R_array[best_pos[0]]       # Best initial radius
rho0_best          = rho_array[best_pos[1]]     # Best initial density

"""Creating best star to plot"""

CoreBest = SC(R0_best, L0, T0, rho0_best, M0)
m, y = CoreBest.solve((M0, 0), 0.001, 1e-4,
                      variable_step=True, debug=False)
R, P, L, T = y                          # Function values
Rho = CoreBest.rho(P, T)
eps = np.zeros_like(T)

CoreRadius = R[np.where(L/L0 <= 0.995)] # Stellar Core Radius
CoreRadius = CoreRadius[0]/6.96e8

for i, t in enumerate(T):
        eps[i] = CoreBest.epsilon(T[i], Rho[i]) #Energy production for best star


print(np.min(combined_grid_vals))
print(best_pos)
print("Best R: ", R0_best, change[best_pos[0]])
print("Best rho: ", rho0_best, change[best_pos[1]])
print("Rel. Error R: ", 100*(1 - abs(R[-1] - R0)/R0), "%")
print("Rel. Error L: ", 100*(1 - abs(L[-1] - L0)/L0), "%")
print("Rel. Error m: ", 100*(1 - abs(m[-1] - M0)/M0), "%")

"""Plotting L, m, epsilon, P and rho for best initial values
   as a function of radius"""
plt.subplots(3, 2, figsize=(9, 7))
plt.tight_layout(pad = 3.5)

plt.subplot( 3, 2, 1)
plt.plot(R/6.96e8, m/1.989e30, label=r"$M(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(m/1.989e30), np.max(m/1.989e30), 10), "r--")
plt.ylabel(r"$M/M_\odot$", fontsize=16)
plt.xlabel(r"$R/R_\odot$", fontsize=16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(loc=0)

plt.subplot(3, 2, 2)
plt.semilogy(R/6.96e8, Rho*1e-3, label=r"$\rho(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(Rho*1e-3), np.max(Rho*1e-3), 10), "r--")
plt.xlabel(r"$R/R_\odot$", fontsize=16)
plt.ylabel(r"$\rho [g cm^{-3}]$", fontsize=16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(loc=0)

plt.subplot(3, 2, 3)
plt.semilogy(R/6.96e8, P*1e-15, label=r"$P(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(P*1e-15), np.max(P*1e-15), 10), "r--")
plt.xlabel(r"$R/R_\odot$", fontsize=16)
plt.ylabel(r"$P [PPa]$", fontsize=16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(loc=0)

plt.subplot(3, 2, 4)
plt.plot(R/6.96e8, L/L0, label=r"$L(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(L/L0), np.max(L/L0), 10), "r--")
plt.xlabel(r"$R/R_\odot$", fontsize=16)
plt.ylabel(r"$L/L_\odot$", fontsize=16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(loc=0)

plt.subplot(3, 2, 5)
plt.plot(R/6.96e8, T*1e-6, label=r"$T(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(T*1e-6), np.max(T*1e-6), 10), "r--")
plt.xlabel(r"$R/R_\odot$", fontsize=16)
plt.ylabel(r"$T$[MK]", fontsize=16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(loc=0)

plt.subplot(3, 2, 6)
plt.semilogy(R/6.96e8, eps*1e-6, label=r"$\epsilon(R)$")
plt.plot(CoreRadius*np.ones(10), np.linspace(np.min(eps*1e-6), np.max(eps*1e-6), 10), "r--")
plt.ylabel(r"$\epsilon$ [MJ kg$^{-1}$]", fontsize=16)
plt.xlabel(r"$R/R_\odot$", fontsize=16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(loc=0)
#plt.savefig("BestStar.jpg")

"""Ploting color- and 3D plots for different initial values.
   The generated combined (L, R and m added on dimentionless form) 
   surface is used to find initial values R and rho that make 
   the surface approach zero the closest."""


fig, ax = plt.subplots(1, 4, figsize=(16, 4))
plt.tight_layout(pad = 3.5)

X, Y = np.meshgrid(rho_array/rho0, R_array/R0)
ax1 = plt.subplot(1, 4, 1)
plt.title(r"$L_{last}/L_\odot$")
plt.contourf(X, Y, L_grid_vals)
plt.xlabel(r"$\rho_{init}/\rho_0$", fontsize=12)
plt.ylabel(r"$R_{init}/R_0$", fontsize=12)
plt.xticks(np.arange(0.2, 2.2, 0.2), rotation=60, fontsize=12)
plt.yticks(fontsize=14)

ax2 = plt.subplot(1, 4, 2)
plt.contourf(X, Y, R_grid_vals)
plt.title(r"$R_{last}/R_{init}$")
plt.xlabel(r"$\rho_{init}/\rho_0$", fontsize=12)
plt.xticks(np.arange(0.2, 2.2, 0.2), rotation=60, fontsize=12)
ax2.set_yticklabels([])

ax3 = plt.subplot(1, 4, 3)
plt.contourf(X, Y, m_grid_vals)
plt.title(r"$m_{last}/M_0$")
plt.xlabel(r"$\rho_{init}/\rho_0$", fontsize=12)
plt.xticks(np.arange(0.2, 2.2, 0.2), rotation=60, fontsize=12)
ax3.set_yticklabels([])

ax4 = plt.subplot(1, 4, 4)
plt.contourf(X, Y, combined_grid_vals)
plt.title(r"$L_{last}/L_\odot + R_{last}/R_{init} + m_{last}/M_\odot$")
plt.xlabel(r"$\rho_{init}/\rho_0$", fontsize=12)
plt.xticks(np.arange(0.2, 2.2, 0.2), rotation=60, fontsize=12)
ax4.set_yticklabels([])
cax = fig.add_axes([0.95, 0.18, 0.02, 0.7])
cb = plt.colorbar(pad = 0.2, ax = [ax1, ax2, ax3, ax4], cax = cax)

plt.subplots_adjust(wspace = 0.03)
box = ax4.get_position()
ax4.set_position([box.x0, box.y0, box.width*0.9, box.height])
#plt.savefig("combined_contour.jpg")

fig = plt.figure(figsize=(9, 7))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, combined_grid_vals, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf)
ax.set_xlabel(r"$R_{last}/R_{init}$")
ax.set_ylabel(r"$\rho_{last}/\rho_{init}$")
ax.set_zlabel(r"$L_{last}/L_\odot + R_{last}/R_{init} + m_{last}/M_0$")
ax.view_init(30, 200)
#plt.savefig("BestStar3D.jpg")
plt.show()

