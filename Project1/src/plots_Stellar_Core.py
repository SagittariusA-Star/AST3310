import numpy as np
import matplotlib.pyplot as plt
from Stellar_Core import StellarCore as SC
import matplotlib.gridspec as gridspec
"""This script is used to iterate over different initial values
   as well as plotting the result"""

L_sun   = 3.846e26                          # [W]                Solar luminocity
M_sun   = 1.989e30                          # [kg]               Solar mass
R_sun   = 6.96e8                            # [m]                Solar radius
rho_sun = M_sun/(4*np.pi/3*(R_sun)**3)      # [kg m^-3]          Mean solar density

"""Initial values"""
R0   = 0.72*R_sun
L0   = L_sun
T0   = 5.7e6
rho0 = 5.1*rho_sun
M0   = 0.8*M_sun
Core = SC(R0, L0, T0, rho0, M0)
P0   = Core.P(rho0, T0)

"""Arrays of different initial values"""
Cha         = np.linspace(0.2, 5, 5)        # Change from initial value
R0_array    = np.ones_like(Cha)*R0*Cha      # Initial radii tested
T0_array    = np.ones_like(Cha)*T0*Cha      # Initial temperatures tested
rho0_array  = np.ones_like(Cha)*rho0*Cha    # Initial densitied tested

"""Creating arrays of R, T, rho, L and P as well as m values to plot
   for different initial radii values. Making use of executable strings."""

for i in range(5):
    String1 = "CoreRad{0} = SC(R0_array[i], L0, T0, rho0, M0)".format(i)
    String2 = "mRad{0}, yRad{1}  = CoreRad{2}.solve((M0, 0), 0.001, 1e-4, variable_step=True, debug=False)".format(i, i, i)
    String3 = "RRad{0}, PRad{1}, LRad{2}, TRad{3} = yRad{4}".format(i, i, i, i, i)
    String4 = "RhoRad{0} = CoreRad{1}.rho(PRad{2}, TRad{3})".format(i, i, i, i)
    exec(String1)    # Executing strings
    exec(String2)    # -------||-------
    exec(String3)    # -------||-------
    exec(String4)    # -------||-------

"""Ploting generated arrays"""

plt.figure(figsize=(9, 7))
plt.tight_layout(pad=2.5)

gs = gridspec.GridSpec(3, 2)
ax1 = plt.subplot(gs[0,0])
plt.title(r"$R(M)$")
plt.plot(mRad0/M_sun, RRad0/R_sun)
plt.plot(mRad1/M_sun, RRad1/R_sun)
plt.plot(mRad2/M_sun, RRad2/R_sun)
plt.plot(mRad3/M_sun, RRad3/R_sun)
plt.plot(mRad4/M_sun, RRad4/R_sun)
ax1.set_xticklabels([])
plt.yticks(fontsize=14)
plt.ylabel(r"$R/R_\odot$", fontsize=14)

ax2 = plt.subplot(gs[0,1])
plt.title(r"$\rho(M)$")
plt.plot(mRad0/M_sun, RhoRad0*1e-3)
plt.plot(mRad1/M_sun, RhoRad1*1e-3)
plt.plot(mRad2/M_sun, RhoRad2*1e-3)
plt.plot(mRad3/M_sun, RhoRad3*1e-3)
plt.plot(mRad4/M_sun, RhoRad4*1e-3)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.yticks(fontsize=14)
ax2.set_xticklabels([])
plt.ylabel(r"$\rho$ [g cm$^{-3}$]", fontsize=14)

ax3 = plt.subplot(gs[1,0])
plt.title(r"$L(M)$")
plt.plot(mRad0/M_sun, LRad0/L_sun)
plt.plot(mRad1/M_sun, LRad1/L_sun)
plt.plot(mRad2/M_sun, LRad2/L_sun)
plt.plot(mRad3/M_sun, LRad3/L_sun)
plt.plot(mRad4/M_sun, LRad4/L_sun)
plt.yticks(fontsize=14)
ax3.set_xticklabels([])
plt.ylabel(r"$L/L_\odot$", fontsize=14)

ax4 = plt.subplot(gs[1,1])
plt.title(r"$T(M)$")
plt.plot(mRad0/M_sun, TRad0*1e-6)
plt.plot(mRad1/M_sun, TRad1*1e-6)
plt.plot(mRad2/M_sun, TRad2*1e-6)
plt.plot(mRad3/M_sun, TRad3*1e-6)
plt.plot(mRad4/M_sun, TRad4*1e-6)
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
plt.xlabel(r"$M/M_\odot$", fontsize=14)
plt.ylabel(r"$T$[MK]", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


ax5 = plt.subplot(gs[2,0])
plt.title(r"$P(M)$")
plt.plot(mRad0/M_sun, PRad0*1e-15, label=r"$R_0 = {:.2g}R_\odot$".format(Cha[0]*R0/R_sun))
plt.plot(mRad1/M_sun, PRad1*1e-15, label=r"$R_0 = {:.2g}R_\odot$".format(Cha[1]*R0/R_sun))
plt.plot(mRad2/M_sun, PRad2*1e-15, label=r"$R_0 = {:.2g}R_\odot$".format(Cha[2]*R0/R_sun))
plt.plot(mRad3/M_sun, PRad3*1e-15, label=r"$R_0 = {:.2g}R_\odot$".format(Cha[3]*R0/R_sun))
plt.plot(mRad4/M_sun, PRad4*1e-15, label=r"$R_0 = {:.2g}R_\odot$".format(Cha[4]*R0/R_sun))
plt.xlabel(r"$M/M_\odot$", fontsize=14)
plt.ylabel(r"$P$[PPa]", fontsize=14)
box = ax5.get_position()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax5.legend(loc='center left', bbox_to_anchor=(1.4, 0.3), fontsize=14)
plt.savefig("RadPlot.jpg")

"""Creating arrays of R, T, rho, L and P as well as m values to plot
   for different initial temperatures values. Making use of executable strings."""

for i in range(5):
    String1 = "CoreTemp{0} = SC(R0, L0, T0_array[i], rho0, M0)".format(i)
    String2 = "mTemp{0}, yTemp{1}  = CoreTemp{1}.solve((M0, 0), 0.001, 1e-5, variable_step=True, debug=False)".format(
        i, i, i)
    String3 = "RTemp{0}, PTemp{1}, LTemp{2}, TTemp{3} = yTemp{4}".format(i, i, i, i, i)
    String4 = "RhoTemp{0} = CoreTemp{1}.rho(PTemp{2}, TTemp{3})".format(i, i, i, i)
    exec(String1)
    exec(String2)
    exec(String3)
    exec(String4)

"""Ploting generated arrays"""

plt.figure(figsize=(9, 7))
plt.tight_layout(pad=2.5)

gs = gridspec.GridSpec(3, 2)

ax1 = plt.subplot(gs[0,0])
plt.title(r"$R(M)$")
plt.plot(mTemp0/M_sun, RTemp0/R_sun)
plt.plot(mTemp1/M_sun, RTemp1/R_sun)
plt.plot(mTemp2/M_sun, RTemp2/R_sun)
plt.plot(mTemp3/M_sun, RTemp3/R_sun)
plt.plot(mTemp4/M_sun, RTemp4/R_sun)
ax1.set_xticklabels([])
plt.ylabel(r"$R/R_\odot$", fontsize=14)

ax2 = plt.subplot(gs[0,1])
plt.title(r"$\rho(M)$")
plt.plot(mTemp0/M_sun, RhoTemp0*1e-3)
plt.plot(mTemp1/M_sun, RhoTemp1*1e-3)
plt.plot(mTemp2/M_sun, RhoTemp2*1e-3)
plt.plot(mTemp3/M_sun, RhoTemp3*1e-3)
plt.plot(mTemp4/M_sun, RhoTemp4*1e-3)
plt.ylabel(r"$\rho$ [g cm$^{-3}$]", fontsize=14)
plt.yticks(fontsize=14)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_xticklabels([])

ax3 = plt.subplot(gs[1,0])
plt.title(r"$L(M)$")
plt.plot(mTemp0/M_sun, LTemp0/L_sun)
plt.plot(mTemp1/M_sun, LTemp1/L_sun)
plt.plot(mTemp2/M_sun, LTemp2/L_sun)
plt.plot(mTemp3/M_sun, LTemp3/L_sun)
plt.plot(mTemp4/M_sun, LTemp4/L_sun)
ax3.set_xticklabels([])
plt.ylabel(r"$L/L_\odot$", fontsize=14)
plt.yticks(fontsize=14)

ax4 = plt.subplot(gs[1,1])
plt.title(r"$T(M)$")
plt.plot(mTemp0/M_sun, TTemp0*1e-6)
plt.plot(mTemp1/M_sun, TTemp1*1e-6)
plt.plot(mTemp2/M_sun, TTemp2*1e-6)
plt.plot(mTemp3/M_sun, TTemp3*1e-6)
plt.plot(mTemp4/M_sun, TTemp4*1e-6)
plt.ylabel(r"$T$[MK]", fontsize=14)
plt.xlabel(r"$M/M_\odot$", fontsize=14)
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax5 = plt.subplot(gs[2,0])
plt.title(r"$P(M)$")
plt.plot(mTemp0/M_sun, PTemp0*1e-15, label=r"$T_0 = {:.2g}$MK".format(Cha[0]*T0*1e-6))
plt.plot(mTemp1/M_sun, PTemp1*1e-15, label=r"$T_0 = {:.2g}$MK".format(Cha[1]*T0*1e-6))
plt.plot(mTemp2/M_sun, PTemp2*1e-15, label=r"$T_0 = {:.2g}$MK".format(Cha[2]*T0*1e-6))
plt.plot(mTemp3/M_sun, PTemp3*1e-15, label=r"$T_0 = {:.2g}$MK".format(Cha[3]*T0*1e-6))
plt.plot(mTemp4/M_sun, PTemp4*1e-15, label=r"$T_0 = {:.2g}$MK".format(Cha[4]*T0*1e-6))
plt.xlabel(r"$M/M_\odot$", fontsize=14)
plt.ylabel(r"$P$[PPa]", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
box = ax5.get_position()
ax5.legend(loc='center left', bbox_to_anchor=(1.4, 0.3), fontsize=14)
plt.savefig("TempPlot.jpg")

"""Creating arrays of R, T, rho, L and P as well as m values to plot
   for different initial densities values. Making use of executable strings."""
   
for i in range(5):
    String1 = "Corerho{0} = SC(R0, L0, T0, rho0_array[i], M0)".format(i)
    String2 = "mrho{0}, yrho{1}  = Corerho{2}.solve((M0, 0), 0.001, 1e-4, variable_step=True, debug=False)".format(i, i, i)
    String3 = "Rrho{0}, Prho{1}, Lrho{2}, Trho{3} = yrho{4}".format(i, i, i, i, i)
    String4 = "Rhorho{0} = Corerho{1}.rho(Prho{2}, Trho{3})".format(i, i, i, i)
    exec(String1)
    exec(String2)
    exec(String3)
    exec(String4)

"""Ploting generated arrays"""

plt.figure(figsize=(9, 7))
plt.tight_layout(pad=2.5)

gs = gridspec.GridSpec(3, 2)

ax1 = plt.subplot(gs[0,0])
plt.title(r"$R(M)$")
plt.plot(mrho0/M_sun, Rrho0/R_sun)
plt.plot(mrho1/M_sun, Rrho1/R_sun)
plt.plot(mrho2/M_sun, Rrho2/R_sun)
plt.plot(mrho3/M_sun, Rrho3/R_sun)
plt.plot(mrho4/M_sun, Rrho4/R_sun)
ax1.set_xticklabels([])
plt.yticks(fontsize = 14)
plt.ylabel(r"$R/R_\odot$", fontsize=14)

ax2 = plt.subplot(gs[0,1])
plt.title(r"$\rho(M)$")
plt.plot(mrho0/M_sun, Rhorho0*1e-3)
plt.plot(mrho1/M_sun, Rhorho1*1e-3)
plt.plot(mrho2/M_sun, Rhorho2*1e-3)
plt.plot(mrho3/M_sun, Rhorho3*1e-3)
plt.plot(mrho4/M_sun, Rhorho4*1e-3)
plt.yticks(fontsize=14)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_xticklabels([])
plt.ylabel(r"$\rho$ [g cm$^{-3}$]", fontsize=14)

ax3 = plt.subplot(gs[1,0])
plt.title(r"$L(M)$")
plt.plot(mrho0/M_sun, Lrho0/L_sun)
plt.plot(mrho1/M_sun, Lrho1/L_sun)
plt.plot(mrho2/M_sun, Lrho2/L_sun)
plt.plot(mrho3/M_sun, Lrho3/L_sun)
plt.plot(mrho4/M_sun, Lrho4/L_sun)
plt.yticks(fontsize=14)
ax3.set_xticklabels([])
plt.ylabel(r"$L/L_\odot$", fontsize=14)

ax4 = plt.subplot(gs[1,1])
plt.title(r"$T(M)$")
plt.plot(mrho0/M_sun, Trho0*1e-6)
plt.plot(mrho1/M_sun, Trho1*1e-6)
plt.plot(mrho2/M_sun, Trho2*1e-6)
plt.plot(mrho3/M_sun, Trho3*1e-6)
plt.plot(mrho4/M_sun, Trho4*1e-6)
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r"$M/M_\odot$", fontsize=14)
plt.ylabel(r"$T$[MK]", fontsize=14)

ax5 = plt.subplot(gs[2,0])
plt.title(r"$P(M)$")
plt.plot(mrho0/M_sun, Prho0*1e-15, label=r"$\rho_0 = %.2g$ g cm$^{-3}$" %(Cha[0]*rho0/rho_sun))
plt.plot(mrho1/M_sun, Prho1*1e-15, label=r"$\rho_0 = %.2g$ g cm$^{-3}$" %(Cha[1]*rho0/rho_sun))
plt.plot(mrho2/M_sun, Prho2*1e-15, label=r"$\rho_0 = %.2g$ g cm$^{-3}$" %(Cha[2]*rho0/rho_sun))
plt.plot(mrho3/M_sun, Prho3*1e-15, label=r"$\rho_0 = %.2g$ g cm$^{-3}$" %(Cha[3]*rho0/rho_sun))
plt.plot(mrho4/M_sun, Prho4*1e-15, label=r"$\rho_0 = %.2g$ g cm$^{-3}$" %(Cha[4]*rho0/rho_sun))
plt.xlabel(r"$M/M_\odot$", fontsize=14)
plt.ylabel(r"$P$[PPa]", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
box = ax5.get_position()
ax5.legend(loc='center left', bbox_to_anchor=(1.4, 0.28), fontsize = 14)
plt.savefig("rhoPlot.jpg")
plt.show()

