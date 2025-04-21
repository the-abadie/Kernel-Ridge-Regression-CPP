import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import sys

outpath = str(sys.argv[1])
file = open(outpath + "/MAEs.txt", 'r').readlines()

nSigma  = len(file)
nLambda = len(file[0].split())

maes = np.zeros((nSigma, nLambda))

for line in range(len(file)):
    x = file[line].split()

    maes[line] = np.array(x)


plt.imshow(maes, cmap="jet", interpolation="none", norm=colors.LogNorm())
clb = plt.colorbar()
clb.ax.set_ylabel("MAE", rotation=0, labelpad=10)
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\sigma$")
plt.xlim(-0.5, nLambda-0.5)
plt.ylim(-0.5, nSigma-0.5)
plt.gca().set_aspect('equal')
plt.title(f"Validation Error wrt Hyperparameters ({nLambda} x {nSigma})")
plt.savefig(outpath + "/MAEs.png", dpi=150, bbox_inches="tight")

plt.close()

file = open(outpath + "Eref_Eest.txt", 'r').readlines()

E_tst = np.array(file[0].split(), dtype=float)
E_est = np.array(file[1].split(), dtype=float)


p = np.polyfit (E_tst, E_est, 1)
c = np.linspace(-2500, -500, 10)

plt.scatter(x=E_tst, y=E_est, s=4)
plt.plot([-2500, -500], [-2500, -500], color="black", linestyle=":", zorder=10, label=r"$y=x$")
plt.plot(c, p[0]*c + p[1], color="red", linestyle=":", label="Fit")


plt.xlim(-2750, -250)
plt.ylim(-2750, -250)
plt.xlabel(r"$E_\text{PBE0}$")
plt.ylabel(r"$E_\text{est}$")
plt.title("Data vs. Prediction Using Coulomb Matrix Eigenvalues")
plt.legend()

plt.savefig(outpath + "/err.png", dpi=150, bbox_inches="tight")
plt.close()