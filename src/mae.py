import matplotlib.pyplot as plt
import numpy as np
import os

loc = os.path.dirname(os.path.realpath(__file__))

file = open(f"../MAE.txt", 'r').readlines()

maes = np.zeros((10,10))

for line in range(len(file)):
    x = file[line].split()

    maes[line] = np.array(x)

plt.imshow(maes, cmap="jet", interpolation="none")
plt.colorbar()

plt.savefig(f"{loc}/MAE.png", dpi=150)
plt.close()

file = open(f"../energies.txt", 'r').readlines()

E_tst = np.array(file[0].split(), dtype=float)
E_est = np.array(file[1].split(), dtype=float)

print(E_tst)
print(E_est)
plt.scatter(x=E_tst, y=E_est, s=4)
plt.plot([-2500, -500], [-2500, -500], color="black", linestyle=":", zorder=10)
plt.xlim(-2750, -250)
plt.ylim(-2750, -250)
plt.xlabel(r"$E_\text{PBE0}$")
plt.ylabel(r"$E_\text{est}$")
plt.title("Data vs. Prediction Using Coulomb Matrix Eigenvalues")
plt.savefig(f"{loc}/err.png", dpi=150)