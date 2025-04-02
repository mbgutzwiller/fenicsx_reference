import numpy as np
import matplotlib.pyplot as plt

grid_sizes = [50, 100, 200, 400, 800, 1600]
runtimes_16 = [1.0139906406402588, 4.019821643829346, 9.95474910736084, 37.207207918167114, 247.35333466529846, 2054.6111330986023]#, 15661.0]
runtimes_8 = [0.37127208709716797, 1.326030969619751, 4.8523712158203125, 31.24010705947876, 241.40032076835632, 2073.247344017029]#, 15941.0]
runtimes_4 = [0.3799583911895752, 1.332439661026001, 6.719934701919556, 44.8723349571228, 381.9476659297943, 3104.8]

_index = -1
C_loglog_3 = runtimes_16[_index] / (grid_sizes[_index])**3  # reference line for 2nd order convergence
C_loglog_4 = runtimes_16[_index] / (grid_sizes[_index])**4  # reference line for 2nd order convergence
C_loglog_2 = runtimes_4[0] / (grid_sizes[0])**2  # reference line for 2nd order convergence


plt.figure(figsize=(8, 5))
plt.loglog(grid_sizes, runtimes_16, 'k--o', markersize=8, label="16 Cores", color="green")
plt.loglog(grid_sizes, runtimes_8, 'k--o', markersize=8,label="8 Cores", color="blue")
plt.loglog(grid_sizes, runtimes_4, 'k--o', markersize=8,label="4 Cores", color="purple")

plt.loglog(grid_sizes, C_loglog_2 * np.array(grid_sizes)**2, "--", label="Slope = 1", alpha=0.5, color="black")
plt.loglog(grid_sizes, C_loglog_3 * np.array(grid_sizes)**3, label="Slope = 3", alpha=0.5, color="black")
# plt.loglog(grid_sizes, C_loglog_4 * np.array(grid_sizes)**4, ":", label="Slope = 4", alpha=0.5, color="black")

plt.ylabel("Compute Time [s]")
plt.xlabel("Grid size [-]")
plt.legend()
plt.grid(True, which="both")
plt.savefig(f"/home/merrillg/FEM_reference/scaling_fenicsx", dpi=600)

plt.show()

# a = 3 * 3600
# b = 51*60 + 34 * 60
# c = 34 + 7
# print(a + b + c)

