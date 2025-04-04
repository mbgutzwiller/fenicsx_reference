import numpy as np
import matplotlib.pyplot as plt

# grid_sizes = [50, 100, 200, 400, 800, 1600]
# runtimes_16 = [1.0139906406402588, 4.019821643829346, 9.95474910736084, 37.207207918167114, 247.35333466529846, 2054.6111330986023]#, 15661.0]
# runtimes_8 = [0.37127208709716797, 1.326030969619751, 4.8523712158203125, 31.24010705947876, 241.40032076835632, 2073.247344017029]#, 15941.0]
# runtimes_4 = [0.3799583911895752, 1.332439661026001, 6.719934701919556, 44.8723349571228, 381.9476659297943, 3104.8]

# grid_sizes = [10, 20, 40, 80, 120, 160, 240, 320, 800]
# runtimes_16 = [6.779041290283203, 12.498822212219238, 23.17623209953308, 47.404714822769165, 67.85655760765076, 98.27060437202454, 150.00290513038635, 200.77882838249207, 730.7381122112274]
# runtimes_8 = [0.11923980712890625, 0.14372730255126953, 0.31487035751342773, 1.0421226024627686, 2.281271457672119, 3.94570255279541, 10.414150476455688, 22.740732192993164, 295.5821545124054]
# runtimes_4 = [0.05289721488952637, 0.1142423152923584, 0.2848389148712158, 0.8263530731201172, 2.2762508392333984, 3.95065975189209, 10.736685991287231, 24.061909437179565, 392.0569896697998]
# # runtimes_4 = [392.0569896697998, 24.783448696136475, 10.869752645492554, 3.8268392086029053, 1.9854729175567627, 0.8879952430725098, 0.38422060012817383, 0.11449027061462402, 0.05503129959106445][::-1]
# runtimes_2 = [0.05579066276550293, 0.09388446807861328, 0.2580840587615967, 0.9472675323486328, 2.638576030731201, 5.659003496170044, 16.6921808719635, 39.84356498718262, 662.3824572563171]

[10, 20, 40, 80, 120, 160, 240, 320, 640, 1280]

runtimes_1 = [5245.4585609436035, 539.9593379497528, 68.29340410232544, 29.119383573532104, 8.83642292022705, 3.857235908508301, 1.3382577896118164, 0.25045204162597656, 0.07312464714050293, 0.02946925163269043]


_index = -1
C_loglog_3 = runtimes_4[_index] / (grid_sizes[_index])**3  # reference line for 2nd order convergence
C_loglog_4 = runtimes_16[_index] / (grid_sizes[_index])**4  # reference line for 2nd order convergence
C_loglog_1 = runtimes_16[0] / (grid_sizes[0])  # reference line for 2nd order convergence


plt.figure(figsize=(8, 5))
plt.loglog(grid_sizes, runtimes_16, 'k--o', markersize=8, label="16 Cores", color="green")
plt.loglog(grid_sizes, runtimes_8, 'k--o', markersize=8,label="8 Cores", color="blue")
plt.loglog(grid_sizes, runtimes_4, 'k--o', markersize=8,label="4 Cores", color="purple")
plt.loglog(grid_sizes, runtimes_2, 'k--o', markersize=8,label="2 Cores", color="cyan")

plt.loglog(grid_sizes, C_loglog_1 * np.array(grid_sizes), "--", label="Slope = 1", alpha=0.5, color="black")
plt.loglog(grid_sizes, C_loglog_3 * np.array(grid_sizes)**3, label="Slope = 3", alpha=0.5, color="black")
# plt.loglog(grid_sizes, C_loglog_4 * np.array(grid_sizes)**4, ":", label="Slope = 4", alpha=0.5, color="black")

plt.ylabel("Compute Time [s]")
plt.xlabel("Grid size [-]")
plt.legend()
plt.grid(True, which="both")
plt.savefig(f"/home/merrillg/FEM_reference/scaling_fenicsx_for_conv", dpi=600)

plt.show()

# a = 3 * 3600
# b = 51*60 + 34 * 60
# c = 34 + 7
# print(a + b + c)

