# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize

# import numpy as np
# # from skimage import io
# import torch
# import random
my_set = {"23", "1992", "2189", "210"}
dictionary = {"23": 229, "1992": 2189, "210": 2190}

print(my_set)
for sample_id, value in dictionary.items():
    print(sample_id, value)
    if (value < 1000):
        my_set.remove(sample_id)

print(my_set)
# with open("../results/evaluation/GandalfTheGrayEvaluation.txt", "r") as f:

# Wo,Ar,sample_sigma,prediction,abs_error,rel_error,mse,K, G, frac = np.loadtxt("results/evaluation/EmptyEvaluation.txt", delimiter=",", skiprows=1, unpack=True)

# print(np.min(mse), "minimum value of mse")
# print(np.max(mse), "maximum value of mse")

# norm = Normalize()
# normed_vals = norm(mse)
# print(np.min(normed_vals))
# print(np.max(normed_vals))
# print(norm.inverse(np.min(mse)))
# print(norm.inverse(np.max(mse)))

# import matplotlib.pyplot as plt

# plt.scatter(Wo, Ar, c=mse, norm=Normalize(), cmap="magma")
# plt.colorbar(label="default norm")
# plt.show()

# plt.scatter(Wo, Ar, c=mse, norm="log", cmap="magma")
# plt.colorbar(label="log norm")

# plt.show()

# print(np.min(sample_sigma))

# below_zero = []

# for i, sigma in enumerate(sample_sigma):
#     if sigma <0:
#         below_zero.append([Wo[i], Ar[i], sample_sigma[i], prediction[i], mse[i], K[i], G[i], frac[i]])

# below_zero = np.array(below_zero)

# print(len(below_zero))
# print(len(sample_sigma))

# print(np.average(Wo), np.average(Ar), np.average(sample_sigma), np.average(prediction), np.average(mse), "\n", np.average(K), np.average(G), np.average(frac))
# print(np.average(below_zero, axis=0))

# # significantly different: mse, K, sigma (ofc), 
# print("below zero")
# print(below_zero[: , 5])
# print("max, min")
# print(np.max(below_zero[:, 5]))
# print(np.min(below_zero[:, 5]))
# print("quartiles (25, 50, 75)")
# print(np.quantile(below_zero[:, 5], .25))
# print(np.quantile(below_zero[:, 5], .50))
# print(np.quantile(below_zero[:, 5], .75))


# below_zero = []

# for i, k in enumerate(K):
#     if k <49.8 and k > 5.06:
#         below_zero.append([Wo[i], Ar[i], sample_sigma[i], prediction[i], mse[i], K[i], G[i], frac[i]])

# below_zero = np.array(below_zero)

# print(len(below_zero))
# print(len(sample_sigma))

# print(np.average(Wo), np.average(Ar), np.average(sample_sigma), np.average(prediction), np.average(mse), "\n", np.average(K), np.average(G), np.average(frac))
# print(np.average(below_zero, axis=0))

# # significantly different: mse, K, sigma (ofc), 
# print("below zero")
# print(below_zero[: , 5])
# print("max, min")
# print(np.max(below_zero[:, 5]))
# print(np.min(below_zero[:, 5]))
# print("quartiles (25, 50, 75)")
# print(np.quantile(below_zero[:, 5], .25))
# print(np.quantile(below_zero[:, 5], .50))
# print(np.quantile(below_zero[:, 5], .75))

# print()
# print()
# print("data collection:")
# print("Average Relative Error:", np.average(rel_error))
# print("Average Absolute Error:", np.average(abs_error))
# print("Average MSE:", np.average(mse))