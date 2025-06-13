import matplotlib.pyplot as plt
import numpy as np
from skimage import io

# Fixing random state for reproducibility
np.random.seed(19680801)


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# fig = plt.figure()
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
n=2467
img = io.imread(f"data/test_images/{n:04d}.png")
plt.imshow(img)
plt.show()