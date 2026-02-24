import numpy as np
import matplotlib.pyplot as plt

a = np.load("T100.npy")
b = np.load("T250.npy")
c = np.load("250_v2.npy")
d = np.load("T500.npy")

plt.figure(figsize=(6, 4))
plt.boxplot([a, b, c, d], labels=["T100", "250", "250v2", "t500"])
plt.ylabel("Value")
plt.title("Boxplot from NPY files")
plt.tight_layout()
plt.show()
