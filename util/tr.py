import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.displot( np.abs(0.1*np.random.randn(10000)))
plt.show()
