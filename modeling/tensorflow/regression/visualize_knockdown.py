import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

knockdown=np.load('../processed_data/regression/knockdown_prediction.npy')
knockdown=pd.DataFrame(data=knockdown,columns=['y_','y','UID','experiment','knockdown'])

plt.hist(knockdown.loc[knockdown.UID=='YPR204W','y'])
plt.show()