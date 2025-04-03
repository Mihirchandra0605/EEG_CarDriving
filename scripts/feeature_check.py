import numpy as np 
import matplotlib.pyplot as plt   
import seaborn as sns  

# Load a feature file
features = np.load("../data/features/S001_R01_features.npy")  

# Print the shape and first few values
print("Shape of features:", features.shape)  
print("First few rows:", features[:5])  



plt.hist(features.flatten(), bins=50)  
plt.title("Feature Value Distribution")  
plt.xlabel("Feature Values")  
plt.ylabel("Frequency")  
plt.show()



sns.heatmap(np.corrcoef(features.T), cmap="coolwarm", annot=False)  
plt.title("Feature Correlation Heatmap")  
plt.show()

