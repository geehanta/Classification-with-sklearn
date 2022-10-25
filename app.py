from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris = datasets.load_iris()
type(iris)
print(iris.keys())
print(iris.data.shape)
print(iris.target_names)
x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
print(df.head())