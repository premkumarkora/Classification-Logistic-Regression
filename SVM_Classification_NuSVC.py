#Similar to SVC but uses a parameter to control the number of support vectors.
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
clf = make_pipeline(StandardScaler(), NuSVC())
clf.fit(X, y)

print("Predicted Value ==", *clf.predict([[-0.8, -1]]))