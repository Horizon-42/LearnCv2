import numpy as np

# matrix det
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"A:{A}")
det_A = np.linalg.det(A)
print(f"det(A):{det_A}")
B = np.array([[1,2],[3,4]])
print(f"B:{B}")
det_B = np.linalg.det(B)
print(f"det(B):{det_B}")