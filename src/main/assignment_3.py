import numpy as np

def funct(t, y):
    return t - y**2

# QUESTION 1
def eulerMethod(t0, tn, i, y0):
    h = (tn - t0) / i 

    t = [t0]
    y = [y0]

    for x in range(i):
        y_next = y[-1] + h * funct(t[-1], y[-1])
        t_next = t[-1] + h
        y.append(y_next)
        t.append(t_next)
    
    print("%.5f" % y[-1])

# QUESTION 2
def rungeKutta(t0, tn, i, y0):
    h = (tn - t0) / i 

    t = [t0]
    y = [y0]

    for x in range(i):
        k1 = h * funct(t[-1], y[-1])
        k2 = h * funct(t[-1] + h/2, y[-1] + k1/2)
        k3 = h * funct(t[-1] + h/2, y[-1] + k2/2)
        k4 = h * funct(t[-1] + h, y[-1] + k3)
        y_next = y[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        t_next = t[-1] + h
        y.append(y_next)
        t.append(t_next)
    
    print("%.5f" % y[-1])

# QUESTION 3
def gaussian(A):
    for i in range(len(A)):
        pivot_row = i
        for j in range(i+1, len(A)):
            if abs(A[j][i]) > abs(A[pivot_row][i]):
                pivot_row = j
        A[i], A[pivot_row] = A[pivot_row], A[i]

        for j in range(i+1, len(A)):
            factor = A[j][i] / A[i][i]
            for k in range(i, len(A[0])):
                A[j][k] -= factor * A[i][k]


    x = [0] * len(A)
    for i in range(len(A)-1, -1, -1):
        x[i] = A[i][-1] / A[i][i]
        for j in range(i-1, -1, -1):
            A[j][-1] -= A[j][i] * x[i]
    
    print("[", int(x[0]), int(x[1]), int(x[2]), "]")

# QUESTION 4
def LU(A):
    L = np.eye(A.shape[0])
    U = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(i, A.shape[0]):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        for j in range(i+1, A.shape[0]):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    det = np.prod(np.diag(U))-0.00000000000001
    print("%.5f" % det)
    print()
    print(L)
    print()
    print(U)
    print()

if __name__ == "__main__": 
    # QUESTION 1
    eulerMethod(0, 2, 10, 1)
    print()

    # QUESTION 2
    rungeKutta(0, 2, 10, 1)
    print()

    # QUESTION 3
    A = [[2, -1, 1, 6],
                [1, 3, 1, 0],
                [-1, 5, 4, -3]]
    gaussian(A)
    print()

    # QUESTION 4
    A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]], dtype=float)
    LU(A)

    # QUESTION 5
    A = np.array([[9, 0, 5, 2, 1],
                [3, 9, 1, 2, 1],
                [0, 1, 7, 2, 3],
                [4, 2, 3, 12, 2],
                [3, 2, 4, 0, 8]])
    diagDominate = np.all(np.abs(A.diagonal()) >= np.sum(np.abs(A), axis=1) - np.abs(A.diagonal())) 
    print(diagDominate)
    print()

    # QUESTION 6
    A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])
    posDefinite = np.all(np.linalg.eigvals(A) > 0)
    print(posDefinite)