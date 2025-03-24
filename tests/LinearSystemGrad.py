import numpy as np


class LinearModelSolver:
    def __init__(self, A, b):
        self.A = A  # mxn
        self.b = b  # mx1
        self.output = None
        self.x = np.random.randn(len(b)).reshape(len(b), 1)  # nx1

        print(self.A.shape)
        print(self.b.shape)
        print(self.x.shape)

    def forward(self):
        self.output = self.A @ self.x

    def backward(self):
        self.forward()
        dLdx = 2 * self.A.T @ (self.A @ self.x - self.b)
        self.x -= dLdx * 0.001

    def fit(self, epochs=10000):
        for epoch in range(epochs):
            self.backward()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {np.linalg.norm(self.output - self.b)}")


if __name__ == '__main__':
    A = np.array([[1, 2], [3, 4]]).astype(np.float64)
    b = np.array([[1], [2]]).astype(np.float64)

    A = np.random.randn(100, 100)
    b = np.random.randn(100, 1)

    LMS = LinearModelSolver(A, b)
    LMS.fit(10000)


