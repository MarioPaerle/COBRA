import numpy as np
from util import chronoscore


class LinearModelSolver:
    def __init__(self, A, b):
        self.params = (A, b)
        self.A = A  # mxn
        self.b = b  # mx1
        self.output = None
        self.x = np.random.randn(len(b)).reshape(len(b), 1)  # nx1
        self.dLdx = 0

    def forward(self):
        self.output = self.A @ self.x

    def backward(self):
        self.forward()
        dLdx = 2 * self.A.T @ (self.A @ self.x - self.b)
        self.x -= dLdx * 0.0002

    def s_backward(self):
        self.forward()
        n = len(self.A)
        A = self.A # * np.where(np.random.randn(n, n) > -1, 1, 0)

        self.dLdx = 2 * A.T @ (A @ self.x - self.b)
        self.x -= self.dLdx * 0.2

    def fit(self, epochs=10000, verbose=True):
        for epoch in range(epochs):
            self.backward()
            if epoch % 100 == 0 and verbose:
                print(f"Epoch {epoch}, Loss: {np.linalg.norm(self.output - self.b)}")

    def fit_reset(self, epochs=100, verbose=False):
        for epoch in range(epochs):
            self.backward()
        self.reset()

    def reset(self):
        self.__init__(
            *self.params)


if __name__ == '__main__':
    A = np.random.rand(100, 100)
    b = np.random.rand(100, 1)
    x = np.linalg.solve(A, b)
    print('done')
    print(np.linalg.norm(A @ x - b))
    LMS = LinearModelSolver(A, b)
    LMS.fit(10000)





