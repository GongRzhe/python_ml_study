import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        def demean(X):
            return X - np.mean(X, axis=0)
        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initail_w, eta, n_iters=1e4, epsilon=1e-8):
            w = direction(initail_w)
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break
                cur_iter += 1
            return w

        X_pca = demean(X)
        self.components_=np.empty(shape=(self.n_components,X.shape[1]))
        for i in range(self.n_components):
            initail_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initail_w, eta,n_iters)
            self.components_[i,:]=w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        return self
    def transform(self,X):
        return X.dot(self.components_.T)
    def inverse_transform(self,X):
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
