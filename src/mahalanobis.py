@dataclass
class MahalanobisDistance:
    cov = None
    mean = None
    
    def fit(self,X_train:pd.DataFrame):
        self.cov = X_train.cov().values
        self.mean = X_train.mean().values
        return self
    def transform(self,X_test):
        x_mu = X_test - self.mean
        inv_covmat = np.linalg.inv(self.cov)
        left = np.dot(x_mu, inv_covmat)
        mahal = np.dot(left, x_mu.T)
        return mahal.diagonal()