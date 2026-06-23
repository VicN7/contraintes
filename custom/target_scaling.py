import numpy as np
from sklearn.linear_model import LinearRegression

class LREnergyScaler:
    def __init__(self, center_reduce=True):
        self.lr = LinearRegression()
        self.residual_mean = 0.0
        self.residual_std  = 1.0
        self.center_reduce = center_reduce

    
    def fit(self, atom_count, target):
        self.lr.fit(atom_count, target)
        print(f"Training R2 with linear model : {self.lr.score(atom_count, target)}")

        residual = target - self.lr.predict(atom_count)
        if self.center_reduce:
            self.residual_mean = np.mean(residual)
            self.residual_std  = np.std(residual)
            print(f"residual mean = {self.residual_mean}")
            print(f"residual std = {self.residual_std}")

    
    def transform(self, atom_count, target):
        residual = target - self.lr.predict(atom_count)
        scaled_residual = (residual - self.residual_mean) / self.residual_std
        return scaled_residual
    
    def inverse_transform(self, atom_count, predicted_target):
        predicted_target = np.array(predicted_target).squeeze()
        predicted_E = self.lr.predict(atom_count) + self.residual_std * predicted_target + self.residual_mean
        return predicted_E.reshape(-1, 1)