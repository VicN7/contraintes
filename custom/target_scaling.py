import numpy as np
from sklearn.linear_model import LinearRegression

class LREnergyScaler:
    def __init__(self, possible_elements, center_reduce=False):
        self.Z_TO_COL = {possible_element:i for i, possible_element in enumerate(possible_elements)}
        self.lr = LinearRegression()
        self.residual_mean = 0.0
        self.residual_std  = 1.0
        self.center_reduce = center_reduce

    
    def get_atom_count(self, Z, N):
        n_sample = len(N)
        n_disctinct_atoms = len(self.Z_TO_COL)
        # Get an array of the count of atom in each molecules
        atom_count = np.zeros((n_sample, n_disctinct_atoms))
        start = 0
        for i, n in enumerate(N):
            for z in Z[start:start+n]:
                atom_count[i, self.Z_TO_COL[z]] += 1
            start += n
        return atom_count
    
    def fit(self, Z, N, target):
        atom_count = self.get_atom_count(Z, N)
        
        self.lr.fit(atom_count, target)
        print(f"Training R2 with linear model : {self.lr.score(atom_count, target)}")

        residual = target - self.lr.predict(atom_count)
        if self.center_reduce:
            self.residual_mean = np.mean(residual)
            self.residual_std  = np.std(residual)
            print(f"residual mean = {self.residual_mean}")
            print(f"residual std = {self.residual_std}")

    
    def transform(self, N, Z, target):
        atom_count = self.get_atom_count(Z, N)
        residual = target - self.lr.predict(atom_count)
        scaled_residual = (residual - self.residual_mean) / self.residual_std
        return scaled_residual
    
    def inverse_transform(self, N, Z, predicted_target):
        atom_count = self.get_atom_count(Z, N)
        predicted_target = np.array(predicted_target).squeeze()
        predicted_E = self.lr.predict(atom_count) + self.residual_std * predicted_target + self.residual_mean
        return predicted_E.reshape(-1, 1)