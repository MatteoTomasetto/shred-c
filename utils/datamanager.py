import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import MinMaxScaler
import torch
from utils.postprocessing import mre_numpy, num2p # Error metrics and format

class DataManager():
    def __init__(self, data, dims, train_ratio, valid_ratio, test_ratio):
        self.data =  dict(data.items())
        self.keys = list(data.keys())
        self.ntrajectories = self.data[self.keys[0]].shape[0]
        self.dims = dims
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        if train_ratio + valid_ratio + test_ratio != 1.0:
            raise ValueError("Train, valid and test ratios must sum to 1.")

        self.ntrain = round(train_ratio * self.ntrajectories)
        self.nvalid = round(valid_ratio * self.ntrajectories)
        self.ntest = round(test_ratio * self.ntrajectories)

    def prepare(self):
        ''' train, valid and test split '''

        self.train = np.random.choice(self.ntrajectories, size = self.ntrain, replace = False)
        mask = np.ones(self.ntrajectories)
        mask[self.train] = 0
        valid_test = np.arange(0, self.ntrajectories)[np.where(mask!=0)[0]]
        self.valid = valid_test[::2]
        self.test = valid_test[1::2]

    def POD(self, ranks, starting_times = None):
        ''' compute POD coefficients component-wise for each field '''

        self.data_POD = {}
        self.proj_POD = {}
        self.scaler = {}

        if starting_times is None:
            starting_times = {key: 0 for key in ranks.keys()}

        for key in ranks.keys():
            dim = self.dims[key]
            
            k = ranks[key]
            if k % dim != 0:
                raise ValueError(f"Rank k for key {key} must be a multiple of the dimension {dim}.")
            
            datadim = self.data[key].shape[-1]
            if datadim % dim != 0:
                raise ValueError(f"Data dimension for key {key} must be a multiple of the dimension {dim}.")
            
            for d in range(dim):
                component_data = self.data[key][:, starting_times[key]:, d : datadim : dim]

                _, _, self.proj_POD[key + f"_{d}"] = randomized_svd(component_data[self.train].reshape(-1, datadim//dim), n_components = k//dim)
                component_data_POD = component_data.reshape(-1, datadim//dim) @ self.proj_POD[key + f"_{d}"].transpose()
                component_data_train_POD = component_data[self.train].reshape(-1, datadim//dim) @ self.proj_POD[key + f"_{d}"].transpose()

                self.scaler[key + f"_{d}"] = MinMaxScaler()
                self.scaler[key + f"_{d}"] = self.scaler[key + f"_{d}"].fit(component_data_train_POD)
                component_data_POD = self.scaler[key + f"_{d}"].transform(component_data_POD)
                
                component_data_POD = component_data_POD.reshape(self.ntrajectories, -1, k//dim)
                if d == 0:
                    self.data_POD[key] = component_data_POD
                else:
                    self.data_POD[key] = np.concatenate([self.data_POD[key], component_data_POD], axis=2)

                data_test_reconstructed = component_data[self.test].reshape(-1, datadim//dim) @ self.proj_POD[key + f"_{d}"].transpose() @ self.proj_POD[key + f"_{d}"]
                data_test_reconstructed = data_test_reconstructed.reshape(self.ntest, -1, datadim//dim)
                print(f"Field: {key}, Component: {d}, Test MRE: {num2p(mre_numpy(component_data[self.test], data_test_reconstructed))}")

    def decode(self, data_POD):
        ''' decode POD coefficients to the original space '''

        data_decoded = {}

        for key in data_POD.keys():
            dim = self.dims[key]
            datadim = self.data[key].shape[-1]
            k = data_POD[key].shape[2]

            data_decoded[key] = np.zeros((data_POD[key].shape[0], data_POD[key].shape[1], datadim))

            for d in range(dim):
                component_data_POD = self.scaler[key + f"_{d}"].inverse_transform(data_POD[key][:, :, d*k//dim : (d+1)*k//dim].reshape(-1, k//dim))
                component_data_decoded = component_data_POD @ self.proj_POD[key + f"_{d}"]
                data_decoded[key][:, :, d : datadim : dim] = component_data_decoded.reshape(data_POD[key].shape[0], -1, datadim//dim)

        return data_decoded


class TimeSeriesDataset(torch.utils.data.Dataset):
    ''' Organize input measurements and states in a torch dataset '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len



def Padding(data, lag):
    ''' Apply lagging and padding to time-series data '''
    
    data_out = torch.zeros(data.shape[0] * data.shape[1], lag, data.shape[2])

    for i in range(data.shape[0]):
        for j in range(1, data.shape[1] + 1):
            if j < lag:
                data_out[i * data.shape[1] + j - 1, -j:] = data[i, :j]
            else:
                data_out[i * data.shape[1] + j - 1] = data[i, j - lag : j]

    return data_out

