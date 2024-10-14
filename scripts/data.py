import numpy as np
import pandas as pd


def generate_scenario(seed, mu, cov, size):
    np.random.seed(seed)
    return np.random.multivariate_normal(mean=mu, cov=cov, size=size)


def choose_head_n_stock_data(fname, n):
    f = open(fname + ".txt", "r")
    wf = open(fname + "_nstock." + str(n) + ".txt", "w")
    lines = f.readlines()
    i = 0
    wf.write(" " + str(n) + "\n")
    for line in lines:
        i += 1
        if i >= 2 and i <= n + 1:
            wf.write(line)
        else:
            l = line.split(" ")[1:]
            if len(l) == 3:
                row = int(l[0])
                col = int(l[1])
                if row <= n and col <= n:
                    wf.write(line)
    f.close()
    wf.close()


class DataORLibrary:
    """docstring for ."""

    def __init__(self, fname):
        scale = 1
        self.fname = fname
        f = open(self.fname, "r")
        lines = f.readlines()
        lines = [l.replace("\n", "") for l in lines]
        lines = [l[1:] for l in lines]

        self.n = int(lines[0])
        self.mean_list = []
        self.var_list = []
        self.cov_dict = {}
        for l in lines[:-1]:
            l = l.split(" ")
            if len(l) == 2:
                # mean and variance
                self.mean_list.append(float(l[0]) * scale)
                self.var_list.append(float(l[1]) * scale)
            if len(l) == 3:
                row = int(l[0]) - 1
                col = int(l[1]) - 1
                self.cov_dict[row, col] = (
                    float(l[2]) * self.var_list[row] * self.var_list[col]
                )

    def mean(self):
        return np.array(self.mean_list)

    def get_variance_array(self):
        return np.array(self.var_list)

    def covariance(self):
        m = np.empty((self.n, self.n))
        for key in self.cov_dict:
            row = key[0]
            col = key[1]
            val = self.cov_dict[row, col]
            m[row, col] = val
            m[col, row] = val
        l, v = np.linalg.eig((m + m.T) / 2)
        return (m + m.T) / 2

    def savecsv(self):
        mu = self.mean()
        cov = self.covariance()
        wfname1 = self.fname.replace(".txt", "_mu.csv")
        wfname2 = self.fname.replace(".txt", "_cov.csv")
        np.savetxt(wfname1, mu, delimiter=",")
        np.savetxt(wfname2, cov, delimiter=",")


class FamaFrench:
    def __init__(self, filename):
        if filename:
            self.read(filename)
            self.fname = filename
        else:
            self.values = np.array([])

    def read(self, filename):
        self.dataframe = pd.read_csv(filename)
        self.dataframe["YYYYMM"] = self.dataframe["YYYYMM"].astype(int)
        self.yyyymm = self.dataframe.values[:, 0]
        self.values = self.dataframe.values[:, 1:]
        print(self.values[1:3, :])

    def value(self):
        return self.values

    def mean(self):
        return np.mean(self.values, axis=0)

    def cov(self):
        return np.cov(self.values.T)

    def savecsv(self):
        mu = self.mean()
        cov = self.cov()
        wfname1 = self.fname.replace(".csv", "_mu.csv")
        wfname2 = self.fname.replace(".csv", "_cov.csv")
        np.savetxt(wfname1, mu, delimiter=",")
        np.savetxt(wfname2, cov, delimiter=",")


class YFinance:
    def __init__(self, filename):
        if filename:
            self.read(filename)
        else:
            self.values = np.array([])

    def read(self, filename):
        """read the dataset from a file"""
        self.dataframe = pd.read_csv(filename)
        self.yyyymm = self.dataframe.values[:, 0]
        self.values = self.dataframe.values[:, 1:]
        print(self.values[1:3, :])

    def value(self):
        return self.values

    def mean(self):
        return np.mean(self.values, axis=0)

    def cov(self):
        return np.cov(self.values.T)


class SandP500:
    """docstring for ."""

    def __init__(self, filename):
        if filename:
            self.read(filename)
        else:
            self.values = np.array([])

    def __call__(self):
        return self.values

    def read(self, filename):
        """read the dataset from a file"""
        self.dataframe = pd.read_csv(filename)
        self.yyyymm = self.dataframe.values[:, 0]
        self.values = self.dataframe.values[:, 1:]

    def value(self):
        return self.values

    def mean(self):
        return np.mean(self.values, axis=0)

    def cov(self):
        return np.cov(self.values.T)


class SandP:
    """docstring for ."""

    def __init__(self, filename):
        if filename:
            self.read(filename)
        else:
            self.values = np.array([])

    def read(self, filename):
        """read the dataset from a file"""
        self.dataframe = pd.read_csv(filename)
        self.yyyymmdd = self.dataframe.values[:, 0]
        self.values = self.dataframe.values[:, 1:].astype(float)

    def mean(self):
        return np.mean(self.values, axis=0)

    def cov(self):
        print(self.values.T.shape)
        print(np.cov(self.values.T))
        return np.cov(self.values.T)
