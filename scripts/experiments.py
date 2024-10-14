from numpy.core.function_base import _add_docstring
import model
import data
import numpy as np
import os


def create_directory(dir_name):
    """create directory

    Parameters
    ----------
    dir_name : str(file path)
        create directory name

    Returns
    -------
    None
    """
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    else:
        pass


def construct_utitlity_function(alpha, L, mu_max):
    """construct L linear functions approximating (1-exp(-alpha*x))/alpha
    Args:
        alpha (float): parameter defining a utility function
        L (int): number of linear functions approximating the utility function (L>=2)
        mu_max (float): maximum value of return among all candidate assets
    """
    a = {}
    b = {}
    a[0] = 1
    b[0] = 0
    if L >= 2:
        for l in range(L):
            if l >= 1:
                a[l] = np.exp(-alpha * mu_max * (l) / (L - 1))
                b[l] = (l * mu_max / (L - 1)) * np.exp(-alpha * mu_max * (l) / (L - 1))
    else:
        pass
    return (a, b)


def experiments(dataname, L, gamma1, gamma2, delta, k):
    if dataname in ["port1", "port2", "port3", "port4", "port5"]:
        data_ = data.DataORLibrary("../data/" + dataname + ".txt")
        mu = data_.mean()
        cov = data_.covariance()
        cov = 10000 * cov
        mu = 100 * mu
    elif dataname in ["industry38", "industry49", "100Portfolios"]:
        data_ = data.FamaFrench("../data/" + dataname + ".csv")
        mu = data_.mean()
        cov = data_.cov()
    elif dataname in ["SP468"]:
        data_ = data.SandP("../data/" + dataname + ".csv")
        mu = data_.mean()
        cov = data_.cov()

    mu_max = np.max(mu)
    a, b = construct_utitlity_function(alpha=10 / mu_max, L=L, mu_max=mu_max)

    result_dir = "../results_piecewise_alpha.10_scale.100_10000"
    create_directory(result_dir)
    if dataname != None:
        result_dir = result_dir + "/data." + dataname
        result_dir = result_dir + ",L." + str(L)
        result_dir = result_dir + ",gamma1." + str(gamma1)
        result_dir = result_dir + ",gamma2." + str(gamma2)
        result_dir = result_dir + ",delta." + str(delta)
        result_dir = result_dir + ",k." + str(k)
    else:
        result_dir = result_dir + "/data.port1-10"

    m = model.DistRobustModel(
        mu=mu, cov=cov, gamma1=gamma1, gamma2=gamma2, delta=delta, k=k, a=a, b=b
    )
    scale = 100
    m.write_sdpaformat(scale=scale, wfname=result_dir + ".dat-s")

    m.solve(reduce_flag=1, logname=result_dir + "_method.cpa_reduced")
    m.solve(reduce_flag=0, logname=result_dir + "_method.cpa")


if __name__ == "__main__":
    a, b = construct_utitlity_function(alpha=1, L=1, mu_max=1)
    dataname_list = ["port1"]
    for dataname in dataname_list:
        experiments(dataname=dataname, gamma1=0.1, gamma2=3, delta=10, k=3, L=3)
