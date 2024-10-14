import cvxpy as cp

cp.settings.ERROR = [cp.settings.USER_LIMIT]
cp.settings.SOLUTION_PRESENT = [
    cp.settings.OPTIMAL,
    cp.settings.OPTIMAL_INACCURATE,
    cp.settings.SOLVER_ERROR,
]
import numpy as np
import gurobipy as grb
import time as time
import yaml


class DistRobustModel:
    def __init__(
        self, mu, cov, gamma1, gamma2, delta, k, timelimit=3600, a={0: 1}, b={0: 0}
    ):
        self.mu = mu
        self.cov = cov
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.num_stock = self.mu.shape[0]
        self.delta = delta
        self.delta_ = self.delta / np.sqrt(self.num_stock)
        self.k = k
        self.timelimit = timelimit
        self.a = a
        self.b = b

    def complete_lambda(self, Sigma, lambda_, index):
        index_complement = list(set([i for i in range(self.num_stock)]) - set(index))
        Sigma_11 = Sigma[:, index][index, :]
        Sigma_21 = Sigma[:, index][index_complement, :]
        tmp = np.dot(Sigma_21, np.linalg.solve(Sigma_11, lambda_))
        lambda_full = np.zeros(self.num_stock)
        for i in range(self.num_stock):
            if i in index:
                lambda_full[i] = lambda_[index.index(i)]
            if i in index_complement:
                lambda_full[i] = tmp[index_complement.index(i)]
        return lambda_full

    def complete_beta(self, Sigma, beta, eta, index):
        beta_hat = beta - eta * self.mu[index]
        index_complement = list(set([i for i in range(self.num_stock)]) - set(index))
        Sigma_11 = Sigma[:, index][index, :]
        Sigma_21 = Sigma[:, index][index_complement, :]
        tmp = np.dot(Sigma_21, np.linalg.solve(Sigma_11, beta_hat))
        beta_full = np.zeros(self.num_stock)
        for i in range(self.num_stock):
            if i in index:
                beta_full[i] = self.mu[i] * eta + beta_hat[index.index(i)]
            if i in index_complement:
                beta_full[i] = self.mu[i] * eta + tmp[index_complement.index(i)]
        return beta_full

    def write_sdpaformat(self, scale, wfname):
        dict_var_index = {}
        dict_index_var = {}
        dict_constr_index = {}
        dict_index_constr = {}

        index = 0
        for i in range(self.num_stock):
            dict_index_var[i] = "x_" + str(i)
            dict_var_index["x_" + str(i)] = index
            index += 1
        for i in range(self.num_stock):
            dict_index_var[index] = "z_" + str(i)
            dict_var_index["z_" + str(i)] = index
            index += 1
        for i in range(self.num_stock):
            for j in range(i, self.num_stock):
                dict_index_var[index] = "P_" + str(i) + "_" + str(j)
                dict_var_index["P_" + str(i) + "_" + str(j)] = index
                index += 1
        for i in range(self.num_stock):
            dict_index_var[index] = "p_" + str(i)
            dict_var_index["p_" + str(i)] = index
            index += 1
        for i in range(self.num_stock):
            for j in range(i, self.num_stock):
                dict_index_var[index] = "Q_" + str(i) + "_" + str(j)
                dict_var_index["Q_" + str(i) + "_" + str(j)] = index
                index += 1
        for i in range(self.num_stock):
            dict_index_var[index] = "q_" + str(i)
            dict_var_index["q_" + str(i)] = index
            index += 1
        dict_index_var[index] = "r"
        dict_var_index["r"] = index
        index += 1

        dict_index_var[index] = "s"
        dict_var_index["s"] = index
        index += 1

        dict_index_var[index] = "t"
        dict_var_index["t"] = index
        index += 1

        with open(wfname, "w") as wf:
            wf.write(str(len(dict_index_var)) + "\n")
            wf.write(str(len(self.a) + 3) + "\n")
            for key in self.a:
                wf.write(str(self.num_stock + 1) + " ")
            wf.write(
                str(self.num_stock + 1)
                + " "
                + str(self.num_stock + 1)
                + " "
                + str(-6 * self.num_stock - 3)
                + "\n"
            )
            # set objective
            for index in dict_index_var:
                var = dict_index_var[index]
                if var[0] == "x":
                    wf.write(str(0) + " ")
                elif var[0] == "z":
                    wf.write(str(0) + " ")
                elif var[0] == "P":
                    i = int(var.split("_")[-2])
                    j = int(var.split("_")[-1])
                    if i == j:
                        wf.write("{:.5f}".format(scale * self.cov[i, j]) + " ")
                    else:
                        wf.write("{:5f}".format(scale * 2 * self.cov[i, j]) + " ")
                elif var[0] == "p":
                    i = int(var.split("_")[-1])
                    wf.write("{:5f}".format(-scale * 2 * self.mu[i]) + " ")
                elif var[0] == "Q":
                    i = int(var.split("_")[-2])
                    j = int(var.split("_")[-1])
                    if i == j:
                        wf.write(
                            "{:.5f}".format(
                                scale * self.gamma2 * self.cov[i, j]
                                - scale * self.mu[i] * self.mu[j]
                            )
                            + " "
                        )
                    else:
                        wf.write(
                            "{:.5f}".format(
                                2
                                * (
                                    scale * self.gamma2 * self.cov[i, j]
                                    - scale * self.mu[i] * self.mu[j]
                                )
                            )
                            + " "
                        )
                elif var[0] == "q":
                    i = int(var.split("_")[-1])
                    wf.write(str(0) + " ")
                elif var[0] == "r":
                    wf.write(str(scale * 1) + " ")
                elif var[0] == "s":
                    wf.write(str(scale * self.gamma1) + " ")
                elif var[0] == "t":
                    wf.write("{:5f}".format(scale * 1 / (2 * self.delta_)))
            wf.write("\n")
            # constraints
            # [Q q/2+a^{l}x/2; (q/2+a^{l}x/2)^\top r+b^{l}] \succeq O
            block = 1
            for key in self.a:
                wf.write(
                    str(0)
                    + " "
                    + str(block)
                    + " "
                    + str(self.num_stock + 1)
                    + " "
                    + str(self.num_stock + 1)
                    + " "
                    + "{:.16f}".format(-self.b[key])
                    + "\n"
                )
                for index in dict_index_var:
                    var = dict_index_var[index]
                    if "x" == var[0]:
                        i = int(var.split("_")[-1])
                        wf.write(
                            str(index + 1)
                            + " "
                            + str(block)
                            + " "
                            + str(i + 1)
                            + " "
                            + str(self.num_stock + 1)
                            + " "
                            + "{:.16f}".format(0.5 * self.a[key])
                            + "\n"
                        )
                    elif "q" == var[0]:
                        i = int(var.split("_")[-1])
                        wf.write(
                            str(index + 1)
                            + " "
                            + str(block)
                            + " "
                            + str(i + 1)
                            + " "
                            + str(self.num_stock + 1)
                            + " "
                            + "0.5\n"
                        )
                    elif "Q" == var[0]:
                        i = int(var.split("_")[-2])
                        j = int(var.split("_")[-1])
                        wf.write(
                            str(index + 1)
                            + " "
                            + str(block)
                            + " "
                            + str(i + 1)
                            + " "
                            + str(j + 1)
                            + " "
                            + "1\n"
                        )
                    elif "r" == var[0]:
                        wf.write(
                            str(index + 1)
                            + " "
                            + str(block)
                            + " "
                            + str(self.num_stock + 1)
                            + " "
                            + str(self.num_stock + 1)
                            + " "
                            + "1\n"
                        )
                block += 1
            # 2nd block [P p; p^\top s] \succeq O
            for index in dict_index_var:
                var = dict_index_var[index]
                if "p" == var[0]:
                    i = int(var.split("_")[-1])
                    wf.write(
                        str(index + 1)
                        + " "
                        + str(block)
                        + " "
                        + str(i + 1)
                        + " "
                        + str(self.num_stock + 1)
                        + " "
                        + "1\n"
                    )
                elif "P" == var[0]:
                    i = int(var.split("_")[-2])
                    j = int(var.split("_")[-1])
                    wf.write(
                        str(index + 1)
                        + " "
                        + str(block)
                        + " "
                        + str(i + 1)
                        + " "
                        + str(j + 1)
                        + " "
                        + "1\n"
                    )
                elif "s" == var[0]:
                    wf.write(
                        str(index + 1)
                        + " "
                        + str(block)
                        + " "
                        + str(self.num_stock + 1)
                        + " "
                        + str(self.num_stock + 1)
                        + " "
                        + "1\n"
                    )
            block += 1
            # 3rd block [I x; x^\top t] \succeq O
            for i in range(self.num_stock):
                wf.write(
                    str(0)
                    + " "
                    + str(block)
                    + " "
                    + str(i + 1)
                    + " "
                    + str(i + 1)
                    + " -1\n"
                )
            for index in dict_index_var:
                var = dict_index_var[index]
                if "x" == var[0]:
                    i = int(var.split("_")[-1])
                    wf.write(
                        str(index + 1)
                        + " "
                        + str(block)
                        + " "
                        + str(i + 1)
                        + " "
                        + str(self.num_stock + 1)
                        + " "
                        + "1\n"
                    )
                elif "t" == var[0]:
                    wf.write(
                        str(index + 1)
                        + " "
                        + str(block)
                        + " "
                        + str(self.num_stock + 1)
                        + " "
                        + str(self.num_stock + 1)
                        + " "
                        + "1\n"
                    )
            # 4th block linear constr
            block += 1
            constr_index = 0
            # p +q/2 + Qmu >= 0
            for i in range(self.num_stock):
                index_p = dict_var_index["p_" + str(i)]
                wf.write(
                    str(index_p + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " 1\n"
                )
                index_q = dict_var_index["q_" + str(i)]
                wf.write(
                    str(index_q + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " 0.5\n"
                )
                for j in range(self.num_stock):
                    if j < i:
                        index_Q = dict_var_index["Q_" + str(j) + "_" + str(i)]
                    else:
                        index_Q = dict_var_index["Q_" + str(i) + "_" + str(j)]
                    wf.write(
                        str(index_Q + 1)
                        + " "
                        + str(block)
                        + " "
                        + str(constr_index + 1)
                        + " "
                        + str(constr_index + 1)
                        + " "
                        + str(self.mu[j])
                        + "\n"
                    )
                constr_index += 1
            # -p -q/2 - Qmu >= 0
            for i in range(self.num_stock):
                index_p = dict_var_index["p_" + str(i)]
                wf.write(
                    str(index_p + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " -1\n"
                )
                index_q = dict_var_index["q_" + str(i)]
                wf.write(
                    str(index_q + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " -0.5\n"
                )
                for j in range(self.num_stock):
                    if j < i:
                        index_Q = dict_var_index["Q_" + str(j) + "_" + str(i)]
                    else:
                        index_Q = dict_var_index["Q_" + str(i) + "_" + str(j)]
                    wf.write(
                        str(index_Q + 1)
                        + " "
                        + str(block)
                        + " "
                        + str(constr_index + 1)
                        + " "
                        + str(constr_index + 1)
                        + " "
                        + str(-self.mu[j])
                        + "\n"
                    )
                constr_index += 1
            # x>=0
            for i in range(self.num_stock):
                index_x = dict_var_index["x_" + str(i)]
                wf.write(
                    str(index_x + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " 1\n"
                )
                constr_index += 1
            # z-x>=0
            for i in range(self.num_stock):
                index_x = dict_var_index["x_" + str(i)]
                index_z = dict_var_index["z_" + str(i)]
                wf.write(
                    str(index_x + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " -1\n"
                )
                wf.write(
                    str(index_z + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " 1\n"
                )
                constr_index += 1
            # -1^\topx <=-1
            wf.write(
                str(0)
                + " "
                + str(block)
                + " "
                + str(constr_index + 1)
                + " "
                + str(constr_index + 1)
                + " 1\n"
            )
            for i in range(self.num_stock):
                index_x = dict_var_index["x_" + str(i)]
                wf.write(
                    str(index_x + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " 1\n"
                )
            constr_index += 1
            # -1^\topx >=-1
            wf.write(
                str(0)
                + " "
                + str(block)
                + " "
                + str(constr_index + 1)
                + " "
                + str(constr_index + 1)
                + " -1\n"
            )
            for i in range(self.num_stock):
                index_x = dict_var_index["x_" + str(i)]
                wf.write(
                    str(index_x + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " -1\n"
                )
            constr_index += 1
            # 1^\topz <=k
            wf.write(
                str(0)
                + " "
                + str(block)
                + " "
                + str(constr_index + 1)
                + " "
                + str(constr_index + 1)
                + " "
                + str(-self.k)
                + "\n"
            )
            for i in range(self.num_stock):
                index_z = dict_var_index["z_" + str(i)]
                wf.write(
                    str(index_z + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " -1\n"
                )
            constr_index += 1
            # 1 >= z
            for i in range(self.num_stock):
                index_z = dict_var_index["z_" + str(i)]
                wf.write(
                    str(0)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " -1\n"
                )
                wf.write(
                    str(index_z + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " -1\n"
                )
                constr_index += 1
            # z>=0
            for i in range(self.num_stock):
                index_z = dict_var_index["z_" + str(i)]
                wf.write(
                    str(index_z + 1)
                    + " "
                    + str(block)
                    + " "
                    + str(constr_index + 1)
                    + " "
                    + str(constr_index + 1)
                    + " 1\n"
                )
                constr_index += 1
            wf.write("*INTEGER\n")
            for index in dict_index_var:
                if "z" == dict_index_var[index][0]:
                    wf.write("*" + str(index + 1) + "\n")

    def solve(self, reduce_flag, logname):
        start_all = time.time()
        theta_lower = -100
        self.time_callback = 0
        self.num_callback = 0

        def add_cutting_plane(model, where):
            if where == grb.GRB.callback.MIPSOL:
                start_callback = time.time()
                z_tmp = {}
                z_var = {}
                for var in model._vars:
                    if "z" in var.VarName:
                        i = int(var.VarName.split("_")[-1])
                        z_tmp[i] = model.cbGetSolution(var)
                        z_var[i] = var
                    if "theta" in var.VarName:
                        theta_var = var
                        theta_tmp = model.cbGetSolution(var)
                if reduce_flag == 1:
                    x_tmp, f, w_tmp, beta_tmp = self.solve_inner_reduced_dual(z=z_tmp)
                else:
                    x_tmp, f, w_tmp, alpha_tmp, bata_tmp, lambda_tmp = (
                        self.solve_inner_dual(z=z_tmp)
                    )
                s = 1
                model.cbLazy(
                    theta_var / s
                    >= f / s
                    + grb.quicksum(
                        -(self.delta_ / 2)
                        * (w_tmp[i] * w_tmp[i])
                        * (z_var[i] - z_tmp[i])
                        for i in range(self.num_stock)
                    )
                    / s
                )
                self.time_callback += time.time() - start_callback
                self.num_callback += 1

        self.model = grb.Model("dist_robust")
        z = {}
        for i in range(self.num_stock):
            z[i] = self.model.addVar(vtype="B", name="z_" + str(i))
        theta = self.model.addVar(vtype="C", name="theta", lb=theta_lower)
        self.model.update()
        self.model.addConstr(
            grb.quicksum(z[i] for i in range(self.num_stock)) == self.k
        )
        self.model.update()
        self.model.setObjective(theta)
        self.model.update()
        self.model._vars = self.model.getVars()
        self.model.params.NumericFocus = 0
        self.model.params.LazyConstraints = 1
        self.model.params.TIME_LIMIT = self.timelimit
        self.model.params.LogFile = logname + ".log"
        self.model.update()
        start_solve = time.time()
        self.model.optimize(add_cutting_plane)
        self.time_solve = time.time() - start_solve
        self.time_all = time.time() - start_all
        if self.model.Status == grb.GRB.OPTIMAL:
            z_opt = {}
            x_opt = {}
            for var in self.model.getVars():
                if "z_" in var.VarName:
                    i = int(var.VarName.split("_")[-1])
                    z_opt[i] = var.X
                elif "x_" in var.VarName:
                    i = int(var.VarName.split("_")[-1])
                    x_opt[i] = var.X
        d = {}
        d["params"] = {}
        d["params"]["gamma1"] = float(self.gamma1)
        d["params"]["gamma2"] = float(self.gamma2)
        d["params"]["delta"] = float(self.delta)
        d["params"]["num_stock"] = int(self.num_stock)
        d["params"]["k"] = int(self.k)
        d["results"] = {}
        d["results"]["obj_bst"] = self.model.ObjVal
        d["results"]["obj_bnd"] = self.model.ObjBound
        d["results"]["gap"] = self.model.MIPGap
        d["results"]["num_nodes"] = self.model.NodeCount
        d["results"]["num_callback"] = self.num_callback
        d["results"]["time_all"] = self.time_all
        d["results"]["time_solve"] = self.time_solve
        d["results"]["time_callback"] = self.time_callback
        if self.model.Status == grb.GRB.OPTIMAL:
            x_opt, opt_val, w_opt, beta_opt = self.solve_inner_reduced_dual(z=z_opt)
            d["results"]["status"] = "optimal"
            d["results"]["x_opt"] = {}
            for i in range(self.num_stock):
                d["results"]["x_opt"][i] = float(x_opt[i] * z_opt[i])
        elif self.model.Status == grb.GRB.TIME_LIMIT:
            d["results"]["status"] = "timelimit"
        with open(logname + ".yml", "w") as f:
            yaml.dump(d, f, default_flow_style=False)
        if self.model.Status == grb.GRB.OPTIMAL:
            return x_opt
        else:
            return None

    def solve_inner_primal(self, z):
        index = [i for i in z if z[i] >= 0.5]
        mu_sub = self.mu[index]
        cov_sub = self.cov[:, index]
        cov_sub = cov_sub[index, :]

        x = cp.Variable(len(index))
        p = cp.Variable(len(index))
        q = cp.Variable(len(index))
        P = cp.Variable((len(index), len(index)), symmetric=True)
        Q = cp.Variable((len(index), len(index)), symmetric=True)
        r = cp.Variable()
        s = cp.Variable()
        M = {}
        M2 = cp.Variable((self.num_stock + 1, self.num_stock + 1), symmetric=True)
        for key in self.a:
            M[key] = cp.Variable(
                (self.num_stock + 1, self.num_stock + 1), symmetric=True
            )
        objective = (
            cp.sum_squares(x) / (2 * self.delta_)
            + cp.trace((self.gamma2 * cov_sub - np.outer(mu_sub, mu_sub)) @ Q)
            + r
            + cp.trace(cov_sub @ P)
            - 2 * (mu_sub.T @ p)
            + self.gamma1 * s
        )
        constraints = [p == -q / 2 - Q @ mu_sub]
        constraints += [cp.sum(x) == 1, x >= 0]
        constraints += [M2 >> 0]
        constraints += [
            M2[0 : len(index), 0 : len(index)] == P,
            M2[-1, 0 : len(index)] == p,
            M2[len(index), len(index)] == s,
        ]
        for key in self.a:
            constraints += [
                M[key][0 : len(index), 0 : len(index)] == Q,
                M[key][-1, 0 : len(index)] == (q + self.a[key] * x) / 2,
                M[key][len(index), len(index)] == r + b[key],
            ]
            constraints += [M[key] >> 0]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.MOSEK)
        return (x.value, prob.value)

    def solve_inner_dual(self, z):
        sarat_solve_inner_dual = time.time()
        index = [i for i in z if z[i] >= 0.5]
        mu_sub = self.mu[index]
        cov_sub = self.cov[:, index]
        cov_sub = cov_sub[index, :]

        w = cp.Variable(self.num_stock)
        alpha = cp.Variable(self.num_stock)
        B = {}
        beta = {}
        eta = {}
        for key in self.a:
            B[key] = cp.Variable((self.num_stock, self.num_stock), symmetric=True)
            beta[key] = cp.Variable(self.num_stock)
            eta[key] = cp.Variable()
        lambda_ = cp.Variable(self.num_stock)
        pi = cp.Variable()
        M = {}
        for key in self.a:
            M[key] = cp.Variable(
                (self.num_stock + 1, self.num_stock + 1), symmetric=True
            )
        M2 = cp.Variable((self.num_stock + 1, self.num_stock + 1), symmetric=True)
        M3 = cp.Variable((self.num_stock, self.num_stock))
        objective = (
            -cp.sum_squares(w[index]) * (self.delta_ / 2)
            - sum([self.b[key] * eta[key] for key in self.b])
            + pi
        )
        constraints = [
            w
            >= sum([self.a[key] * beta[key] for key in beta])
            + pi * np.ones(self.num_stock)
        ]
        for i in range(self.num_stock):
            for j in range(self.num_stock):
                constraints += [
                    M3[i, j]
                    == self.gamma2 * self.cov[i, j]
                    - self.mu[i] * self.mu[j]
                    - (alpha[i] * self.mu[j] + alpha[j] * self.mu[i]) / 2
                    - sum([B[key][i, j] for key in B])
                ]
        constraints += [M3 == 0]
        constraints += [-2 * self.mu - alpha - 2 * lambda_ == 0]
        constraints += [-alpha / 2 - sum([beta[key] for key in beta]) == 0]
        constraints += [sum([eta[key] for key in eta]) == 1]
        for key in self.a:
            constraints += [
                M[key][0 : self.num_stock, 0 : self.num_stock] == B[key],
                M[key][self.num_stock, 0 : self.num_stock] == beta[key],
                M[key][self.num_stock, self.num_stock] == eta[key],
            ]
        constraints += [
            M2[0 : self.num_stock, 0 : self.num_stock] == self.cov,
            M2[self.num_stock, 0 : self.num_stock] == lambda_,
            M2[self.num_stock, self.num_stock] == self.gamma1,
        ]
        for key in M:
            constraints += [M[key] >> 0]
        constraints += [M2 >> 0]
        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)
        w_opt = w.value
        beta_opt = {}
        for key in beta:
            beta_opt[key] = beta[key].value
        pi_opt = pi.value
        lambda_opt = lambda_.value
        alpha_opt = alpha.value

        x_dic = {}
        w_dic = {}
        alpha_dic = {}
        beta_dic = {}
        lambda_dic = {}

        for i in range(self.num_stock):
            w_dic[i] = w_opt[i]
            alpha_dic[i] = alpha_opt[i]
            lambda_dic[i] = lambda_opt[i]
            if i in index:
                x_dic[i] = w_opt[i] * self.delta_
            else:
                x_dic[i] = 0
        for key in beta_opt:
            beta_dic[key] = {}
            for i in range(self.num_stock):
                beta_dic[key][i] = beta_opt[key][i]
        return (x_dic, prob.value, w_dic, alpha_dic, beta_dic, lambda_dic)

    def solve_inner_reduced_dual(self, z):
        sarat_solve_inner_dual = time.time()
        index = [i for i in z if z[i] >= 0.5]
        mu_sub = self.mu[index]
        cov_sub = self.cov[:, index]
        cov_sub = cov_sub[index, :]

        w = cp.Variable(len(index))
        alpha = cp.Variable(len(index))
        B = {}
        beta = {}
        eta = {}
        for key in self.a:
            B[key] = cp.Variable((len(index), len(index)), symmetric=True)
            beta[key] = cp.Variable(len(index))
            eta[key] = cp.Variable()
        lambda_ = cp.Variable(len(index))
        pi = cp.Variable()
        M = {}
        for key in self.a:
            M[key] = cp.Variable((len(index) + 1, len(index) + 1), symmetric=True)
        M2 = cp.Variable((len(index) + 1, len(index) + 1), symmetric=True)
        M3 = cp.Variable((len(index), len(index)))
        objective = (
            -cp.sum_squares(w) * (self.delta_ / 2)
            - sum([self.b[key] * eta[key] for key in self.b])
            + pi
        )
        constraints = [
            w
            >= sum([self.a[key] * beta[key] for key in self.a])
            + pi * np.ones(len(index))
        ]
        scale = 1
        for i in range(len(index)):
            for j in range(len(index)):
                constraints += [
                    M3[i, j] / scale
                    == self.gamma2 * cov_sub[i, j] / scale
                    - mu_sub[i] * mu_sub[j] / scale
                    - (alpha[i] * mu_sub[j] + alpha[j] * mu_sub[i]) / 2 / scale
                    - sum([B[key][i, j] for key in B]) / scale
                ]
        constraints += [M3 == 0]
        constraints += [-2 * mu_sub - alpha - 2 * lambda_ == 0]
        constraints += [-alpha / 2 - sum([beta[key] for key in beta]) == 0]
        constraints += [sum([eta[key] for key in eta]) == 1]
        for key in self.a:
            constraints += [
                M[key][0 : len(index), 0 : len(index)] == B[key],
                M[key][len(index), 0 : len(index)] == beta[key],
                M[key][len(index), len(index)] == eta[key],
            ]
        constraints += [
            M2[0 : len(index), 0 : len(index)] == cov_sub,
            M2[len(index), 0 : len(index)] == lambda_,
            M2[len(index), len(index)] == self.gamma1,
        ]
        for key in M:
            constraints += [M[key] >> 0]
        constraints += [M2 >> 0]
        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)
        w_opt = w.value
        beta_opt = {}
        for key in beta_opt:
            beta_opt[key] = beta[key].value
        pi_opt = pi.value
        lambda_opt = lambda_.value

        lambda_full = self.complete_lambda(
            Sigma=self.gamma1 * self.cov, lambda_=lambda_opt, index=index
        )
        beta_full = {}
        for key in beta:
            beta_full[key] = self.complete_beta(
                Sigma=self.gamma2 * self.cov,
                beta=beta[key].value,
                eta=eta[key].value,
                index=index,
            )

        x_dic = {}
        w_dic = {}
        w_full_dic = {}
        for i in range(len(index)):
            x_dic[index[i]] = w_opt[i] * self.delta_
        for i in range(self.num_stock):
            w_full_dic[i] = np.max(
                [sum([self.a[key] * beta_full[key][i] for key in self.a]) + pi_opt, 0]
            )
            if i not in index:
                x_dic[i] = 0
        return (x_dic, prob.value, w_full_dic, beta_full)


class MeanVarianceModel:
    def __init__(self, mu, cov, k, r_bar):
        self.mu = mu
        self.cov = cov
        self.num_stock = self.mu.shape[0]
        self.k = k
        self.r_bar = r_bar

    def solve(self):
        model = grb.Model("mean variance model")
        x = {}
        z = {}
        for i in range(self.num_stock):
            x[i] = model.addVar(vtype="C", name="x_" + str(i), ub=1)
            z[i] = model.addVar(vtype="B", name="z_" + str(i), ub=1)
        model.update()
        for i in range(self.num_stock):
            model.addConstr(x[i] <= z[i])
        model.addConstr(grb.quicksum(x[i] for i in range(self.num_stock)) == 1)
        model.addConstr(
            grb.quicksum(self.mu[i] * x[i] for i in range(self.num_stock)) >= self.r_bar
        )
        model.addConstr(grb.quicksum(z[i] for i in range(self.num_stock)) <= self.k)
        model.setObjective(
            grb.quicksum(
                self.cov[i, j] * x[i] * x[j]
                for j in range(self.num_stock)
                for i in range(self.num_stock)
            )
        )
        model.optimize()
        if model.Status == grb.GRB.OPTIMAL:
            z_opt = {}
            x_opt = {}
            for var in model.getVars():
                if "z_" in var.VarName:
                    i = int(var.VarName.split("_")[-1])
                    z_opt[i] = var.X
                elif "x_" in var.VarName:
                    i = int(var.VarName.split("_")[-1])
                    x_opt[i] = var.X
        return x_opt


class RobustModel:
    def __init__(self, mu, cov, k, kappa, sigma):
        self.mu = mu
        self.cov = cov
        self.num_stock = self.mu.shape[0]
        self.k = k
        self.kappa = kappa
        self.L = np.linalg.cholesky(self.cov)
        self.sigma = sigma

    def solve(self):
        model = grb.Model("robust model")
        x = {}
        z = {}
        for i in range(self.num_stock):
            x[i] = model.addVar(vtype="C", name="x_" + str(i), ub=1)
            z[i] = model.addVar(vtype="B", name="z_" + str(i), ub=1)
        v = model.addMVar(
            self.num_stock, vtype="C", lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY
        )
        aux = model.addVar(vtype="C", lb=0)
        model.update()
        for i in range(self.num_stock):
            model.addConstr(x[i] <= z[i])
            model.addConstr(
                v[i] == grb.quicksum(self.L[j, i] * x[i] for j in range(self.num_stock))
            )
        model.addConstr(grb.quicksum(x[i] for i in range(self.num_stock)) == 1)
        model.addConstr(
            aux * aux
            >= self.kappa * grb.quicksum(v[i] * v[i] for i in range(self.num_stock))
        )
        model.addConstr(grb.quicksum(z[i] for i in range(self.num_stock)) <= self.k)
        model.setObjective(
            grb.quicksum(
                self.cov[i, j] * x[i] * x[j]
                for j in range(self.num_stock)
                for i in range(self.num_stock)
            )
            - self.sigma
            * grb.quicksum(self.mu[i] * x[i] for i in range(self.num_stock))
            + self.sigma * aux
        )
        model.params.NumericFocus = 3
        model.optimize()
        if model.Status == grb.GRB.OPTIMAL:
            z_opt = {}
            x_opt = {}
            for var in model.getVars():
                if "z_" in var.VarName:
                    i = int(var.VarName.split("_")[-1])
                    z_opt[i] = var.X
                elif "x_" in var.VarName:
                    i = int(var.VarName.split("_")[-1])
                    x_opt[i] = var.X
        return x_opt


class BNTRobustModel:
    def __init__(self, mu, cov, k, kappa):
        self.mu = mu
        self.cov = cov
        self.num_stock = self.mu.shape[0]
        self.k = k
        self.kappa = kappa
        self.L = np.linalg.cholesky(self.cov)

    def solve(self):
        model = grb.Model("robust model")
        x = {}
        z = {}
        for i in range(self.num_stock):
            x[i] = model.addVar(vtype="C", name="x_" + str(i), ub=1)
            z[i] = model.addVar(vtype="B", name="z_" + str(i), ub=1)
        v = model.addMVar(
            self.num_stock, vtype="C", lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY
        )
        aux = model.addVar(vtype="C", lb=0)
        model.update()
        for i in range(self.num_stock):
            model.addConstr(x[i] <= z[i])
            model.addConstr(
                v[i] == grb.quicksum(self.L[j, i] * x[i] for j in range(self.num_stock))
            )
        model.addConstr(grb.quicksum(x[i] for i in range(self.num_stock)) == 1)
        model.addConstr(
            aux * aux >= grb.quicksum(v[i] * v[i] for i in range(self.num_stock))
        )
        model.addConstr(grb.quicksum(z[i] for i in range(self.num_stock)) <= self.k)
        model.setObjective(
            -grb.quicksum(self.mu[i] * x[i] for i in range(self.num_stock))
            + np.sqrt(self.kappa) * aux
        )
        model.params.NumericFocus = 3
        model.optimize()
        if model.Status == grb.GRB.OPTIMAL:
            z_opt = {}
            x_opt = {}
            for var in model.getVars():
                if "z_" in var.VarName:
                    i = int(var.VarName.split("_")[-1])
                    z_opt[i] = var.X
                elif "x_" in var.VarName:
                    i = int(var.VarName.split("_")[-1])
                    x_opt[i] = var.X
        return x_opt
