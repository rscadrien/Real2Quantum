import pennylane as qml
from pyqubo import Array
import numpy as np
from pennylane import numpy as nppl
from collections import Counter
import networkx as nx
from Parent_class.QUBOProblem_multibit import QUBOProblem_multibit

class PortfolioOptimization_multibit(QUBOProblem_multibit):
    def __init__(self, mu, Sigma, m=5, lam=1.0):
        self.n = mu.size
        self.m = m
        self.mu = mu
        self.Sigma = Sigma
        self.lam = lam

        super().__init__()  # calls _build_objective()
        
    def _build_objective(self):
         w = self.weights()
         #risk terms
         self.H_pyqubo += sum(
             self.Sigma[i][j] * w[i] * w[j]
             for i in range(self.n)
             for j in range(self.n)
         )
         #return terms
         self.H_pyqubo += -self.lam *sum(self.mu[i] * w[i] for i in range(self.n))
        
    def add_normalization_constraint(self,P_norm):
        w = self.weights()
        budget = sum(w[i] for i in range(self.n))
        self.H_pyqubo += P_norm * (budget - 1)**2
        self._compiled = False
    
    def add_l2_constraint(self, P_l2, mode = "diversity"):
        w = self.weights()
        if mode == "diversity":
            sign = -1
        elif mode == "sparsity":
            sign = +1
        else :
            raise ValueError("mode must be 'diversity' or 'sparsity'")    
        term = sum(w[i] * w[i] for i in range(self.n))
        self.H_pyqubo += P_l2* sign* term
        self._compiled = False
        
    def _sector_constraint_soft(self, indices, w_sector, P):
        w = self.weights()
        sector_weight = sum(w[i] for i in indices)
        return P * (sector_weight - w_sector)**2
        
    def _sector_constraint_slack(self,  sector, indices, w_sector, P):
        w = self.weights()
        sector_weight = sum(w[i] for i in indices)        
        n_slack = 6
        z = Array.create(f"z_{sector}", shape=n_slack,  vartype="BINARY")

        slack = sum((2**(-(k+1))) * z[k] for k in range(n_slack))
        return P* (sector_weight+ slack -w_sector)**2
        
    def add_sector_constraint(self, sector_constraints, method="equal", P_sector=10):
        for sector, data in sector_constraints.items():
            indices = data["assets"]
            w_sector = data["w_sector"]

            if method == "equal":
                self.H_pyqubo += self._sector_constraint_soft(indices, w_sector, P_sector)
            elif method == "min":
                self.H_pyqubo += self._sector_constraint_slack(sector, indices, w_sector, P_sector)
            else:
                raise ValueError("Unknown method")
        self._compiled = False
    
    def add_upper_bound(self, u, P_upper):
        w = self.weights()
        self.H_pyqubo += P_upper * sum((w[i] - u[i])**2 for i in range(self.n))
        self._compiled = False

    def add_target_return(self, R_target, P_target):
        w = self.weights()

        ret = sum(self.mu[i] * w[i] for i in range(self.n))

        self.H_pyqubo += P_target * (ret - R_target)**2
        self._compiled = False

    def add_turnover_constraint(self, w_old, P_turnover):
        w = self.weights()

        self.H_pyqubo += P_turnover * sum((w[i] - w_old[i])**2 for i in range(self.n))
        self._compiled = False