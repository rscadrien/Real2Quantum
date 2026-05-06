import pennylane as qml
from pyqubo import Array
import numpy as np
from pennylane import numpy as nppl
from collections import Counter
import networkx as nx
from Parent_class.QUBOProblem_multibit import QUBOProblem_multibit

class PortfolioOptimization_multibit(QUBOProblem_multibit):
    """
    QUBO formulation of a portfolio optimization problem using a multibit encoding
    of asset weights.

    The objective function encodes a mean-variance trade-off:
        risk - lambda * return

    where:
        - risk is given by w^T Sigma w
        - return is given by mu^T w

    Additional constraints (normalization, sector exposure, turnover, etc.)
    can be added as quadratic penalty terms.

    Parameters
    ----------
    n : int
        Number of assets.
    mu : array-like of shape (n,)
        Expected returns of the assets.
    Sigma : array-like of shape (n, n)
        Covariance matrix of asset returns.
    m : int, optional (default=5)
        Number of bits used to encode each asset weight.
    lam : float, optional (default=1.0)
        Risk-return trade-off parameter. Higher values favor return over risk.
    """
    def __init__(self, n, mu, Sigma, m=5, lam=1.0):
        self.mu = mu
        self.Sigma = Sigma
        self.lam = lam
        super().__init__(n,m)  # calls _build_objective()
        
    def _build_objective(self):
        """
        Construct the base objective function.

        The Hamiltonian includes:
            - Quadratic risk term: sum_{i,j} Sigma[i,j] * w[i] * w[j]
            - Linear return term: -lambda * sum_i mu[i] * w[i]

        This method is automatically called during initialization.
        """
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
        """
        Add a budget (normalization) constraint enforcing sum_i w[i] = 1.

        Parameters
        ----------
        P_norm : float
            Penalty strength for enforcing the normalization constraint.
        """
        w = self.weights()
        budget = sum(w[i] for i in range(self.n))
        self.H_pyqubo += P_norm * (budget - 1)**2
        self._compiled = False
    
    def add_l2_constraint(self, P_l2, mode = "diversity"):
        """
        Add an L2 regularization constraint.

        Parameters
        ----------
        P_l2 : float
            Penalty strength for the L2 constraint.
        mode : str, optional (default="diversity")
            Mode for the L2 constraint. Can be "diversity" or "sparsity".
        """
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
        """
        Build a soft equality constraint for a sector allocation.

        Parameters
        ----------
        indices : list of int
            Indices of assets belonging to the sector.
        w_sector : float
            Target total weight for the sector.
        P : float
            Penalty strength.

        Returns
        -------
        pyqubo expression
            Quadratic penalty enforcing sum_{i in sector} w[i] ≈ w_sector.
        """
        w = self.weights()
        sector_weight = sum(w[i] for i in indices)
        return P * (sector_weight - w_sector)**2
        
    def _sector_constraint_slack(self,  sector, indices, w_sector, P):
        """
        Build a sector constraint using slack variables to enforce a lower bound.

        This encodes:
            sum_{i in sector} w[i] >= w_sector

        via binary slack variables.

        Parameters
        ----------
        sector : str
            Name of the sector (used for variable naming).
        indices : list of int
            Indices of assets in the sector.
        w_sector : float
            Minimum required weight for the sector.
        P : float
            Penalty strength.

        Returns
        -------
        pyqubo expression
            Quadratic penalty with slack variables.
        """
        w = self.weights()
        sector_weight = sum(w[i] for i in indices)        
        n_slack = 6
        z = Array.create(f"z_{sector}", shape=n_slack,  vartype="BINARY")

        slack = sum((2**(-(k+1))) * z[k] for k in range(n_slack))
        return P* (sector_weight+ slack -w_sector)**2
        
    def add_sector_constraint(self, sector_constraints, method="equal", P_sector=10):
        """
        Add sector allocation constraints.

        Parameters
        ----------
        sector_constraints : dict
            Dictionary defining sector constraints. Format:
                {
                    "sector_name": {
                        "assets": list of indices,
                        "w_sector": float
                    },
                    ...
                }
        method : {"equal", "min"}, optional
            Constraint type:
                - "equal": enforce exact allocation
                - "min": enforce minimum allocation using slack variables
        P_sector : float, optional
            Penalty strength.
        """
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
        """
        Add upper bound constraints on asset weights.

        Parameters
        ----------
        u : array-like of shape (n,)
            Upper bounds for each asset weight.
        P_upper : float
            Penalty strength.
        """
        w = self.weights()
        self.H_pyqubo += P_upper * sum((w[i] - u[i])**2 for i in range(self.n))
        self._compiled = False

    def add_target_return(self, R_target, P_target):
        """
        Add a constraint enforcing a target portfolio return.

        Parameters
        ----------
        R_target : float
            Desired portfolio return.
        P_target : float
            Penalty strength.
        """
        w = self.weights()

        ret = sum(self.mu[i] * w[i] for i in range(self.n))

        self.H_pyqubo += P_target * (ret - R_target)**2
        self._compiled = False

    def add_turnover_constraint(self, w_old, P_turnover):
        """
        Add a turnover constraint penalizing deviations from a previous portfolio.

        Parameters
        ----------
        w_old : array-like of shape (n,)
            Previous portfolio weights.
        P_turnover : float
            Penalty strength controlling how strongly turnover is penalized.
        """
        w = self.weights()

        self.H_pyqubo += P_turnover * sum((w[i] - w_old[i])**2 for i in range(self.n))
        self._compiled = False