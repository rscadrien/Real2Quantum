from pyqubo import Array
import pennylane as qml
import numpy as np
from pennylane import numpy as nppl
from collections import Counter
import networkx as nx
from Parent_class.QUBOProblem_Binary import QUBOProblem_Binary

class PortfolioOptimization_Binary(QUBOProblem_Binary):

    def __init__(self, n, mu, Sigma, lam=1.0):
        self.n = n
        self.mu = mu
        self.Sigma = Sigma
        self.lam = lam
        self.x = Array.create("x", shape=self.n, vartype="BINARY")

        super().__init__()  # calls _build_objective()

    def _build_objective(self):
        self.H_pyqubo += sum(
            self.Sigma[i, j] * self.x[i] * self.x[j]
            for i in range(self.n)
            for j in range(self.n)
        )
        self.H_pyqubo += -self.lam * sum(
            self.mu[i] * self.x[i] for i in range(self.n)
        )

    def add_budget_constraint(self, P, K):
        """
        Add a budget constraint enforcing exactly K selected assets.

        Constraint: (sum(x_i) - K)^2

        Parameters
        ----------
        P : float
            Penalty strength.
        K : int
            Desired number of selected assets.
        """
        self.H_pyqubo += P * (sum(self.x[i] for i in range(self.n))-K)**2
        self._compiled = False
    
    def _sector_constraint_soft(self, indices, K, P):
        """
        Soft sector constraint enforcing an approximate upper bound on selected assets.

        This method penalizes deviations from the constraint:

            sum(x_i for i in sector) ≈ K

        using a quadratic penalty term.

        Parameters
        ----------
        indices : list[int]
            Indices of assets belonging to the sector.
        K : int
            Maximum (target) number of selected assets in the sector.
        P : float
            Penalty strength controlling constraint enforcement.

        Returns
        -------
        pyqubo expression
            Quadratic penalty term enforcing the soft constraint.
        """
        s = sum(self.x[i] for i in indices)
        return P * (s-K)**2
    def _sector_constraint_slack(self,  sector, indices, K, P):
        """
        Hard sector constraint using slack variables to enforce an exact bound.

        This method enforces:

            sum(x_i for i in sector) <= K

        by introducing binary slack variables such that:

            sum(x_i) + slack = K

        where slack is represented in binary form.

        Parameters
        ----------
        sector : str
            Name/identifier of the sector (used for naming slack variables).
        indices : list[int]
            Indices of assets belonging to the sector.
        K : int
            Maximum number of allowed selected assets in the sector.
        P : float
            Penalty strength enforcing the constraint.

        Returns
        -------
        pyqubo expression
            Quadratic penalty term including slack variables.
        """
        s = sum(self.x[i]  for i in indices)

        n_slack = int(np.ceil(np.log2(K+1)))
        z = Array.create(f"z_{sector}", shape=n_slack,  vartype="BINARY")

        slack = sum((2**k)*z[k] for k in range(n_slack))
        return P* (s+ slack -K)**2
        
    def add_sector_constraint(self, sector_constraints, method="slack", P=10):
        """
        Add sector-level constraints to the portfolio optimization model.

        This method enforces per-sector constraints on the number of selected
        assets using either:
        - "soft": quadratic penalty approximation
        - "slack": exact constraint using binary slack variables

        Parameters
        ----------
        sector_constraints : dict
            Dictionary describing constraints per sector. Format:

                {
                    "Technology": {
                        "assets": [0, 1, 2, 5],
                        "max": 2
                    },
                    "Finance": {
                        "assets": [3, 4, 6],
                        "max": 1
                    }
                }

        method : str, optional
            Constraint method to use:
            - "soft": relaxed quadratic penalty
            - "slack": exact formulation with auxiliary binary variables
            Default is "slack".

        P : float, optional
            Penalty strength for constraint enforcement (default is 10).

        Raises
        ------
        ValueError
            If an unknown method is provided.

        Returns
        -------
        None
            Updates the internal QUBO Hamiltonian in-place.
        """
        for sector, data in sector_constraints.items():
            indices = data["assets"]
            K = data["max"]

            if method == "slack":
                self.H_pyqubo += self._sector_constraint_slack(sector, indices, K, P)
            elif method == "soft":
                self.H_pyqubo += self._sector_constraint_soft(indices, K, P)
            else:
                raise ValueError("Unknown method")
        self._compiled = False