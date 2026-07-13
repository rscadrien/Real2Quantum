from pyqubo import Array
import pennylane as qml
import numpy as np
from pennylane import numpy as nppl
from collections import Counter
import networkx as nx
from real2quantum.base.QUBOProblem import QUBOProblem_Binary, QUBOProblem_multibit

class PortfolioOptimization_Binary(QUBOProblem_Binary):
    """
    QUBO formulation of the mean-variance portfolio optimization problem.

    This class encodes a portfolio selection problem as a quadratic
    unconstrained binary optimization (QUBO) problem, where each binary
    variable x_i ∈ {0,1} indicates whether asset i is included in the portfolio.

    The objective function is:

        minimize   x^T Σ x  -  λ μ^T x

    where:
    - Σ is the covariance matrix (risk term)
    - μ is the expected return vector
    - λ controls the risk-return tradeoff

    The formulation can be extended with constraints such as:
    - Budget constraint (fixed number of selected assets)
    - Sector constraints (soft or exact)

    Parameters
    ----------
    n : int
        Number of assets.
    mu : array-like
        Expected returns vector of shape (n,).
    Sigma : array-like
        Covariance matrix of shape (n, n).
    lam : float, optional
        Risk-return tradeoff parameter (default is 1.0).

    Attributes
    ----------
    mu : array-like
        Expected returns.
    Sigma : array-like
        Covariance matrix.
    lam : float
        Risk-return tradeoff parameter.

    Notes
    -----
    - A higher λ favors high-return portfolios (more aggressive).
    - A lower λ emphasizes risk minimization (more conservative).
    - Constraints are added as quadratic penalties to the QUBO.
    - Slack variables may be introduced for exact constraints, increasing
      the number of qubits required.
    """
    def __init__(self, n, mu, Sigma, lam=1.0):
        self.mu = mu
        self.Sigma = Sigma
        self.lam = lam

        super().__init__(n)  # calls _build_objective()

    def _build_objective(self):
        """
        Construct the mean-variance QUBO objective function.

        The objective encodes:

            x^T Σ x  -  λ μ^T x

        where:
        - The quadratic term (x^T Σ x) represents portfolio risk.
        - The linear term (μ^T x) represents expected return.

        Notes
        -----
        This method is automatically called during initialization and
        populates `self.H_pyqubo`.
        """
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

        The constraint is encoded as a quadratic penalty:

            P * (Σ_i x_i - K)^2

        which enforces that exactly K assets are selected in the solution.

        Parameters
        ----------
        P : float
            Penalty strength. Must be large enough to enforce the constraint.
        K : int
            Target number of selected assets.

        Notes
        -----
        - This is a *hard constraint* enforced via penalty.
        - Increasing P improves feasibility but may worsen conditioning.
        - Invalidates any previously compiled model.
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