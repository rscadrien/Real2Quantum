from real2quantum.finance.Portfolio_optimization_multibit import PortfolioOptimization_Multibit
import numpy as np
from pyqubo import Array

class RiskMinimization_Multibit(PortfolioOptimization_Multibit):
    """
    QUBO formulation of a pure risk minimization portfolio problem using
    multibit encoding of asset weights.

    This class is a special case of `PortfolioOptimization_Multibit` where:
        - Expected returns are set to zero
        - The objective reduces to minimizing portfolio variance:
              w^T Sigma w

    No return term is included, making this suitable for constructing
    minimum-variance portfolios.

    Parameters
    ----------
    Sigma : array-like of shape (n, n)
        Covariance matrix of asset returns.
    m : int, optional (default=5)
        Number of bits used to encode each asset weight.
    """
    def __init__(self, Sigma,m=5):
        """
        Initialize the risk minimization problem.

        The expected returns vector is set to zero and the risk-return
        trade-off parameter is fixed to zero, effectively removing
        the return contribution from the objective.
        """
        super().__init__(mu=np.zeros(Sigma.shape[0]), Sigma=Sigma, m=m, lam=0.0)

    def _build_objective(self):
        """
        Construct the objective function corresponding to portfolio risk.

        The Hamiltonian includes only the quadratic risk term:
            sum_{i,j} Sigma[i,j] * w[i] * w[j]

        This overrides the parent implementation to explicitly remove
        any return contribution.
        """
        w = self.weights()
        #risk terms
        self.H_pyqubo += sum(
            self.Sigma[i][j] * w[i] * w[j]
            for i in range(self.n)
            for j in range(self.n)
        )


    