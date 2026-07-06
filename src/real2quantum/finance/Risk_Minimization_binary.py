from real2quantum.finance.Portfolio_optimization_binary import PortfolioOptimization_Binary
import numpy as np
from pyqubo import Array

class RiskMinimization_Binary(PortfolioOptimization_Binary):
    """
    Special case of PortfolioOptimization_Binary where only risk is minimized:
    Objective = x^T Σ x
    """

    def __init__(self, Sigma):
        super().__init__(mu=np.zeros(Sigma.shape[0]), Sigma=Sigma, lam=0.0)

    def _build_objective(self):
        """
        Construct the risk-only objective:
        Risk term: x^T Σ x
        """
        self.H_pyqubo = 0  # ensure reset

        self.H_pyqubo += sum(
            self.Sigma[i, j] * self.x[i] * self.x[j]
            for i in range(self.n)
            for j in range(self.n)
        )
