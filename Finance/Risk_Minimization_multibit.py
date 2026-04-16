from Portfolio_optimization_multibit import PortfolioOptimization_Multibit
import numpy as np
from pyqubo import Array

class RiskMinimization_Multibit(PortfolioOptimization_Multibit):

    def __init__(self, Sigma,m=5):
        super().__init__(mu=np.zeros(Sigma.shape[0]), Sigma=Sigma, m=m, lam=0.0)

    def _build_objective(self):
         w = self.weights()
         #risk terms
         self.H_pyqubo += sum(
             self.Sigma[i][j] * w[i] * w[j]
             for i in range(self.n)
             for j in range(self.n)
         )


    