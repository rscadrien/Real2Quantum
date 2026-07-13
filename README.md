# Real2Quantum v0.1
## Introduction
**Real2Quantum** is a package designed to make quantum computing more accessible for a wide range of applications.
It provides implementations of various optimization problems, with version v0.1 focusing on finance use cases such as portfolio optimization and risk minimization.
Future releases aim to expand into additional domains, including logistics, transportation, and energy distribution.
Real2Quantum is framework-agnostic and can generate the Hamiltonian in Pennylane, Qiskit, Cirq and D-wave format.

With Real2Quantum, you can define an optimization problem and incorporate domain-specific constraints in a natural way. The framework then automatically maps the problem into its quantum formulation by constructing the corresponding Hamiltonian. Other options are also available, such as solving the problem using the QAOA algorithm on a PennyLane simulator.

## Installation
```python
pip install real2quantum
```

## How to use it
We will illustrate for the portfolio optimization problem (binary version). The other implemented problems (risk minimization and the multibit variations) follows the same structure.  It is simply defined by the expected returns, the covariance matrix, and a risk–return trade-off parameter. The expected return and the covariance matrix can be calculated from the history of the closing prices in the stock market using the function calculate_portfolio_parameters :
```python
from real2quantum.finance.preprocessing import calculate_portfolio_parameters
import numpy as np
# Example: 5 days, 3 assets
prices = np.array([
    [100, 50, 200],
    [102, 51, 198],
    [101, 52, 202],
    [105, 51, 205],
    [103, 53, 210]
])
mu, Sigma = calculate_portfolio_parameters(prices)
```

```python
#Define the risk–return trade-off parameter
lam = 1.0
# Call the portfolio optimization
from real2quantum.finance.portfolio_optimization import PortfolioOptimization_Binary
PFO_test = PortfolioOptimization_Binary(n= mu.size, mu=mu, Sigma=Sigma, lam=lam)
```
This creates a `PFO_test` object representing the optimization problem.

You can then add constraints directly through class methods. For instance, if you want to invest in exactly K assets out of the n available, you can include a budget constraint:

```python
K=2
P = 5.0 # penalty parameter enforcing the constraint
PFO_test.add_budget_constraint(P,K)
```
`P` controls how strongly violations of the constraint are penalized. If `P` is too small, the optimizer may prefer a solution with a better objective value but a violated constraint. If P is very large, the constraint dominates the energy scale of the problem.

When the problem is defined and all constraints are included, you can derive the Hamiltonian of your problem. This Hamiltonian can then be used in your preferred quantum optimization algorithms (e.g., Grover’s algorithm, QAOA, etc.).

By setting the eco option, you can choose between different frameworks: PennyLane (default), Qiskit, Cirq  or D-Wave.

for Pennylane:
```python
H = PFO_test.build_hamiltonian(eco = 'PennyLane')
```
It returns a PennyLane-compatible Hamiltonian.
for Qiskit:
```python
H = PFO_test.build_hamiltonian(eco = 'Qiskit')
```
It returns a Qiskit-compatible representation.
for Cirq
```python
H = PFO_test.build_hamiltonian(eco = 'Cirq')
```
It returns a Cirq-compatible representation.
for D-wave
```python
h_dwave, J_dwave, offset_dwave = PFO_test.build_hamiltonian(eco = 'DWave')
```
It returns (h, J, offset) for D-Wave-style QUBO/Ising usage.

By default, the offset is included in the Hamiltonian. You can exclude it by setting `offset_incl=False`.

The translation of the real-world problem into a Hamiltonian is the main output of Real2Quantum. However, the framework also provides additional functionalities.

You can visualize the graph corresponding to the optimization problem as follows:

```python
Graph = PFO_test.build_graph()
```
Finally, you can solve the optimization problem locally using the QAOA algorithm on a PennyLane simulator:
```python
p=3 #Number of QAOA layer
Solution = PFO_test.solver(p)
```

## How to contribute to Real2Quantum
Real2Quantum is flexible enough to make it easy for anyone to create a new optimization problem for quantum computing. In the current version, there are two parent classes, QUBOProblem_Binary and QUBOProblem_Multibit, which contain common methods such as building the graph, constructing the Hamiltonian, and solving the problem using the QAOA algorithm.

To define a new type of optimization problems, follow these guidelines:
1. Choose the appropriate QUBO problem subclass: `QUBOProblem_Binary` or `QUBOProblem_Multibit`
2. Store all domain-specific parameters in the `__init__` method.
3. Implement the objective function in  `_build_objective`.
4. Define constraint methods that update `self.H_pyqubo`.
5. Whenever the Hamiltonian is modified, set `self._compiled = False` to ensure recompilation.

 For example, for portfolio optimization:
```python
class PortfolioOptimization_Binary(QUBOProblem_Binary):

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

        Constraint: (sum(x_i) - K)^2
        """
        self.H_pyqubo += P * (sum(self.x[i] for i in range(self.n))-K)**2
        self._compiled = False
# + others constraints
```
Real2Quantum aims to be an open-source project. Contributions are welcome, feel free to improve it by adding new types of optimization problems, new constraints to the existing ones, or new methods to the parent classes.
## License
MIT Licence
