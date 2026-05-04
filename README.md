# Real2Quantum v0.1
## Introduction
**Real2Quantum** is a package designed to make quantum computing more accessible for a wide range of applications.
It provides implementations of various optimization problems, with version v0.1 focusing on finance use cases such as portfolio optimization and risk minimization.
Future releases aim to expand into additional domains, including logistics, transportation, and energy distribution.
Real2Quantum is designed for the PennyLane framework. However, it can be useful for any quantum hardware by using the corresponding Pennylane plugin (https://pennylane.ai/devices/).

With Real2Quantum, you can define an optimization problem and incorporate domain-specific constraints in a natural way. The framework then automatically maps the problem into its quantum formulation by constructing the corresponding Hamiltonian.

## Installation
1. Clone this repository:
```bash
git clone https://github.com/rscadrien/Real2Quantum.git
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

## How to use it
We will illustrate for the portfolio optimization problem (binary version). The other implemented problems (risk minimization and the multibit variations) follows the same structure.  It is simply defined by the number of assets, the expected returns, the covariance matrix, and a risk–return trade-off parameter:

```bash
#Defining parameters
n = 5  # number of assets

mu = np.array([0.10, 0.12, 0.07, 0.09, 0.11])

Sigma = np.array([
    [0.10, 0.02, 0.01, 0.03, 0.02],
    [0.02, 0.08, 0.02, 0.01, 0.03],
    [0.01, 0.02, 0.09, 0.02, 0.01],
    [0.03, 0.01, 0.02, 0.07, 0.02],
    [0.02, 0.03, 0.01, 0.02, 0.06]
])
lam = 1.0
PFO_test = PortfolioOptimization_Binary(n= mu.size, mu=mu, Sigma=Sigma, lam=lam)
```
This creates a PFO_test object representing the optimization problem.

You can then add constraints directly through class methods. For instance, if you want to invest in exactly K assets out of the n available, you can include a budget constraint:

```bash
K=3
P = 5.0 # penalty parameter enforcing the constraint
PFO_test.add_budget_constraint(P,K)
```

When the problem is defined and all the constraints are included, you can obtain the Hamiltonian of your problem in the PennyLane framework. You can then use this Hamiltonian in your favorite quantum optimization algorithms (Grover, QAOA, etc.).

```bash
H = PFO_test.build_hamiltonian()
```
You can also visualize the graph corresponding to the optimization problem:

```bash
Graph = PFO_test.build_graph()
```
Finally, you can solve the optimization problem locally using the QAOA algorithm on a PennyLane simulator:
```bash
p=3 #Number of QAOA layer
Solution = PFO_test.solver(p)
```
The other implemented problems (risk minimization and the multibit variations) follows the same structure than PortfolioOptimization_Binary. 

## License
MIT Licence
