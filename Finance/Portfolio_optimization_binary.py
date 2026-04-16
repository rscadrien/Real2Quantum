from pyqubo import Array
import pennylane as qml
import numpy as np
from pennylane import numpy as nppl
from collections import Counter
class PortfolioOptimization_Binary:
    """
    Portfolio Optimization formulated as a QUBO problem and solved using QAOA.

    This class builds a quadratic unconstrained binary optimization (QUBO)
    formulation of a portfolio selection problem and converts it into a
    quantum Hamiltonian compatible with PennyLane.

    Attributes
    ----------
    n : int
        Number of assets.
    mu : array-like
        Expected returns of the assets.
    Sigma : array-like
        Covariance matrix representing risk between assets.
    lam : float
        Risk-return tradeoff parameter.
    x : pyqubo Array
        Binary decision variables (0 or 1 for each asset).
    H_pyqubo : pyqubo expression
        QUBO Hamiltonian.
    """
    
    def __init__(self, mu, Sigma, lam=1.0):
        """
        Initialize the portfolio optimization problem.

        Parameters
        ----------
        n : int
            Number of assets.
        mu : array-like, optional
            Expected returns.
        Sigma : array-like, optional
            Covariance matrix.
        lam : float, optional
            Risk-return tradeoff parameter (default is 1.0).
        """
        self.n = mu.size
        self.mu = mu
        self.Sigma = Sigma
        self.lam = lam
        self.x = Array.create("x", shape=self.n, vartype="BINARY")
        
        self.H_pyqubo = 0
        
        # compiled model caches
        self._compiled = False
        self.index = None
        self.h = None
        self.J = None
        self.offset = None
        self.n_wires = None
        
        
        #build objective immediately
        self._build_objective()
    # =========================
    # Objective
    # =========================
        
    def _build_objective(self):
        """
        Construct the portfolio objective function:

        Objective = Risk - λ * Return

        Risk term: x^T Σ x
        Return term: μ^T x
        """
        self.H_pyqubo += sum(self.Sigma[i,j] *self.x[i]*self.x[j]
           for i in range(self.n)
           for j in range(self.n))
        self.H_pyqubo += -self.lam *sum(self.mu[i]*self.x[i] for i in range(self.n))

     # =========================
    # Constraints
    # =========================   
    
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

    # =========================
    # Compilation layer (CRITICAL FIX)
    # =========================
    def _compile(self):
        model = self.H_pyqubo.compile()
        self.h, self.J, self.offset = model.to_ising()

        # collect ALL variables (including slack)
        all_vars = set(self.h.keys())
        for (v1, v2) in self.J.keys():
            all_vars.add(v1)
            all_vars.add(v2)
        # --- enforce ordering ---
        asset_vars = [f"x[{i}]" for i in range(self.n) if f"x[{i}]" in all_vars]
        slack_vars = sorted(v for v in all_vars if v not in asset_vars)
        ordered_vars = asset_vars + slack_vars

        self.index = {v: i for i, v in enumerate(ordered_vars)}
        self.n_wires = len(self.index)

        self._compiled = True
            
    # =========================
    # Hamiltonian
    # =========================   
    
    def build_hamiltonian(self):
        """
        Compile the QUBO model and convert it into a PennyLane Hamiltonian.

        Returns
        -------
        qml.Hamiltonian
            Ising Hamiltonian corresponding to the QUBO.
        """
        if not self._compiled:
            self._compile()

        coeffs = []
        ops = []

        # linear terms
        for var, h_val in self.h.items():
            coeffs.append(h_val)
            ops.append(qml.PauliZ(self.index[var]))

        # quadratic terms
        for (v1, v2), J_val in self.J.items():
            coeffs.append(J_val)
            ops.append(qml.PauliZ(self.index[v1]) @ qml.PauliZ(self.index[v2]))

        return qml.Hamiltonian(coeffs, ops)
        
    def print_variable_order(self):
        inv_index = {i: v for v, i in self.index.items()}
        for i in range(self.n_wires):
            print(i, inv_index[i])

    def _build_mixer(self):
        """
        Build the standard QAOA mixer Hamiltonian:

        H_mixer = Σ X_i

        Returns
        -------
        qml.Hamiltonian
            Mixer Hamiltonian.
        """
        if not self._compiled:
            self._compile()

        coeffs = [1.0] * self.n_wires
        ops = [qml.PauliX(i) for i in range(self.n_wires)]

        return qml.Hamiltonian(coeffs, ops)  

    def qaoa_circuit(self, dev, p):
        """
        Construct the QAOA circuit.

        Parameters
        ----------
        dev : qml.device
            Quantum device.
        p : int
            Number of QAOA layers.

        Returns
        -------
        function
            A QNode representing the QAOA circuit.
        """
        H = self.build_hamiltonian()
        H_mixer = self._build_mixer()
        n_wires = self.n_wires

        @qml.qnode(dev)
        def circuit(gammas, betas):

            for i in range(n_wires):
                qml.Hadamard(i)

            for k in range(p):
                qml.qaoa.cost_layer(gammas[k], H)
                qml.qaoa.mixer_layer(betas[k], H_mixer)

            return qml.expval(H)

        return circuit

    def solver(self, p, dev ="default.qubit", optimizer=None, 
               steps=100, init_params=None, num_samples=1000):
        """
        Solve the portfolio optimization problem using QAOA.

        Parameters
        ----------
        p : int
            Number of QAOA layers.
        dev : str
            Quantum device name.
        optimizer : qml optimizer
            Classical optimizer.
        steps : int
            Number of optimization steps.
        init_params : array-like, optional
            Initial QAOA parameters.
        num_samples : int
            Number of samples for solution extraction.

        Returns
        -------
        dict
            Dictionary containing optimal parameters, energy, and top solutions.
        """
        if optimizer is None:
            optimizer = qml.AdamOptimizer(stepsize=0.1)

        # ensure compilation BEFORE device creation
        if not self._compiled:
            self._compile()

        dev = qml.device(dev, wires=self.n_wires)
        circuit = self.qaoa_circuit(dev, p)

        if init_params is None:
            params = nppl.random.uniform(0, 1, 2 * p, requires_grad=True)
        else:
            params = init_params

        def cost_fn(params):
            gammas = params[:p]
            betas = params[p:]
            return circuit(gammas, betas)

        energy_history = []

        for step in range(steps):
            params = optimizer.step(cost_fn, params)
            energy = cost_fn(params)
            energy_history.append(energy)

            if step % 10 == 0:
                print(f"Step {step}: Energy = {energy}")

        gammas_opt = params[:p]
        betas_opt = params[p:]

        # =========================
        # Sampling
        # =========================
        H = self.build_hamiltonian()

        @qml.qnode(dev, shots=num_samples)
        def sample_circuit(gammas, betas):

            for i in range(self.n_wires):
                qml.Hadamard(i)

            H_mixer = self._build_mixer()

            for k in range(p):
                qml.qaoa.cost_layer(gammas[k], H)
                qml.qaoa.mixer_layer(betas[k], H_mixer)

            return qml.sample(wires=range(self.n_wires))

        samples = sample_circuit(gammas_opt, betas_opt)

        bitstrings = ["".join(map(str, sample)) for sample in samples]
        bitstrings_corrected = [
            "".join(str(1 - int(b)) for b in s)
            for s in bitstrings
        ]

        counts = Counter(bitstrings_corrected)
        top_5 = counts.most_common(5)

        return {
            "gammas_opt": gammas_opt,
            "betas_opt": betas_opt,
            "energy": energy_history[-1],
            "history": energy_history,
            "top_solutions": top_5,
            "n_qubits": self.n_wires,
        }