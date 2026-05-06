from abc import ABC, abstractmethod
import pennylane as qml
from pyqubo import Array
import numpy as np
from pennylane import numpy as nppl
from collections import Counter
import networkx as nx

class QUBOProblem_Binary(ABC):
    """
    Abstract base class for defining and solving QUBO problems with binary variables.

    This class provides a full pipeline to:
    - Define a QUBO objective using PyQUBO
    - Compile it into QUBO and Ising representations
    - Convert it into a PennyLane Hamiltonian
    - Visualize the problem as a graph
    - Solve it using the QAOA algorithm

    Subclasses must implement the `_build_objective` method to define the
    problem-specific cost function.

    Parameters
    ----------
    n : int
        Number of binary decision variables.

    Attributes
    ----------
    n : int
        Number of original binary variables.
    x : pyqubo.Array
        Binary variable array.
    H_pyqubo : pyqubo expression
        Symbolic QUBO Hamiltonian.
    model : pyqubo.Model or None
        Compiled PyQUBO model.
    h : dict or None
        Linear coefficients of the Ising model.
    J : dict or None
        Quadratic coefficients of the Ising model.
    offset : float or None
        Energy offset of the Ising model.
    index : dict or None
        Mapping from variable names to qubit indices.
    n_wires : int or None
        Number of qubits (including auxiliary/slack variables).
    _compiled : bool
        Whether the model has been compiled.
    """

    def __init__(self,n):

        self.H_pyqubo = 0
        self.n = n
        self.x = Array.create("x", shape=self.n, vartype="BINARY")

        # compiled model caches
        self._compiled = False
        self.model = None
        self.index = None
        self.h = None
        self.J = None
        self.offset = None
        self.n_wires = None

        # Force child class to define objective
        self._build_objective()

    @abstractmethod
    def _build_objective(self):
        """
        Define the QUBO objective function.

        This method must be implemented by subclasses. It should construct
        the PyQUBO expression and assign it to `self.H_pyqubo`.

        Notes
        -----
        This method is automatically called during initialization.
        """    
        pass
        
    def _compile(self):
        """
        Compile the PyQUBO model.

        This converts the symbolic Hamiltonian into a concrete model that can
        be transformed into QUBO or Ising form.

        Notes
        -----
        This method is called automatically when needed.
        """
        self.model = self.H_pyqubo.compile()
        self._compiled = True
        self.n_wires = len(self.model.variables)
        
    # =========================
    # Graph
    # =========================

    def get_qubo(self):
        """
        Return the QUBO representation.

        Returns
        -------
        dict
            Dictionary of QUBO coefficients with keys (i, j).
        float
            Constant energy offset.
        """
        if not self._compiled:
            self._compile()
        qubo, offset = self.model.to_qubo()
        return qubo, offset
    
    def get_ising(self):
        """
        Return the Ising representation.

        Returns
        -------
        dict
            Linear coefficients h_i.
        dict
            Quadratic coefficients J_ij.
        float
            Constant energy offset.
        """
        
        if not self._compiled:
            self._compile()
        h, J, offset = self.model.to_ising()
        return h, J, offset


    def build_graph(self):
        """
        Build and visualize the QUBO interaction graph.

        Nodes represent variables and edges represent quadratic couplings.

        Returns
        -------
        networkx.Graph
            Graph representation of the QUBO problem.

        Notes
        -----
        - Node attributes store linear biases.
        - Edge attributes store interaction weights.
        """
        qubo, offset = self.get_qubo()
        G = nx.Graph()
        for (i, j), w in qubo.items():
            if i == j:
                G.add_node(i, bias=w)
            else:
                G.add_edge(i, j, weight=w)
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        # Format in scientific notation with 3 decimals
        formatted_edge_labels = {
            edge: f"{weight:.3e}" for edge, weight in edge_labels.items()
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels)
        return G
        
    # =========================
    # Hamiltonian
    # =========================   
    
    def build_hamiltonian(self):
        """
        Construct the PennyLane Hamiltonian from the Ising model.

        The method:
        - Converts QUBO → Ising
        - Orders variables (original first, auxiliary after)
        - Maps variables to qubit indices
        - Builds a Pauli-Z Hamiltonian

        Returns
        -------
        qml.Hamiltonian
            Hamiltonian suitable for variational quantum algorithms.

        Notes
        -----
        Auxiliary (slack) variables introduced during compilation are
        automatically included.
        """
        self.h, self.J, self.offset = self.get_ising()
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
        """
        Print the mapping between qubit indices and variables.

        Useful for interpreting measurement results.
        """

        inv_index = {i: v for v, i in self.index.items()}
        for i in range(self.n_wires):
            print(i, inv_index[i])

    def _build_mixer(self):
        """
        Construct the standard QAOA mixer Hamiltonian.

        The mixer is defined as:

            H_mixer = Σ_i X_i

        Returns
        -------
        qml.Hamiltonian
            Mixer Hamiltonian acting on all qubits.
        """
        if not self._compiled:
            self._compile()

        coeffs = [1.0] * self.n_wires
        ops = [qml.PauliX(i) for i in range(self.n_wires)]

        return qml.Hamiltonian(coeffs, ops)  

    def qaoa_circuit(self, dev, p):
        """
        Create a QAOA quantum circuit.

        Parameters
        ----------
        dev : qml.device
            PennyLane quantum device.
        p : int
            Number of QAOA layers.

        Returns
        -------
        function
            A QNode that evaluates the expectation value of the cost Hamiltonian.

        Notes
        -----
        The circuit:
        - Starts in a uniform superposition
        - Alternates cost and mixer layers
        - Returns ⟨H⟩
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
        Solve the QUBO problem using the QAOA algorithm.

        Parameters
        ----------
        p : int
            Number of QAOA layers.
        dev : str, optional
            Name of the PennyLane device.
        optimizer : qml.Optimizer, optional
            Classical optimizer (default: Adam).
        steps : int, optional
            Number of optimization steps.
        init_params : array-like, optional
            Initial parameters for QAOA.
        num_samples : int, optional
            Number of measurement samples.

        Returns
        -------
        dict
            Dictionary containing:
            - 'gammas_opt': optimal gamma parameters
            - 'betas_opt': optimal beta parameters
            - 'energy': final energy
            - 'history': energy evolution
            - 'top_solutions': most frequent bitstrings
            - 'n_qubits': number of qubits used

        Notes
        -----
        The solver:
        1. Optimizes QAOA parameters
        2. Samples the optimized circuit
        3. Returns the most probable solutions
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