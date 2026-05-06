from abc import ABC, abstractmethod
import pennylane as qml
from pyqubo import Array
import numpy as np
from pennylane import numpy as nppl
from collections import Counter
import networkx as nx

class QUBOProblem_multibit(ABC):
    """
    Abstract base class for multi-bit encoded QUBO problems.

    This class provides a framework to define optimization problems where
    each logical variable is encoded using multiple binary variables
    (bit decomposition). The resulting QUBO formulation is constructed
    using `pyqubo`, and can be converted to an Ising Hamiltonian suitable
    for quantum algorithms such as QAOA.

    Parameters
    ----------
    n : int
        Number of logical variables.
    m : int
        Number of binary bits used to encode each variable.

    Attributes
    ----------
    n : int
        Number of logical variables.
    m : int
        Number of bits per variable.
    b : pyqubo.Array
        Binary decision variables of shape (n, m).
    alpha : list of float
        Normalized weights used to reconstruct real-valued variables
        from their binary encoding.
    H_pyqubo : pyqubo expression
        Symbolic QUBO Hamiltonian.
    model : pyqubo.Model or None
        Compiled pyqubo model.
    index : dict or None
        Mapping from variable names to qubit indices.
    h : dict or None
        Linear coefficients of the Ising Hamiltonian.
    J : dict or None
        Quadratic coefficients of the Ising Hamiltonian.
    offset : float or None
        Constant offset of the Hamiltonian.
    n_wires : int or None
        Number of qubits required for the problem.
    _compiled : bool
        Whether the pyqubo model has been compiled.
    """
    
    def __init__(self,n,m):
        self.H_pyqubo = 0
        self.n = n
        self.m = m
        #Creation of binary variables
        self.b = Array.create('b', shape=(self.n,self.m), vartype='BINARY')
        #coefficients alpha_k = 2^{-k} (normalized)
        self.alpha = [2**(-k) for k in range(self.m)]
        norm = sum(self.alpha)
        self.alpha = [a / norm for a in self.alpha]
        # compiled model caches
        self._compiled = False
        self.model = None
        self.index = None
        self.h = None
        self.J = None
        self.offset = None
        self.n_wires = None
        #build objective immediately
        self._build_objective()
    @abstractmethod
    def _build_objective(self):
        """
        Must be implemented by subclasses.
        Should populate self.H_pyqubo.
        """
        pass
        
    def weights(self):
        """
        Construct symbolic expressions for the encoded variables.

        Each logical variable is reconstructed as a weighted sum of its
        binary representation.

        Returns
        -------
        list
            List of symbolic expressions representing the reconstructed
            variables.
        """
        return [
            sum(self.alpha[k] * self.b[i][k] for k in range(self.m))
            for i in range(self.n)
        ]
    def _bitstring_to_weights(self, bitstring):
        """
        Convert a measured bitstring into decoded variable values.

        Parameters
        ----------
        bitstring : str or array-like
            Binary string representing a sampled solution.

        Returns
        -------
        np.ndarray
            Array of reconstructed variable values of size (n,).
        """
        w = np.zeros(self.n)
        for i in range(self.n):
            for k in range(self.m):
                idx = self.index[f"b[{i}][{k}]"]
                w[i] += self.alpha[k] * bitstring[idx]
        return w
    # =========================
    # Compilation layer
    # =========================

    def _compile(self):
        """
        Compile the symbolic QUBO model using pyqubo.

        This step transforms the symbolic Hamiltonian into a numerical
        model that can be converted into QUBO or Ising form.
        """
        self.model = self.H_pyqubo.compile()
        self.n_wires = len(self.model.variables)
        self._compiled = True
    
    def get_qubo(self):
        """
        Retrieve the QUBO representation of the problem.

        Returns
        -------
        dict
            Dictionary mapping variable pairs to QUBO coefficients.
        float
            Constant offset of the QUBO.
        """
        if not self._compiled:
            self._compile()
        qubo, offset = self.model.to_qubo()
        return qubo, offset
    
    def get_ising(self):
        """
        Retrieve the Ising representation of the problem.

        Returns
        -------
        dict
            Linear coefficients h of the Ising Hamiltonian.
        dict
            Quadratic coefficients J of the Ising Hamiltonian.
        float
            Constant offset of the Ising Hamiltonian.
        """
        if not self._compiled:
            self._compile()
        h, J, offset = self.model.to_ising()
        return h, J, offset

    # =========================
    # Graph
    # =========================               

    def build_graph(self):
        """
        Construct and visualize the QUBO interaction graph.

        Nodes represent variables with associated biases, and edges
        represent pairwise interactions.

        Returns
        -------
        networkx.Graph
            Graph representation of the QUBO problem.
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
        Compile the QUBO model and convert it into a PennyLane Hamiltonian.

        The method maps QUBO variables to qubits, separates logical and
        auxiliary (slack) variables, and constructs the corresponding
        Ising Hamiltonian.

        Returns
        -------
        qml.Hamiltonian
            Ising Hamiltonian corresponding to the QUBO problem.
        """
        self.h, self.J, self.offset = self.get_ising()

        # collect ALL variables (including slack)
        all_vars = set(self.h.keys())
        for (v1, v2) in self.J.keys():
            all_vars.add(v1)
            all_vars.add(v2) 
        # 1. asset variables (b[i][k])
        asset_vars = []
        for i in range(self.n):
            for k in range(self.m):
                name = f"b[{i}][{k}]"
                if name in all_vars:
                    asset_vars.append(name)

        # 2. slack variables (everything else)
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
        Print the mapping between qubit indices and variable names.

        This is useful for interpreting measurement results and debugging
        the variable-to-qubit assignment.
        """
        inv_index = {i: v for v, i in self.index.items()}
        for i in range(self.n_wires):
            print(i, inv_index[i])

    # =========================
    # QAOA circuit
    # ========================= 
    
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

    # =========================
    # Solver
    # ========================= 

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
        
        decoded_top_5 = []
        for bitstring, count in top_5:
            weights = self._bitstring_to_weights(bitstring)
            decoded_top_5.append({
            "bitstring": bitstring,
            "count": count,
            "weights": weights,
        })
        return {
            "gammas_opt": gammas_opt,
            "betas_opt": betas_opt,
            "energy": energy_history[-1],
            "history": energy_history,
            "top_solutions": decoded_top_5,
            "n_qubits": self.n_wires,
        }