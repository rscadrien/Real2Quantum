from abc import ABC, abstractmethod
import pennylane as qml
from pyqubo import Array
import numpy as np
from pennylane import numpy as nppl
from collections import Counter
import networkx as nx

class QUBOProblem_Binary(ABC):

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
        Must be implemented by subclasses.
        Should populate self.H_pyqubo.
        """
        pass
        
    def _compile(self):
        self.model = self.H_pyqubo.compile()
        self._compiled = True
        self.n_wires = len(self.model.variables)
        
    # =========================
    # Graph
    # =========================

    def get_qubo(self):
        if not self._compiled:
            self._compile()
        qubo, offset = self.model.to_qubo()
        return qubo, offset
    
    def get_ising(self):
        if not self._compiled:
            self._compile()
        h, J, offset = self.model.to_ising()
        return h, J, offset


    def build_graph(self):
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

        Returns
        -------
        qml.Hamiltonian
            Ising Hamiltonian corresponding to the QUBO.
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