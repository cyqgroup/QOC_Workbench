# QOC Workbench Knowledge Base Index

This index organizes the QOC Workbench knowledge base into reusable research modules for Hamiltonian families, control strategies, optimization algorithms, paper notes, predictive tools, diagnostics, and tutorials.

## Core Sections

### 1. Problems and Models
- [Grover Search](./01_Problems/Grover_Search.md)
- [3-SAT](./01_Problems/3-SAT.md)
- [Transverse Field Ising Model](./01_Problems/TFIM_Model.md)
- [Lipkin-Meshkov-Glick Model](./01_Problems/LMG_Model.md)
- [Hamming Weight Spike](./01_Problems/Hamming_Weight_Spike.md)
- [Quantum Chemistry Hamiltonians](./01_Problems/Quantum_Chemistry_Hamiltonians.md)
- [QUBO and Ising Models](./01_Problems/QUBO_Ising_Models.md)
- [Maximum-Weight Independent Set](./01_Problems/MWIS.md)

### 2. Evolution Strategies and Controls
- [Fourier Parameterization](./02_Strategies/Fourier_Parameterization.md)
- [Counter-Diabatic Driving](./02_Strategies/Counter-diabatic_Driving.md)
- [Adaptive Scheduling and FOAPT](./02_Strategies/Adaptive_Scheduling_FOAPT.md)
- [Floquet Variational Counter-Diabatic Driving](./02_Strategies/Floquet_VCD.md)
- [Quantum Catalysis](./02_Strategies/Quantum_Catalysis.md)
- [Non-Stoquastic Hamiltonians](./02_Strategies/Non_Stoquastic_Hamiltonians.md)
- [Bang-Bang Control](./02_Strategies/Bang-Bang_Control.md)

### 3. Optimization and Search Algorithms
- [Reinforcement-Learning-Based Adiabatic Design](./03_Algorithms/RL_Based_Adiabatic_Design.md)
- [PINN for QA Optimization](./03_Algorithms/PINN_for_QA_Optimization.md)

### 4. Paper Notes
Paper notes summarize control mechanisms, Hamiltonian families, useful diagnostics, and reusable ideas for the auditable search loop.

### 5. Predictive Tools
- [Gap Evolution Prediction](./05_Predictive_Tools/Gap_Evolution_Prediction.md)
- [Neural Quantum Digital Twins](./05_Predictive_Tools/Neural_Quantum_Digital_Twins.md)
- [Extrapolation Capability](./05_Predictive_Tools/Extrapolation_Capability.md)

### 6. Diagnostics and Visualization
- [Diagnostic Plots](./06_Visualization/Diagnostic_Plots.md)

### 7. Tutorials and Reviews
- [Quantum Control Tutorial](./07_Tutorials/Quantum_Control_Tutorial.md)

## Knowledge Compilation Rules

1. Use relative Markdown links for cross-references.
2. Include complete LaTeX formulas when a Hamiltonian, control term, or metric is defined.
3. Link new concepts back to existing problem, strategy, or paper notes whenever possible.
4. Keep entries executable: each strategy note should identify candidate controls, constraints, metrics, and expected artifacts when relevant.

Last updated: `2026-04-08`
