# TFIM Curated Results

This directory collects the manuscript-facing random transverse-field Ising model results in one place. The full audit bundles remain under `artifacts/`.

## Subdirectories

- `weighted_cd_neural_generator/`: MLP-style weighted-CD coefficient generator and downstream evolution on N=6 test instances.
- `gnn_n6_to_n8/`: graph-neural generator trained on N=6 instances and evaluated on unseen N=8 instances.
- `gnn_n6_to_n8_T2p5_eval/`: longer-time T=2.5 evaluation of the N=6-to-N=8 GNN generator.

These summaries document both coefficient-prediction metrics and downstream physical evolution metrics.
