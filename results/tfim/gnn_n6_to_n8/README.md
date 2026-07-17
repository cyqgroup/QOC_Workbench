# TFIM GNN N=6 to N=8 Transfer

Source artifact: `artifacts/20260616_173145_tfim_weighted_cd_gnn_n6_to_n8/`.

This directory summarizes the graph-neural weighted-CD generator trained on N=6 instances and evaluated on unseen N=8 instances.

## Files

- `coefficient_metrics.json`: coefficient-prediction metrics for the transfer task.
- `summary_by_method.csv`: downstream N=8 evolution metrics by method.
- `n8_evolution_results.csv`: per-instance N=8 evolution results.
- `n8_coefficient_scatter.svg`: N=8 coefficient scatter plot.
- `n8_per_instance_fidelity.svg`: N=8 per-instance final-fidelity comparison.
- `n8_fidelity_distribution.svg`: N=8 fidelity distribution.

The learned GNN parameters, training history, labels, and code snapshot are kept in the source artifact bundle.
