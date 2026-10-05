# Experiments

`configs/experiments.yaml` is the canonical experiment plan. The seven notebooks are interfaces to that plan and do not contain core algorithm implementations.

Parameter selection uses the deterministic 10% train holdout, 24 validation questions total (8 per dataset), fixed `K=50`, one seeded 10-trial Bayesian optimization over the simplex-constrained SAR weights, and the specified final labeled split. The selected parameters are written to a frozen JSON artifact.

Final experiments must load that frozen artifact and fail when it is missing rather than silently falling back to defaults.

The canonical ablation variants are:

`SAR-Full`, `SAR-Without-Relevance`, `SAR-Without-Gap`, `SAR-Without-Evidence`, `SAR-Without-Tuning`, and `Question-Only`.

Dynamic weighting remains an explicitly disabled extension in the canonical protocol.
