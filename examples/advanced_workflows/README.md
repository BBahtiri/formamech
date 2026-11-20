# Advanced Workflows Examples (P1–P6)

Ready-to-run JSON configuration inputs for the P1–P6 FormaMech CLI workflows:

- `notched_design_problem.json`: FEM-guarded design optimization (`formamech optimize bundles/notched-v1-cp-freeze --problem examples/advanced_workflows/notched_design_problem.json`)
- `active_plan.json` & `active_train_pool.json`: deterministic `TRAIN`-only active FEM selection (`formamech active select --plan examples/advanced_workflows/active_plan.json --pool examples/advanced_workflows/active_train_pool.json`)
- `viscoelastic_load_path.json`: ramp–hold–unload load path for `formamech stateful rollout`

See `docs/advanced-workflows.md` for complete commands and claim boundaries.
