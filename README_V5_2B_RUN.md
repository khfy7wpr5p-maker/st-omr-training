# V5-2B Operator Run Note

Run `notebooks/st_omr_meter_v5_2b_23_adaptation_colab.ipynb` only after this branch's CI is green.

Expected order:

1. Verify Pillow 12.3.0 and checkout the exact V5-2B branch.
2. Mount the existing Drive dataset and frozen specialist root.
3. Bind the already completed 15/15 human QA attestation.
4. Derive 600 approved staff-relative slot crops; 60 belong to the 30 diagnostic seeds and 540 to the 270 adaptation TRAIN samples.
5. Locate frozen 2/3/4 checkpoints by exact SHA.
6. Fine-tune only 2-AI and 3-AI with the frozen single-run CPU configuration.
7. Evaluate candidates on the untouched 30 diagnostic seeds with unchanged thresholds and frozen 4-AI.

Do not run a second hyperparameter configuration if the gate is HOLD. Report the gate evidence instead. VAL and FINAL_HOLDOUT remain closed.
