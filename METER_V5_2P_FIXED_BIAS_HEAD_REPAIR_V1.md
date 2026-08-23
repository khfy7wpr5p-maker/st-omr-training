# Meter V5-2P — Frozen-Backbone Fixed-Bias Head Repair V1

## Purpose

V5-2N and V5-2O show that the exact frozen 64D digit-specialist representations retain strong V5 class information while the unchanged frozen heads place the V5 domain far below the existing decision boundary.

The repair question is therefore narrowed to the smallest shared-runtime trainable surface that can rotate the decision direction without globally translating the classifier:

- freeze the complete convolutional feature extractor;
- freeze the existing scalar head bias;
- train only the 64-element `head.weight` for 2-AI and 3-AI;
- keep the frozen runtime thresholds unchanged.

Freezing `head.bias` is deliberate. V5-2O measured a large source-to-V5 placement shift, but changing a shared scalar bias would also translate historical-source logits and is threshold-equivalent global calibration drift. V5-2P instead tests whether a small head-direction rotation can use the already-measured frozen representation to accommodate both domains.

## Exact evidence prerequisite

V5-2P requires the completed V5-2N and V5-2O TRAIN-only reports. In particular:

- 2-AI V5 rank AUC remains effectively perfect and the unchanged frozen head direction strictly separates V5 TRAIN;
- 3-AI V5 rank AUC remains at least 0.999 while its strict gap is slightly non-positive;
- both specialists preserve source-to-V5 class-gap direction;
- both V5 positive and negative sets remain below the unchanged frozen decision boundary;
- neither prior audit selected a threshold, bias, classifier, architecture, or repair training setting.

If these carried-forward facts are absent or materially changed, V5-2P fails closed.

## Trainable surface

For each of 2-AI and 3-AI independently:

- `features.*`: frozen and required bit-identical to the source checkpoint;
- `head.bias`: frozen and required bit-identical to the source checkpoint;
- `head.weight`: the only trainable tensor, 64 parameters;
- 4-AI: frozen control and never trained.

No residual branch, extra layer, domain router, source/V5 provenance input, or runtime architecture change is introduced.

## TRAIN surfaces

Only already-open TRAIN data are used:

- V5 adaptation TRAIN: exactly 540 approved existing slot crops, 90 positives / 450 negatives per specialist;
- historical M4A TRAIN: exactly 26,964 rows through the frozen historical crop/preprocess contract.

The 30 V5 diagnostic seeds receive zero gradient updates and are opened only after historical retention passes. The 900 reserved V5 TRAIN examples remain closed. V5 VALIDATION and FINAL_HOLDOUT remain closed.

## Objective

The fixed full-batch objective is, independently for each specialist:

`L = 0.5 * mean(BCE_w1(V5_adaptation_train)) + 0.5 * mean(BCE_w1(M4A_historical_train))`

Properties:

- equal domain weighting removes raw 26,964-vs-540 count dominance;
- within each domain, the observed binary class frequency is preserved;
- `pos_weight = 1.0`;
- no replay ratio;
- no class reweighting;
- no threshold or bias optimization;
- no hyperparameter sweep.

## Solver

The head-only problem is convex in the 64 trainable weights because the frozen features and frozen bias are constants.

Use deterministic CPU full-batch PyTorch LBFGS:

- initialization: exact frozen `head.weight`;
- `lr = 1.0`;
- `max_iter = 100`;
- `max_eval = 125`;
- `history_size = 20`;
- `tolerance_grad = 1e-9`;
- `tolerance_change = 1e-12`;
- `line_search_fn = strong_wolfe`;
- no weight decay;
- no momentum;
- no minibatch sampling;
- no checkpoint selection or early-stopping sweep.

The fixed final solver state is the only candidate. Non-finite loss/weights, unexpected trainable tensors, changed frozen tensors, or solver-contract drift fail closed.

## Gate order

1. **Historical M4A VALIDATION retention first**, using V5-2M corrected relative retention contract at unchanged thresholds:
   - F1 drop <= 0.005;
   - recall drop <= 0.005;
   - precision drop <= 0.005.
2. Only if retention PASS, evaluate the immutable first-30 V5 diagnostic:
   - 2/4 >= 8/10;
   - 3/4 >= 8/10;
   - 4/4 >= 9/10;
   - denominator exact-4 >= 26/30.
3. PASS of both gates may justify a separately reviewed V5 VALIDATION-opening stage. It does not itself open validation or authorize production.

There is no automatic second configuration.

## Safety boundary

V5-2P authorizes no:

- convolutional/backbone update;
- head-bias update;
- threshold tuning;
- alternative threshold evaluation;
- new BBox;
- new crop geometry;
- new spatial heuristic;
- old D11 glyph/window reuse;
- 900 reserved TRAIN opening;
- V5 VALIDATION opening;
- FINAL_HOLDOUT opening;
- 4-AI mutation;
- Resolver wiring;
- runtime domain routing;
- production promotion.

Candidate checkpoints are evidence artifacts only until all later gates explicitly pass.