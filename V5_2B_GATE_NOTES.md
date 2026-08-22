# V5-2B diagnostic gate

The 30 immutable seeds are excluded from gradient updates and are evaluated only after the fixed training run. The gate uses unchanged thresholds 2=0.48, 3=0.60, 4=0.47 and exactly-one slot arbitration. Required PASS counts are 2/4 >= 8/10, 3/4 >= 8/10, 4/4 >= 9/10, denominator exact-4 >= 26/30. A PASS authorizes only the next validation-BBox stage; it does not open VAL automatically or permit production promotion.
