# Stage 7-D — Synthetic Curriculum V1

Stage 7-D extends the closed Stage 7-C baseline with the larger synthetic curriculum before real-data work.

## Frozen corpus identity

- source commit: `adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90`
- config fingerprint: `154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb`
- build id: `d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352`
- manifest SHA-256: `44a963cd7dbc612fa29c2953ea8b2c8776d89ce470074e8f8b3fe25c6e165f34`
- transport SHA-256: `4a9f3bb337ef99386081dff29c4c1fc3047dc3ada4db13c93b6254e680918e2b`
- 512 families / 1,536 images / 512 MusicXML targets
- 410 train / 51 validation / 51 test families

## Gates

7-D0 validates the small canonical export evidence object against the frozen identities above.

7-D1 will verify the copied archive and persisted corpus identities before the corpus can be used by the next run package.

7-D2 will use train and validation only. The 51-family test split stays sealed for the later benchmark decision.

This lane does not change Stage 8 real-data rights, provenance, admission, or duplicate/leakage requirements. It also does not add multivoice, piano, orchestral, cross-staff, tie/slur, tuplet, or nonzero-key support.
