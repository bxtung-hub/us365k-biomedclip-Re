# Dataset split evidence

The preserved manifest audit reports:

- train: 218,402 records, 7,005 unique cases;
- validation: 74,044 records, 2,336 unique cases;
- test: 71,919 records, 2,335 unique cases;
- case overlap: 0 for train-validation, train-test, and validation-test;
- image overlap: 0 for the same three comparisons;
- total manifest errors: 0.

Accordingly, the supported statement is **case-disjoint and image-disjoint**. This repository does not upgrade that statement to patient-disjoint because the preserved evidence does not establish that `case_id` is a patient identifier.
