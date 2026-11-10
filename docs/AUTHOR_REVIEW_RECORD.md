# Author review record

This is a review form, not a completed sign-off. Automated test and replay evidence is recorded separately. No independent human clinical or code review is claimed by creating this file.

Before submission, record the actual reviewer, date, code revision and artifact hashes for each completed review. Preserve findings and resolutions, including checks that fail.

Keep a named sign-off record locally. During anonymous review, release only reviewer roles and an anonymized assurance summary; do not include author identities in the submission package.

| Review item | Evidence to inspect | Reviewer/date/revision | Findings and resolution |
|---|---|---|---|
| Cohort and label provenance | PolypGen inclusion/exclusion and center review, annotation checks; histology original metadata and hash selection. | Pending author entry | Patient linkage must not be inferred from sequence or filename similarity. |
| Prediction unit and clinical meaning | Frame presence, central-region tumor patch label; sequence eligibility and patient-macro endpoint definitions. | Pending author entry | Sampled patch positivity does not supply a clinical patient/slide diagnosis. |
| Implementation | Weight formula and spherical updates, candidate order, temperature, source validation, source/target disjointness, freeze-before-target execution. | Pending author entry | Inspect the code as well as the passing fixture tests. |
| Numeric evidence | Complete seed grid, raw predictions, recomputed metrics, paired bootstrap definitions, valid resampling counts, tuning budgets and measured fits. | Pending author entry | Separate seed SD, fixed-site group uncertainty and hospital variability. |
| Manuscript and figures | Every displayed table/coordinate against its versioned report; scoped conclusions, citations, official/custom distinction, page limit. | Pending author entry | Do not combine three-seed and ten-seed outputs without explicit labels. |
| Release and disclosure | Exact runtime locks/builds, data access instructions, anonymous release contents, checkpoint terms and actual GenAI assistance. | Pending author entry | Tokens and personal paths must not enter an anonymous public package. |

For a reviewed artifact, a useful entry is: reviewer; date; source revision/hash; prediction/report hashes; commands inspected or rerun; discrepancy; resolution; residual limitation. A second author review should be recorded only if it actually took place.
