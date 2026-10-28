# Source-subset audit

All 48 source-only conditions are feasible and have distinct retained subsets. All observed class labels are retained in each source center; the complete original set of class-complete inner validation centers is maintained. Every target frame and sequence is unchanged.

| Target | Dominant source | Full source groups | 50% dominant retention: total groups | 25% dominant retention: total groups |
|---|---|---:|---:|---:|
| C2 | C6 | 35 | 26 | 21 |
| C4 | C6 | 27 | 18 | 13 |
| C5 | C6 | 33 | 24 | 19 |
| C6 | C4 | 19 | 14 | 11 |

Both arms have the same total group count within each target/retention pair. The 192 center-specific selections comprised 78 all-available choices and 114 uniform rejection samples. Maximum rejection attempts: 4; total rejected draws: 23. No exact-combination fallback was needed. The predeclared fallback and infeasible-quota behavior were checked with focused synthetic fixtures.

Quotas in the matched-scarcity arm start at minimum class-covering quotas. Remaining units are assigned to the available center with largest proportional-budget deficit, ties resolved by center identifier. No frame-level sampling or image validation fallback is used.

See conditions_plan.json for complete source/target indices, selected/dropped groups, sampling attempts, class counts, measured source composition and partition hashes. Measured_source_retention.csv and figures/measured_source_retention.png expose the difference between requested dominant-center retention, total group retention and total frame retention.
