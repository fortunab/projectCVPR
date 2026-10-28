# PolypGen v3 isolated data audit

This audit describes a **preliminary reviewed-sequence subset**, not an evaluation of the complete PolypGen release. It was reconstructed from the isolated copy of `PolypGen2021_MultiCenterData_v3 (1).zip`. The existing project data and outputs are outside this run's scope.

The authoritative current records are [eligibility_summary.json](eligibility_summary.json), [center_class_eligibility.csv](center_class_eligibility.csv), [manifest.audit.json](manifest.audit.json), [sequence_class_center_audit.csv](sequence_class_center_audit.csv), and [full_release_inventory.csv](full_release_inventory.csv). This document summarizes those records; it does not change the manifest, mapping, counts, or experiment protocol.

## Inventory and inclusion

The full-release inventory contains **8,037 original images**: 1,537 center-folder still images and 6,500 video-sequence frames. Inclusion is based on both annotation checks and reviewed center provenance.

| Inventory status | Original images |
| --- | ---: |
| Included reviewed sequences | 5,444 |
| Excluded still images without documented sequence/patient grouping | 1,535 |
| Excluded still images with annotation errors | 2 |
| Excluded sequences with conflicting center sources | 435 |
| Excluded sequence pending annotation review | 36 |
| Excluded sequences with unresolved center | 585 |
| **Total original images** | **8,037** |

The resulting manifest contains **5,444 frames from 37 sequences**, with 4,328 negative and 1,116 positive frame labels. All included frames have sequence metadata, and the strict manifest reports zero blocking rejections.

The manifest scanner separately reports 13,816 rejected image-file candidates: 11,223 files outside original-image leaves, 1,537 still frames excluded by the sequence-only policy, and 1,056 sequence frames excluded by reviewed-metadata policy. These scanner counts include non-original image files and must not be interpreted as 13,816 rejected original observations. The 8,037-row inventory is the record for original-image accounting. Likewise, `unresolved_sequences: []` in the strict manifest audit means that no unresolved sequence entered this manifest; it does not mean that all release mappings were resolved.

## Held-out-center eligibility

| Center | Negative frames | Positive frames | Sequences | Sequences containing negative frames | Sequences containing positive frames | Role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | 223 | 0 | 3 | 3 | 0 | Source only |
| C2 | 51 | 449 | 2 | 1 | 2 | Eligible target |
| C3 | 0 | 0 | 0 | 0 | 0 | Absent from subset |
| C4 | 1,409 | 144 | 10 | 9 | 4 | Eligible target |
| C5 | 782 | 199 | 4 | 4 | 1 | Eligible target |
| C6 | 1,863 | 324 | 18 | 18 | 8 | Eligible target |
| **Total** | **4,328** | **1,116** | **37** | **35** | **15** | **Four target folds** |

Sequence counts by class are overlapping: a mixed sequence can contain both negative and positive frames. They therefore need not sum to the number of sequences.

The eligible outer held-out centers are **C2, C4, C5, and C6**. In each outer fold, the corresponding center is excluded from fitting and source-only model selection. C1 can contribute negative source examples but is excluded from target-fold selection because it contains no positive frames in this subset. Two-class AUROC is undefined when the target contains only one class, and a C1 target could not assess sensitivity to positives. C3 has no included sequences. The four-fold mean and worst-center results must therefore be labeled as results across these four eligible centers, not across six complete PolypGen centers.

Eligibility establishes label coverage, not adequate numbers of independent groups. In particular, all 51 C2 negative frames occur in one sequence, and all 199 C5 positive frames occur in one sequence. Their corresponding class performance is especially sensitive to the content of that sequence.

## Center provenance and exact sequence exclusions

[reviewed_sequence_centers.csv](reviewed_sequence_centers.csv) records a source for every included sequence-center assignment. Evidence has two distinct forms:

| Evidence | Included sequences and center |
| --- | --- |
| Explicit center token in **every original frame filename** | Positive `seq16`–`seq23` → C6; negative `seq11_neg`–`seq13_neg` → C5 |
| Explicit sequence label and center color in Ali et al., Figure 7 | Positive `seq2, seq3, seq6, seq10` → C4; `seq5` → C5; `seq12, seq13` → C2 |
| Explicit sequence label and center color in Ali et al., Figure 7 | Negative `seq1_neg, seq2_neg, seq3_neg, seq8_neg, seq9_neg, seq10_neg` → C4; `seq4_neg, seq5_neg, seq6_neg` → C1; `seq14_neg`–`seq23_neg` → C6 |

The Figure 7 mappings are interpretations of explicitly displayed primary-paper labels and colors, not filename-derived release metadata. Positive `seq17`–`seq22` are supported by their explicit C6 filename tokens; their mapping is not inferred from the figure's ellipsis. The release README and directory listings contain no complete textual sequence-to-center mapping. The primary paper is [Ali et al., Scientific Data 2023, DOI: 10.1038/s41597-023-01981-y](https://doi.org/10.1038/s41597-023-01981-y).

The following exclusions were fixed before feature extraction:

- **Unresolved center:** positive `seq4, seq7, seq8, seq9, seq14, seq15`, totaling 585 frames. No center is assigned from image appearance, numbering, folder adjacency, or residual aggregate counts.
- **Conflicting center sources:** positive `seq11` (228 frames) and `seq7_neg` (207 frames), totaling 435 frames. Figure 7 colors `seq11` as C3, but the paper's center totals and actual still-image counts produce a contradiction involving its 228 frames. Figure 7 colors `seq7_neg` as C1, while the reported C1/C4 negative totals differ by exactly its 207 frames. These consistency checks do not establish a corrected assignment; neither sequence is reassigned by counts.
- **Pending annotation review:** positive-folder `seq1`, 36 frames. Its current masks and boxes consistently yield negative labels for all frames, and Figure 7 indicates C4. A separate annotation-quality concern remains unresolved; exclusion does not establish that its annotations are wrong. This is distinct from an unresolved center mapping.

The copied [polypgen_sequence_class_audit.json](polypgen_sequence_class_audit.json) is a **historical review artifact**, not the current inclusion record. It predates the decision to dispute `seq11` and still lists its C3 mapping as verified. The [v2 mapping review](polypgen_sequence_center_mapping_review_v2.json), current reviewed metadata, and fresh sequence audit supersede that historical assignment.

## Labels and annotation findings

For annotated frames, a mask is positive if at least one pixel has intensity greater than 127. Where a matching bounding-box annotation exists, polyp-presence decisions must agree. Unannotated frames in the release's documented `negativeOnly` sequences are assigned negative labels using that documented sequence role. Positive-folder membership alone does not establish a positive frame label.

The full inventory found two annotation errors, both in still images **outside the selected sequence scope**:

| Original image | Finding |
| --- | --- |
| `data_C2/images_C2/EndoCV2021_001010.jpg` | Mask/bounding-box presence disagreement |
| `data_C3/images_C3/C3_EndoCV2021_00489].jpg` | Missing annotation; a missing file is not a negative label |

These two rows are recorded as annotation-error exclusions rather than included in the 1,535 otherwise excluded still images. The fresh sequence audit reports zero annotation errors in the included sequences.

## Grouping, duplicate checks, and limits

The manifest preserves each video's sequence-folder group. Included rows have both file-byte SHA-256 and decoded RGB pixel SHA-256 records, together with annotation hashes where annotations exist. The duplicate guard joins entire groups when exact shared image content connects them and checks for incompatible duplicate labels. For this manifest, `duplicate_content_groups` is empty: **no exact byte or decoded-pixel duplicate group was reported among the 5,444 included frames**. This check does not detect all near-duplicates, establish patient independence, or certify content outside the included manifest.

Patient identities are unavailable. Distinct video sequences are not proven to come from distinct patients, so sequence-level splitting and uncertainty estimates must retain that limitation. The still-image center folders establish center provenance, but the five supplied single-frame metadata CSVs contain only `fileList`, `sizeType`, `sizeVal`, and `annotations`; they supply no documented patient or sequence ID. Still images are excluded from this initial experiment rather than assigned an invented patient identity.

The subset is smaller than the complete release, excludes unresolved and disputed data, and has substantial center-class imbalance. It supports a preliminary held-out-center comparison under the stated scope. It does not establish full-release performance, patient-independent generalization, or a clinical operating threshold. Current experiments use cached **ImageNet-pretrained ResNet18** features, as recorded in the eligibility summary; results should not be described as DINOv2 or encoder-adaptation experiments.

## Reproducibility identifiers

- Archive SHA-256: `a22f956a9f0fc3f941927108a3de8217fc61f1944c7eda598536dbabc27dbff3`
- Manifest SHA-256: `8f1753aae483e2c4bab59b72c6ae47df91d7bd32c4bbc2819aa2b6de58726ec0`
- Reviewed sequence-metadata SHA-256: `ac622dc38b81a7e0edd2f06f5e505d77ad53e8ca82b6b7c183ad78b8c0ee2eb6`

The manifest and reviewed metadata are the fixed inputs for the isolated run. See the existing [PROTOCOL.md](../PROTOCOL.md) for selection and evaluation rules; this README is a data-audit explanation and does not amend those rules.
