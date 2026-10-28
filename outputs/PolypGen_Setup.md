# PolypGen access and CBMP setup

The official dataset page is https://www.synapse.org/Synapse:syn26376615/wiki/613312.
Its revised download is **syn45200214, version 1**, named
`PolypGen2021_MultiCenterData_v3.zip`.

On October 17, 2026, an existing archive was found in
`C:\Users\Lenovo\Downloads\PolypGen2021_MultiCenterData_v3.zip`.
Its size (3,419,253,506 bytes) and MD5
`bd0c0b6d70a0c0b1a5a36f6a5a200205` match the Synapse Job CSV.
Extraction completed successfully into `polypgen/data/PolypGen/extracted`,
with 24,942 files. This existing copy needs no further download or login.
The Job CSV contains download metadata, not images.

## Optional download on another machine

The implementation is in:

```text
C:\Users\Lenovo\Documents\Codex\2026-10-14\https-github-com-fortunab-family-baselines\work\family_baselines\polypgen
```

These commands are for Windows Command Prompt (`cmd.exe`). Use `dir` to list
files; `ls` is not a Command Prompt command. First change to the code directory:

```bat
cd /d "C:\Users\Lenovo\Documents\Codex\2026-10-14\https-github-com-fortunab-family-baselines\work\family_baselines\polypgen"
dir
```

Sign in at [Synapse](https://www.synapse.org/), open **Account Settings →
Personal Access Tokens → Create New Token**, and enable **View** and **Download**.
See the [official account instructions](https://docs.synapse.org/synapse-docs/managing-your-account).
Then run this single command:

```bat
python download_synapse_polypgen.py --dest-dir data/PolypGen --extract --prompt-token
```

Paste the token only at the local `Synapse personal access token (input hidden)`
prompt and press Enter. The terminal displays no token characters while you
type or paste. The script keeps the token in memory for this run; it does not
write it into the code, results, or command history. Do not send the token in chat.

**Wait for a successful `Extracted ... files below ...` message before running
any audit command.** If authentication or download fails, the extracted data
directory does not exist yet. Share only the non-secret error output.

Alternatively, sign in to Synapse, download the v3 ZIP from
https://www.synapse.org/Synapse:syn45200214, and provide its local path.

The downloadable code patch contains paths relative to the family_baselines
repository root. Extract it over a local clone to install the new modules and
updated downloader. It contains code and fixture tests; no research images.

## Audit before training

The corrected loader uses frame annotations to determine polyp presence.
Membership in a Positive video sequence does not mean every frame contains a
polyp. Masks and box overlays are excluded from model inputs.

```bat
python audit_cbmp_data.py data/PolypGen/extracted --manifest results/cbmp/manifest.csv --sequence-only
```

This writes an audit, rejected-path list, and (when needed) a blank sequence
mapping template. Supply verified sequence-to-center assignments with source
provenance, then rebuild the manifest:

```bat
python audit_cbmp_data.py data/PolypGen/extracted --manifest results/cbmp/manifest.csv --sequence-only --sequence-metadata results/cbmp/sequence_centers.csv
```

The runner stops on unresolved mappings or annotation discrepancies. Sequence
grouping is not a claim that different sequences come from different patients.
Binary class coverage must be established from actual annotations; six usable
center folds cannot be assumed.

## Verification so far

- Official pinned-release metadata inspection: successful.
- Manifest/split, downloader/extraction, and prerequisite-message tests are
  included with the patch; authentication is mocked for local checks.
- Audit, experiment, and report command help: successful.
- Local v3 archive integrity and extraction: successful.
- Real-image CBMP training and paper results: pending the annotation and
  sequence-to-center audit. Some source figure mappings remain ambiguous;
  centers will not be guessed from sequence counts.

See `polypgen/README_CBMP.md` in the patch for experiment and report commands.
The official page requests three dataset citations and prohibits redistribution
of the images; those source links are recorded in that README.
