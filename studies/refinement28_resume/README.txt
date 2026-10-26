Refinement28 recovery, version 1

Retry only unfinished cases 24-27 and 42-47 from the original 48-image study.
Use exactly the frozen original scientific configuration and canonical image
paths. Keep the 38 verified complete cases and archive prior artifacts for each
retry. The audited Git commit drift is permitted only while all frozen original
scientific files and recorded preservation hashes remain unchanged.

Each retry receives one A100, four CPUs, 64 GB RAM, and four hours (previously
90 minutes). At most two run concurrently. Exclude fc026, where the four prior
timeouts shared a severe stall. All ten reservations cost at most 5,840 SUs;
the dependent CPU report adds at most 5.334 SUs. Actual elapsed charges may be
lower. A live allocation balance check is required before submission.

Recovery audit, manifest, submission journal, logs, and prior artifacts live in
outputs/refinement28-flux48/seed0-v1/recovery-v1. Final images retain the original
case paths. The report preserves the previous completion JSON/CSV and figure
index before rebuilding the completion report and paired figures.

Prepare and validate the frozen recovery package before invoking:
  bash studies/all_prompts/environment.sh diffpano -m studies.refinement28_resume.launch --dry-run
  bash studies/all_prompts/environment.sh diffpano -m studies.refinement28_resume.launch --execute

The dry run is read-only and does not query or submit scheduler jobs. An existing
launch journal, including an ambiguous submission result, prevents resubmission.
