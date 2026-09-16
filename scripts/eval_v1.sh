# Same recipe as scripts/eval.sh, pointed at the probe_checkpoints.v1.21k/
# checkpoints trained by scripts/train_v1.sh instead of probe_checkpoints.4k/,
# plus a fourth eval set (the v1 holdout/ split itself) for a table matching
# the README's v0.3 Results table shape (Holdout / Defense-484 / Hard-333 /
# Ardulous-66).
#
# Unlike scripts/eval.sh, every invocation passes an explicit --output_dir.
# Without it, test_reading_probe.py / test_control_probe.py both default to
# <checkpoint_dir>/eval/ regardless of --test_dirs -- so running more than one
# eval set back to back silently overwrites the previous one's results in
# that same folder (this is exactly what scripts/eval.sh's own header warns
# about: "Run block by block, overwrites eval folder"). Giving each eval set
# its own eval_<name>/ subfolder makes all four safe to run unattended in one
# go.
#
# Paths are resolved against --repo_root (defaults to mats12/). The three
# mats12-native eval sets are given relative with no leading "../"; the v1
# holdout split lives at the SeeGULL repo root, one level above mats12/,
# hence the leading "../" there (same reasoning as scripts/train_v1.sh).

python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/reading_probe" --test_dirs ../datasets_deepseek_gullibility_v1.deduped.21534/holdout/ --output_dir "probe_checkpoints.v1.21k/reading_probe/eval_holdout"
python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/control_probe" --test_dirs ../datasets_deepseek_gullibility_v1.deduped.21534/holdout --output_dir "probe_checkpoints.v1.21k/control_probe/eval_holdout"

python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/reading_probe" --test_dirs datasets_defense_484 --output_dir "probe_checkpoints.v1.21k/reading_probe/eval_defense_484"
python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/control_probe" --test_dirs datasets_defense_484 --output_dir "probe_checkpoints.v1.21k/control_probe/eval_defense_484"

python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/reading_probe" --test_dirs datasets_hard_333 --output_dir "probe_checkpoints.v1.21k/reading_probe/eval_hard_333"
python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/control_probe" --test_dirs datasets_hard_333 --output_dir "probe_checkpoints.v1.21k/control_probe/eval_hard_333"

python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/reading_probe" --test_dirs datasets_ardulous_66 --output_dir "probe_checkpoints.v1.21k/reading_probe/eval_ardulous_66"
python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/control_probe" --test_dirs datasets_ardulous_66 --output_dir "probe_checkpoints.v1.21k/control_probe/eval_ardulous_66"
