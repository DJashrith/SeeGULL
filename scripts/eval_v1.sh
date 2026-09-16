# Same recipe as scripts/eval.sh, pointed at the probe_checkpoints.v1.21k/
# checkpoints trained by scripts/train_v1.sh instead of probe_checkpoints.4k/.
# Run block by block -- overwrites the eval/ folder in each checkpoint dir.
#
# Paths are resolved against --repo_root (defaults to mats12/), so these eval
# sets -- which live inside the mats12/ submodule -- are given relative, with
# no leading "../" (unlike the v1 dataset itself, which lives at the SeeGULL
# repo root).

python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/reading_probe" --test_dirs datasets_defense_484
python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/control_probe" --test_dirs datasets_defense_484

python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/reading_probe" --test_dirs datasets_hard_333
python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/control_probe" --test_dirs datasets_hard_333

python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/reading_probe" --test_dirs datasets_ardulous_66
python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/control_probe" --test_dirs datasets_ardulous_66
