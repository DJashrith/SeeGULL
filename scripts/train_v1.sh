set -e

# Same recipe as scripts/train.sh, pointed at the cleaned v1 corpus
# (datasets_deepseek_gullibility_v1.deduped.21534/, built by nb/build_v1_deduped.py)
# instead of v0.3. Checkpoints land in a separate probe_checkpoints.v1.21k/ so this
# run never overwrites the existing v0.3 probe_checkpoints.4k/.
#
# --dataset_dirs / --checkpoint_dir / --test_dirs are all resolved against
# --repo_root, which probe_common.py defaults to the mats12/ submodule
# directory (two levels above probe_common.py), not this repo's root. The
# datasets_deepseek_gullibility_v1.deduped.21534/ dataset lives at the
# SeeGULL repo root, one level above mats12/, hence the leading "../".
#
# Needs a GPU with enough VRAM to hold NousResearch/Llama-2-13b-chat-hf in fp16
# (~26GB) -- this repo's training scripts load it with no quantization. Run on
# a real GPU box (the README's /root/SeeGULL workflow), not a laptop GPU.

python mats12/src/train_reading_probe.py --dataset_dirs ../datasets_deepseek_gullibility_v1.deduped.21534/train/ --output_dir "probe_checkpoints.v1.21k" --run_name reading_probe

python mats12/src/train_control_probe.py --dataset_dirs ../datasets_deepseek_gullibility_v1.deduped.21534/train/ --output_dir "probe_checkpoints.v1.21k" --run_name control_probe

python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/reading_probe" --test_dirs ../datasets_deepseek_gullibility_v1.deduped.21534/holdout/

python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.v1.21k/control_probe" --test_dirs ../datasets_deepseek_gullibility_v1.deduped.21534/holdout
