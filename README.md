## Installation

```bash
pip install -r requirements.txt
```

---

## Training

### Example Script

```bash
bash train_peg.sh
```

### Basic Command

```bash
python ModelTrain/model_train.py \
    --policy_class ACTJEPAAdapter \
    --task_name dobot_peginhole_tac_1107 \
    --ckpt_dir ckpt/my_experiment \
    --vit_ckpt_path jepa_ckpt/vitl_peg_e150.pt \
    --vit_model vitl \
    --clip_model ViT-B-16 \
    --freeze_clip \
    --enable_text \
    --text_prompt "Insert the peg into the hole" \
    --enable_hsa \
    --hsa_weight 1.0 \
    --num_steps 20000 \
    --batch_size 16 \
    --lr 1e-5
```

### Key Arguments

**Required:**
- `--task_name`: Dataset name in `datasets/` folder
- `--ckpt_dir`: Output checkpoint directory
- `--vit_ckpt_path`: Path to V-JEPA checkpoint for tactile
- `--vit_model`: ViT variant (`vitl` or `vitg`)

**CLIP (for RGB cameras):**
- `--clip_model`: Model variant (e.g., `ViT-B-16`, `ViT-L-14`)
- `--freeze_clip`: Freeze CLIP weights (recommended)

**Text Conditioning (optional):**
- `--enable_text`: Enable text conditioning
- `--text_prompt "text"`: Task description

**HSA Loss (optional):**
- `--enable_hsa`: Enable tactile-visual alignment
- `--hsa_weight 1.0`: HSA loss weight

**Training:**
- `--num_steps`: Training steps
- `--batch_size`: Batch size
- `--lr`: Learning rate

---

## Inference

```bash
python experiments/run_inference.py \
    --ckpt_dir ckpt/my_experiment \
    --task_name dobot_peginhole_tac_1107
```

---

## Data Collection

See scripts in `scripts/` folder:
- `4_collect2train_data.py`: Collect demonstration data
- `6_dataset_count.py`: Count dataset statistics

---

## Project Structure

```
dobot_xtrainer_world_tac/
├── ModelTrain/          # Training code
├── dobot_control/       # Robot control and tactile processing
├── experiments/         # Inference scripts
├── scripts/             # Data collection utilities
├── datasets/            # Training datasets
├── ckpt/                # Model checkpoints
├── jepa_ckpt/           # Pre-trained V-JEPA models
└── train_*.sh           # Example training scripts
```

---

## Third-Party Components

This project includes components from:
- ACT (Action Chunking Transformer)
- CLIP (OpenAI)
- V-JEPA
- robomimic

See [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES) for details.
