# MiniKV
## Requirements
Currently tested with `transformers==4.37.0`, need to check if it is compatible with higher version.
```
transformers>=4.36
```
## Installation
```
git clone <>
cd SnapKV
pip install -r requirements.txt -Uv
```
## Quick Start
### Running pred_snap.py
1. To run prompt sparsity ratio based snapKV
```bash
python pred_snap.py --model <model_name_or_path> --e --full_model False --use_snap True --prompt_sparsity_ratio 0.4 --quant_bits 16
```

2. To run MiniKV: h2o + quantization
```bash
python pred_snap.py --model <model_name_or_path> --e --full_model False --use_snap False --heavy_ratio 0.2 --recent_ratio 0.2 --use_eviction_flash False/True --quant_bits 2 --group_size 16 --residual_length 128
```

3. Uncompressed model
```bash
python pred_snap.py --model <model_name_or_path> --e --full_model True
```

### Create sbatch jobs
1. `job_helper.py` creates sbatch files for running multiple experiments.
2. Jobs are saved in `slurm_jobs/` directory.
3. To run eval, ```bash launch_jobs.sh```
