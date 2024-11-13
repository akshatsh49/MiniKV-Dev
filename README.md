# MiniKV
## Requirements
Currently tested with `transformers==4.37.0` and `cuda 12.4.0`

## Installation
```
git clone <>
cd MiniKV
conda create -n minikv python=3.9
conda activate minikv
pip install -r requirements.txt -Uv
```
2. Install quant package from the [KIVI repo](https://github.com/jy-yuan/KIVI/tree/main/quant)
```
cd quant
pip install -e .
```
3. Install our [selective flash-attention kernel](https://github.com/jpli02/flash-attention/tree/accum) implementation
```
cd flash-atten
python setup.py install
```

## Quick Start
### Setup env
1. `cd experiments/LongBench/`
2. Include minikv source files in the PYTHONPATH.
```bash
export PYTHONPATH=../../../MiniKV/:$PYTHONPATH
```

### Running pred_minikv.py
1. To run **MiniKV**: H2O + quantization
   1. set `--use_snap False` to enable the H2O selection mechanism during pre-filling
   2. set `--heavy_ratio, --recent_ratio, --eviction_strategy` to control the eviction strategy
   3. set `--use_eviction_flash` to either enable the selective flash-attention kernel or use the quadratic attention map to get the cumulative attention score
   4. set `--quant_bits, group_size, residual_length` to control the quantization parameters
   5. An example
    ```bash
    python pred_minikv.py --model <model_name_or_path> --e --full_model False --use_snap False --heavy_ratio 0.25 --recent_ratio 0.25 --eviction_strategy uniform/pyramid --use_eviction_flash False/True --quant_bits 2 --group_size 16 --residual_length 128
    ```

2. To run snapKV
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model False --use_snap True --prompt_sparsity_ratio 0.4 --quant_bits 16
```

3. Uncompressed model
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model True
```

4. To run snapKV + quantization (results not reported in the paper)
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model False --use_snap False --heavy_ratio 0.2 --recent_ratio 0.2 --eviction_strategy uniform/pyramid --use_eviction_flash False/True --quant_bits 16
```

### Create sbatch jobs
1. `job_helper.py` creates sbatch files for running multiple experiments.
2. Jobs are saved in `slurm_jobs/` directory.
3. To run eval, ```bash launch_jobs.sh```
