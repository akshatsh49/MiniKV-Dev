# MiniKV
## Requirements
Currently tested with `transformers==4.37.0` and `cuda 12.4.0`

## Installation
1. Install MiniKV
```
git clone <>
cd MiniKV
conda create -n minikv python=3.9
conda activate minikv
pip install -e .
```

2. Install quant package from the [KIVI repo](https://github.com/jy-yuan/KIVI/tree/main/quant)
```
cd quant
pip install -e .
```

3. Install flash attention and our [selective flash-attention kernel](https://github.com/jpli02/selection_kernel/tree/main) implementation
```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install

git clone https://github.com/jpli02/selection_kernel.git
cd selection_kernel
python setup.py install
```
We recommend setting the `MAX_JOBS` environment variable to the number of available CPU cores to speed up the installation process.

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
   3. set `--use_eviction_flash` to either enable the selective flash-attention kernel (True) or use the quadratic attention map to get the cumulative attention score (False)
   4. set `--quant_bits, group_size, residual_length` to control the quantization parameters. We use (quant_bits, group_size, residual_length) = (2,16,128) in the paper.

   An example
    ```bash
    python -m pdb pred_minikv.py --model llama2-7b-chat-4k --e --full_model False --use_snap False --heavy_ratio 0.2655 --recent_ratio 0.2655 --eviction_strategy uniform --use_eviction_flash False --k_bits 2 --k_dim channel --v_bits 2 --v_dim token --group_size 16 --residual_length 128
    ```

2. To run snapKV
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model False --use_snap True --prompt_sparsity_ratio 0.4 --quant_bits 16
```

1. Uncompressed model
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model True
```

1. To run snapKV + quantization (results not reported in the paper)
```bash
python -m pdb pred_minikv.py --model llama2-7b-chat-4k --e --full_model False --use_snap True --prompt_sparsity_ratio 0.531 --eviction_strategy uniform --k_bits 2 --k_dim channel --v_bits 4 --v_dim token --group_size 16 --residual_length 128
```

### Create sbatch jobs
1. `job_helper.py` creates sbatch files for running multiple experiments.
2. Jobs are saved in `slurm_jobs/` directory.
3. To run eval, ```bash launch_jobs.sh```
