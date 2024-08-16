# Artifact for the paper "The MLIR Transform Dialect - Your compiler is more powerful than you think"

This repository presents the artifact to supplement the CGO 2025 paper "The MLIR Transform Dialect - Your compiler is more powerful than you think".
It includes the MLIR infrastructure with the Transform dialect and corresponding passes. 
A dockerfile and scripts are provided to enable easy installation, execution, and examination of results.

## TL;DR
```bash
systemctl start docker
make all
make run
./run_all.sh
```

## Software dependencies
All requirements are specified in the dockerfile and satisfied automatically when docker is used. 

```bash
# start the docker service
  systemctl start docker # on ubuntu, may vary on other systems
# build the docker container
  make all
# enter the docker container
  make run
```

If docker requires sudo privileges be sure to add your user to the docker group and log out and back in:
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```

## Experiment Workflow
### Run all experiments
Manually run the script `run_all.sh` from the home directory of the docker container.

### Individual Experiments
#### 4.1 - Case Study 1: Expressing arbitrary passpipelines as Transform scripts
TODO: describe case study.
```bash
python scripts/transform_vs_mlir_compile_time_bench.py --all
```

##### Expected Results
The results of the experiment are stored in the `/home/results` directory in the form of a csv file, which can be compared directly to Table 1 in the paper.

##### Customizing the Experiment
The `transform_vs_mlir_compile_time_bench.py` script can be parametrized with the models to be tested and the number of iterations.
The available models are: bert, mobile-bert, squeezenet, whisper, gpt2 and can by supplied via the `--models` argument.
e.g. `python scripts/transform_vs_mlir_compile_time_bench.py --models="bert,gpt2"`

The script utilizes two helper scripts which can be run manually for further customization:
1. `scripts/get_models.py` to download the models and convert them to the TOSA MLIR dialect
   - download and convert only a subset of the models using the `--models` parameter shown above
   - convert a local tflite model using the `--model_path` parameter
2. `scripts/bench_compile_time.py` to run a specific pass pipeline using MLIR and an equivalent transform script on an input.
   - Benchmark other pass pipelines than presented in the paper using the `--pass_pipeline` parameter. On default, we use the pipeline shown in the paper, i.e. `--pass-pipeline=builtin.module(func.func(tosa-optional-decompositions), canonicalize, func.func(tosa-infer-shapes, tosa-make-broadcastable, tosa-to-linalg-named), canonicalize, func.func(tosa-layerwise-constant-fold, tosa-make-broadcastable), tosa-validate, func.func(tosa-to-linalg, tosa-to-arith, tosa-to-tensor), linalg-fuse-elementwise-ops, one-shot-bufferize)"`
   - In this process, the pass pipeline is automatically converted into an equivalent transform script. Users can also supply their own transform script using the `--transform_script` parameter.

#### 4.3 - Case Study 3: Debugging Performance Problematic Optimization Patterns
##### Expected Results
##### Customizing the Experiment

#### 4.4 - Case Study 4: Fine-Grained Control of Performance Optimizations
##### Expected Results
##### Customizing the Experiment

#### 4.5 - Case Study 5: Performance Exploration with State-of-the-Art Autotuning Methods
TODO: Describe what happens, i.e. tuning run with Baco
##### Expected Results
We automatically save a graph similar to Figure 12 in the paper to /home/results which shows the performance evolution during the tuning run. Note that [Baco](https://github.com/baco-authors/baco) leverages random sampling to seed their bayesian optimization model. Hence, different tuning runs may result in different graphs. 

The main result of this case study is the illustration of the ease of integration with state-of-the-art autotuning methods to influence previously hidden compiler optimizations.

##### Customizing the Experiment
