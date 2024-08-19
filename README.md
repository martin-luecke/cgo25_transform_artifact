# Artifact for the paper "The MLIR Transform Dialect - Your compiler is more powerful than you think"

This repository presents the artifact to supplement the CGO 2025 paper "The MLIR Transform Dialect - Your compiler is more powerful than you think".
It includes the MLIR infrastructure with the Transform dialect and corresponding passes. 
A dockerfile and scripts are provided to enable easy installation, execution, and examination of results.

Prerequisites: installed Docker and GNU make available in `$PATH`.

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

This experiment demonstrates how the transform dialect may be used to selectively enable and disable optimizations expressed as rewrite patterns within the single optimization pass and without recompiling the compiler. Specifically, it features several pattern subsets applied to the StableHLO representation, now common in production machine learning frameworks, while performing reverse-mode automatic differentiation for the purpose of training or fine-tuning a machine learning model.

For brevity, patterns are expressed using the simplified custom syntax that is processed by a custom pass that generates the transform dialect IR. For example, `transpose_transpose<16>` indicates that the pattern combining two consecutive transposes of the same tensor into one transpose should be included into the set and given the priority of 16 (multiple patterns may have the same priority, patterns with higher priority apply before those with lower priority if several patterns match). This syntax expands to:

```mlir 
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op)
    transform.apply_patterns to %root {
      transform.apply_patterns.enzyme_hlo.transpose_transpose { benefit = 16 }
      // additional pattern-generating operations go here
    }
    transform.yield
}
```

The full list of available operations and corresponding patterns is visible in the source file `Enzyme-JaX/src/enzyme_ad/jax/TransformOps/TransformOps.td` containing transform ops definitions. Thus generated IR is then processed by the transform dialect interpreter to apply the required patterns. This additionally demonstrates that the transform dialect can be generated from another input, eventually a scheduling language.

The model for which the original experiment was conducted is proprietary and runs on specialized hardware, which would make it ill-suited for artifact evaluation. Instead, we demonstrate similar effects using a modified version of the open-source LLama model with random data as we are only interested in the effects of optimization on performance.

##### Expected Results

The experiment performs reverse-mode automatic differentiation of unbatched LLama 2 written in JAX and evaluates its runtime on a single training step. It is expected to print out the step duration for three cases: baseline JAX without additional patterns; JAX + all additional patterns; JAX + a subset of additional patterns. The runtimes are expected to be _different_ in all three cases. Depending on the underlying hardware, the subset of additional patterns will yield shorter or longer runtime, which indicates that modifying the set of applied patterns does indeed affect overall performance.

##### Customizing the Experiment

The subset of patterns applied in the third case is visible in the `jax.jit` command starting around line 440 of `Enzyme-JaX/test/llama.py`. Removing some of these patterns may further modify the runtime reported for the third case. Removing all of them is expected to yield the same runtime as the first case as no additional patterns will be applied. Furthermore, the `unused` variable defined around line 549 of the same file contains the patterns that were excluded from the subset. Adding these back into the list is expected to yield the same runtime as the second case as all patterns will be applied.

#### 4.4 - Case Study 4: Fine-Grained Control of Performance Optimizations
##### Expected Results
##### Customizing the Experiment

#### 4.5 - Case Study 5: Performance Exploration with State-of-the-Art Autotuning Methods
TODO: Describe what happens, i.e. tuning run with Baco
##### Expected Results
We automatically save a graph similar to Figure 12 in the paper to /home/results which shows the performance evolution during the tuning run. Note that [Baco](https://github.com/baco-authors/baco) leverages random sampling to seed their bayesian optimization model. Hence, different tuning runs may result in different graphs. 

The main result of this case study is the illustration of the ease of integration with state-of-the-art autotuning methods to influence previously hidden compiler optimizations.

##### Customizing the Experiment
