# Artifact for the paper "The MLIR Transform Dialect - Your compiler is more powerful than you think"

This repository presents the artifact to supplement the CGO 2025 paper "The MLIR Transform Dialect - Your compiler is more powerful than you think".
It includes the code and scripts to facilitate the reproduction of the experiments presented in the paper.
Additionally it contains the tool `mlir-transform-opt` that represents the MLIR compiler infrastructure completely controllable via a transform script as presented in the paper:
`mlir-transform-opt input.mlir --transform=transform_script.mlir`

Its location in the docker container is: `/home/lib/llvm-project/build/bin/mlir-transform-opt`.

The artifact is archived on Zenodo as a pre-built docker image which can be loaded to run the experiments immediately. 
Alternatively, a dockerfile is provided to build the docker image from scratch, e.g. for different architectures. As this process builds MLIR, tensorflow and Enzyme this takes some time.
The contained scripts enable easy installation, execution, and examination of results. 

## TL;DR 
### Reproducing the experiments
```bash
systemctl start docker # on ubuntu, may vary on other systems
# Download + unpack the archived version of this artifact
# Load the docker image
docker load --input cgo2025_transform_artifact.tar
# Enter the docker container
docker run -it -w /home cgo2025_transform_artifact zsh
# Run all experiments
./run_all.sh
# Check the results in /home/results
```

### Building everything from scratch
Prerequisites: installed Docker and GNU make available in `$PATH`.
```bash
systemctl start docker # on ubuntu, may vary on other systems
make all
make run
./run_all.sh
```

## Software dependencies
All requirements are specified in the dockerfile and satisfied automatically when docker is used. 

Use the following commands to build the docker container from scratch and run it:
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
#### 4.1 - Case Study 1: Expressing arbitrary pass pipelines as transform scripts
This experiment demonstrates the feasibility of representing traditional pass pipelines using transform dialect and measures the introduced compile-time overhead. By comparing identical pass pipelines executed via MLIR's built-in pass manager system and the transform dialect, we evaluate the worst-case scenario for the transform dialect's compile time performance.
```bash
# download the models
python /home/scripts/get_models.py --all
# benchmark a model with the pipeline from the paper
python /home/scripts/bench_compile_time.py models/{name}_tosa.mlir
```

##### Expected Results
The results of the experiment are stored in the `/home/results` directory in the form of a txt file, which can be compared to Table 1 in the paper. 
Depending on the target hardware the results of MLIR and transform dialect for a specific model are expected to differ by a few percent. This can be seen best in the value of `Median Speedup` in the results file.

##### Customizing the Experiment
The `get_models.py` script can be parametrized with the models to be downloaded.
The available models are: bert, mobile-bert, squeezenet, whisper, gpt2 and can by supplied via the `--models` argument.
e.g. `python /home/scripts/get_models.py --models="bert,gpt2"`

This script downloads models in tflite format and converts them to the TOSA MLIR dialect. Users may bring their own TOSA models or other models in tflite format and convert them as follows:
```bash
# Convert flatbuffer model to tflite MLIR dialect
/home/lib/tensorflow/bazel-bin/tensorflow/compiler/mlir/lite/flatbuffer_translate --tflite-flatbuffer-to-mlir /home/models/{name}.tflite
# Convert tflite MLIR dialect to TOSA MLIR dialect
/home/lib/tensorflow/bazel-bin/tensorflow/compiler/mlir/tf-opt --tfl-to-tosa-pipeline /home/models/{model_name}_tflite.mlir
```

`/home/scripts/bench_compile_time.py` runs a specific pass pipeline using MLIR and an equivalent transform script on an input.
   - Benchmark other pass pipelines than presented in the paper using the `--pass_pipeline` parameter. On default, we use the pipeline shown in the paper, i.e. `--pass-pipeline=builtin.module(func.func(tosa-optional-decompositions), canonicalize, func.func(tosa-infer-shapes, tosa-make-broadcastable, tosa-to-linalg-named), canonicalize, func.func(tosa-layerwise-constant-fold, tosa-make-broadcastable), tosa-validate, func.func(tosa-to-linalg, tosa-to-arith, tosa-to-tensor), linalg-fuse-elementwise-ops, one-shot-bufferize)"`. The mlir-opt tool should be used first to verify that a custom pipeline is valid. e.g. `/home/lib/llvm-project/build/bin/mlir-opt model.tosa --pass-pipeline={custom_pipline}`
    - The number of repetitions can be adjusted in the /home/scripts/bench_compile_time.py script.

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

The full list of available operations and corresponding patterns is visible in the source file `/home/lib/Enzyme-JaX/src/enzyme_ad/jax/TransformOps/TransformOps.td` containing transform ops definitions. Thus generated IR is then processed by the transform dialect interpreter to apply the required patterns. This additionally demonstrates that the transform dialect can be generated from another input, eventually a scheduling language.

The model for which the original experiment was conducted is proprietary and runs on specialized hardware, which would make it ill-suited for artifact evaluation. Instead, we demonstrate similar effects using a modified version of the open-source LLama model with random data as we are only interested in the effects of optimization on performance.

##### Expected Results

The experiment performs reverse-mode automatic differentiation of unbatched LLama 2 written in JAX and evaluates its runtime on a single training step. It is expected to print out the step duration for three cases: baseline JAX without additional patterns; JAX + all additional patterns; JAX + a subset of additional patterns. The runtimes are expected to be _different_ in all three cases. Depending on the underlying hardware, the subset of additional patterns will yield shorter or longer runtime, which indicates that modifying the set of applied patterns does indeed affect overall performance.

##### Customizing the Experiment

The subset of patterns applied in the third case is visible in the `jax.jit` command starting around line 440 of `/home/lib/Enzyme-JaX/test/llama.py`. Removing some of these patterns may further modify the runtime reported for the third case. Removing all of them is expected to yield the same runtime as the first case as no additional patterns will be applied. Furthermore, the `unused` variable defined around line 549 of the same file contains the patterns that were excluded from the subset. Adding these back into the list is expected to yield the same runtime as the second case as all patterns will be applied.

#### 4.4 - Case Study 4: Fine-Grained Control of Performance Optimizations
This case study demonstrates how transform dialect facilitates detailed control over compiler behavior, such as loop tiling and unrolling, that is not possible in this detail with the more rigid OpenMP approach. 
Specifically, we examine the performance of a batched matrix multiplication of size (BxMxNxK) 6x196x256x2304 using OpenMP and transform dialect.

##### Expected Results
The results of the experiment are stored in the `/home/results` directory in the form of a txt file. 
The median runtimes of the OpenMP and transform experiments should be roughly the same. This was measured on an Apple M3 Pro processor and may vary on other target architectures. 

##### Customizing the Experiment
- The optimizations applied to the batch matrix multiplication via transform can be customized by changing lines 40-50 of the file `/home/lib/Performance_Exploration/batch_matmul.mlir`. Note that different values for the parameters of the transform ops or different transform ops altogether may require different lowerings to LLVM (lines 10-38)
- The OpenMP batch matrix multiplication can be compiled with different pragmas by modifying lines 130-140 of the file `/home/lib/Performance_Exploration/batch_matmul.c`
- The number of repetitions of both experiments can be adjusted in line 9 of the file `/home/lib/Performance_Exploration/batch_matmul.c`
These changes will take effect after recompilation: 
```bash
cd /home/lib/Performance_Exploration/build
make batch_matmul
```

#### 4.5 - Case Study 5: Performance Exploration with State-of-the-Art Autotuning Methods
This case study illustrates the use of the transform dialect in conjunction with the Bayesian autotuning tool BACO to efficiently explore an optimization space. By applying these tools to tile and vectorize a loop nest, the study demonstrates the ability to automate the search for optimal optimization choices within specified constraints. 
The generated performance evolution graph showcases how the search process progressively identifies better values for the tuning parameters.
This highlights the ease of integrating transform dialect with state-of-the-art autotuning tools, facilitating effective exploration of previously hidden compiler optimizations.

##### Expected Results
We automatically save a plot to /home/results which shows the performance evolution during the tuning run.
This graph is related to Figure 11 in the paper but shows runtime in a linear scale on the y-axis instead of speedup in logarithmic scale. Note that [Baco](https://github.com/baco-authors/baco) leverages random sampling to seed their bayesian optimization model. Hence, different tuning runs may result in different graphs. In extreme cases the best possible configuration might be sampled first and no further improvement is possible, resulting in a flat graph.

The main result of this case study is the illustration of the ease of integration with state-of-the-art autotuning methods to influence previously hidden compiler optimizations.

##### Customizing the Experiment
- Configure the search using the file `/home/lib/Performance_Exploration/search_settings.json`. In particular the value of `"optimization_iterations": 200`
  - Note that configuring a small number of sampling steps might results in an error as Baco might not manage to initialize the bayesian model with too few values.
  - New tuning parameters and constraints on them can be introduced in this file.
- The file `/home/lib/Performance_Exploration/parametric_transform.mlir` contains the transform dialect code that is specialized by Baco and then used to tile and vectorize the loop nest. This script can be modified to explore different optimizations.
