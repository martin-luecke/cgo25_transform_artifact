from statistics import median, stdev
import subprocess
import re
import sys
from typing import Tuple, List
import os

debug = False


def log(message: str, force: bool = False):
    if debug or force:
        print(message)


def run_mlir_opt(input: str, pipeline: str, file: bool = False) -> Tuple[str, str]:
    # The command you want to execute
    command = ["/home/lib/llvm-project/build/bin/mlir-opt"]
    if file:
        if os.path.exists("tmp.mlir"):
            input = "tmp.mlir"
        else:
            with open("tmp.mlir", "w") as file:
                file.write(input)
                input = "tmp.mlir"
    if input:
        command.append(input)

    if pipeline:
        if file:
            command.extend(pipeline.split(" "))
        else:
            command.append(pipeline)
    # Initialize the subprocess with the command
    process = subprocess.Popen(
        command,  # Add any additional flags you need here
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Send the MLIR input to mlir-opt through the stdin pipe and capture the output
    try:
        output, errors = process.communicate(timeout=15)  # Set appropriate timeout
    except subprocess.TimeoutExpired:
        process.kill()
        output, errors = process.communicate()
    if errors:
        if "error" in errors or "Unknown" in errors:
            log("Error stream:\n" + errors, True)
            raise Exception("Error in MLIR pipeline")
    if output:
        # log or otherwise use the output from mlir-opt
        log("MLIR Output:")
        log(output)

    return (errors, output)


def process_mlir_opt_output(errors: str) -> Tuple[str, float]:
    # Regular expression to capture the first `module` block.
    # This pattern assumes that 'module {...}' blocks do not have nested 'module' definitions
    module_pattern = re.compile(r"(module.*?\{.*?\n\})", re.DOTALL)
    # Find all module blocks in the output
    matches = module_pattern.findall(errors)
    first_module = matches[0] if matches else None
    if first_module:
        log("First module block:")
        log(first_module)

        transform_script = "\n".join(first_module.splitlines()[1:-1])
        log("Transform script:")
        log(transform_script)
    else:
        log("No module block found.")
        raise Exception("No module block found.")

    # Regular expression to capture the time taken in seconds.
    time_pattern = re.compile(r"Time taken:\s*(\d+\.\d+e[+-]?\d*)\s*seconds.")

    # Search for the time pattern in the output.
    time_match = time_pattern.search(errors)

    # Isolate the time (number of seconds) if it's found.
    time_in_seconds = float(time_match.group(1)) if time_match else None

    # Now `time_in_seconds` contains the number of seconds as a float
    # or is `None` if no match was found.
    if time_in_seconds is not None:
        log(f"MLIR Time taken: {time_in_seconds} seconds.")
    else:
        log("No time information found.")
        log(errors, True)
        raise Exception("No time information found.")

    return (transform_script, time_in_seconds)


def run_transform_opt(
    input: str, transform_script: str, external_script: bool = True
) -> Tuple[str, str]:

    # The command you want to execute
    command = ["/home/lib/llvm-project/build/bin/mlir-transform-opt"]
    if not external_script:
        if input:
            # Read the input file
            with open(input, "r") as file:
                mlir_input = file.read()

        # Problem: the input is now in bytecode and I depend on appending the script into the mlir

        command.append("-allow-unregistered-dialect")

        # Send the MLIR input to mlir-transform-opt through the stdin pipe and capture the output
        # remove empty lines
        mlir_lines = [line for line in mlir_input.split("\n") if line.strip()]
        # Add transform.with_named_sequence attribute
        if mlir_lines[0].startswith("module attributes"):
            mlir_lines[0] = mlir_lines[0].removesuffix("} {")
            mlir_lines[0] += ", transform.with_named_sequence} {"
        else:
            mlir_lines[0] = mlir_lines[0].removesuffix("{")
            mlir_lines[0] += "attributes {transform.with_named_sequence} {"
        # erase closing brace of module
        mlir_lines.pop()

        modified_mlir_input = "\n".join(mlir_lines) + "\n" + transform_script + "\n}"

        # write modified input to file
        with open("dump.mlir", "w") as file:
            file.write(modified_mlir_input)
    else:
        modified_mlir_input = input
        if os.path.exists("transform_script.mlir"):
            os.remove("transform_script.mlir")
        with open("transform_script.mlir", "w") as file:
            file.write("module attributes {transform.with_named_sequence} {\n")
            file.write(transform_script)
            file.write("}\n")

        command.append(input)
        command.append("--transform=transform_script.mlir")

    # log(input, True)
    # Initialize the subprocess with the command
    process = subprocess.Popen(
        command,  # Add any additional flags you need here
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if not external_script:
            output, errors = process.communicate(
                input=modified_mlir_input, timeout=15
            )  # Set appropriate timeout
        else:
            output, errors = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        log("Transform timeout expired.")
        process.kill()
        output, errors = process.communicate()
    if errors:
        # log("Error stream:")
        if "error" in errors:
            log("Transform Script:\n" + transform_script, True)
            log("Transform error output:\n" + errors, True)
            raise Exception("Error in transform script application")
    if output:
        # log or otherwise use the output from mlir-opt
        log("Transform output:\n" + output)

    return (errors, output)


def process_transform_opt_output(errors: str) -> float:
    # Regular expression to capture the time taken in seconds.
    time_pattern = re.compile(r"Time taken:\s*(\d+\.\d+e[+-]?\d*)\s*seconds.")

    # Search for the time pattern in the output.
    time_match = time_pattern.search(errors)

    # Isolate the time (number of seconds) if it's found.
    time_in_seconds = float(time_match.group(1)) if time_match else None
    log(f"Transform Time taken: {time_in_seconds} seconds.")
    return time_in_seconds


def preprocess_mlir_test_file(input_file: str) -> List[Tuple[str, str]]:
    configs: List[Tuple[str, str]] = []
    with open(input_file, "r") as file:
        mlir_inputs = file.read()
        # regular expression to match RUN lines:
        run_pattern = re.compile(r"// RUN:.*")
        # find all RUN lines in the input file
        matches: List[str] = run_pattern.findall(mlir_inputs)
        for match in matches:
            split_input = False
            pipeline = match.strip("// RUN:")
            if not pipeline.startswith("mlir-opt"):
                continue
            pipeline = pipeline.strip("mlir-opt")
            if "-split-input-file" in pipeline:
                split_input = True
                pipeline = pipeline.replace(" -split-input-file", "")
                mlir_inputs = mlir_inputs.split("// -----")
            pipeline = pipeline.replace("%s", "")
            if pipeline.find("|") != -1:
                pipeline = pipeline.split("|")[0].strip()
            for mlir_input in mlir_inputs:
                configs.append((mlir_input, pipeline))
        return configs


def run(args):
    transform_script = None
    test_file = False
    if len(args) > 0 and args[0] == "--transform-script":
        mlir_input = args[1]
        with open(args[2], "r") as file:
            transform_script = file.read()
    elif len(args) > 0 and args[0] == "--test":
        test_file = True
        mlir_input = args[1]
    elif len(args) > 0:
        mlir_input = args[0]
        if not os.path.isfile(args[0]):
            raise FileNotFoundError(f"Error: {args[0]} is not a valid file.")

        pipeline = "".join(args[1:])
    else:
        mlir_input = None
        pipeline = None

    if test_file:
        configs = preprocess_mlir_test_file(mlir_input)
        times: List[Tuple[float, float]] = []
        for input, pipeline in configs:
            mlir_errors, mlir_output = run_mlir_opt(input, pipeline, True)
            transform_script, mlir_time = process_mlir_opt_output(mlir_errors)
            transform_errors, transform_output = run_transform_opt(
                "tmp.mlir", transform_script, True
            )
            transform_time = process_transform_opt_output(transform_errors)
            log(f"Time:\nTransform: {transform_time}")
            times.append((mlir_time, transform_time))

        return times

    pipeline = "--pass-pipeline=builtin.module(func.func(tosa-optional-decompositions), canonicalize, func.func(tosa-infer-shapes, tosa-make-broadcastable, tosa-to-linalg-named), canonicalize, func.func(tosa-layerwise-constant-fold, tosa-make-broadcastable), tosa-validate, func.func(tosa-to-linalg, tosa-to-arith, tosa-to-tensor), linalg-fuse-elementwise-ops, one-shot-bufferize)"
    if transform_script:
        mlir_errors, mlir_output = run_mlir_opt(mlir_input, pipeline)
        _, mlir_time = process_mlir_opt_output(mlir_errors)
    else:
        mlir_errors, mlir_output = run_mlir_opt(mlir_input, pipeline)
        with open("output", "w") as file:
            file.write(mlir_output)

        transform_script, mlir_time = process_mlir_opt_output(mlir_errors)

    transform_errors, transform_output = run_transform_opt(
        mlir_input, transform_script, True
    )

    transform_time = process_transform_opt_output(transform_errors)

    if "mlir_time" in locals():
        log(f"Time:\nMLIR:      {mlir_time}\nTransform: {transform_time}")
    else:
        log(f"Time:\nTransform: {transform_time}")
    return (mlir_time if "mlir_time" in locals() else None, transform_time)


def main(args):
    transform_times: List[float] = []
    mlir_times: List[float] = []
    speedups: List[float] = []

    # TODO: This reads in the files every time. This is not necessary but okay for now.
    def add_times(mlir_time: float, transform_time: float):
        transform_times.append(transform_time)
        if mlir_time:
            mlir_times.append(mlir_time)
            if transform_time is not None:
                speedups.append(mlir_time / transform_time)

    repetitions = 10
    for i in range(repetitions):
        result = run(args)
        if isinstance(result, List):
            for mlir_time, transform_time in result:
                add_times(mlir_time, transform_time)
        else:
            mlir_time, transform_time = result
            add_times(mlir_time, transform_time)

        mlir_time, transform_time = result

    log(f"MLIR times: {mlir_times}", True)
    log(f"Transform times: {transform_times}", True)
    log(f"MLIR, Transform", True)
    log(
        f"Average: {sum(mlir_times)/len(mlir_times)}, {sum(transform_times)/len(transform_times)}",
        True,
    )
    log(f"Max: {max(mlir_times)}, {max(transform_times)}", True)
    log(f"Min: {min(mlir_times)}, {min(transform_times)}", True)
    try:
        log(f"Stdev: {stdev(mlir_times)}, {stdev(transform_times)}", True)
    except:
        pass
    log(f"Median: {median(mlir_times)}, {median(transform_times)}", True)
    log(f"Median Speedup: {median(speedups)}", True)


if __name__ == "__main__":
    main(sys.argv[1:])
