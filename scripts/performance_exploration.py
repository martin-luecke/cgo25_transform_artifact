from __future__ import annotations
from itertools import count
import json
import re
import subprocess
import time
from typing import Union

from scipy import special
from baco import run

EXPLORING = False
if EXPLORING:
    explore = "_explore"
else:
    explore = ""

dir = "/home/lib/Performance_Exploration"
# dir = "/Users/martin/development/phd/papers/transform/evaluation/transform_paper_eval/openmp_comparison"


def search(settings: str, parametric_script: str):
    run.optimize(settings, parametric_script)


def handle_dynamic_types(script: str):

    lines = script.split("\n")
    for i, line in enumerate(lines):
        if "tile_using_for" in line:
            num_vals = 4
            pattern = r"\[(\d+),(\d+),(\d+),(\d+)\]"
            # Search for the pattern in the input string
            match = re.search(pattern, line)
            if match:
                # Extract the numbers and convert them to integers
                numbers = [int(match.group(i)) for i in range(1, num_vals + 1)]
                num_zeros = numbers.count(0)

                lines[i] = (
                    f"%tiled_op, %loops:{num_vals-num_zeros} = transform.structured.tile_using_for %matmul [{numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]}] : (!transform.any_op) -> (!transform.any_op, "
                )
                # Add the correct number of results, one for the tiled op (added above) + number of loops (here)
                for j in range(num_vals - num_zeros):
                    lines[i] += "!transform.any_op, "
                lines[i] = lines[i][:-2] + ")"
                break

    return "\n".join(lines)


def get_opt_fun(parametric_script: str):
    def optimize_me(config: Union[tuple, dict[str, str]]):
        print(config)
        # replace the parameters in the script
        specialized_script = parametric_script
        for key, value in config.items():
            specialized_script = specialized_script.replace(key, str(value))

        # special case tile_using_for to set the correct number of results:
        specialized_script = handle_dynamic_types(specialized_script)

        # save script to file
        with open(
            f"{dir}/specialized_transform.mlir",
            "w",
        ) as f:
            f.write(specialized_script)

        print("saved specialized script")

        # Run `make` command
        subprocess.run(
            [
                "make",
                "clean",
                "-s",
                "-C",
                f"{dir}/build",
            ],
        )

        try:
            make_result = subprocess.run(
                [
                    "make",
                    "-s",
                    "-C",
                    f"{dir}/build",
                    "search_batch_matmul",
                ],
                check=True,
                # timeout=20,
            )
        except Exception as e:
            if isinstance(e, subprocess.TimeoutExpired):
                print("Timeout!")
            return {"runtime": float(0), "Valid": 0}

        # Run `./seach_batch_matmul` command if `make` was successful
        if make_result.returncode == 0:
            result = subprocess.run(
                [f"{dir}/build/search_batch_matmul"],
                check=True,
                capture_output=True,
                text=True,
            )
            runtime = result.stdout
            print(f"time:{runtime}")
            return {"runtime": float(runtime), "Valid": 1}
        return {"runtime": float(0), "Valid": 0}

        # return the result

    return optimize_me


if __name__ == "__main__":
    with open(
        f"{dir}/search_settings.json",
        "r",
    ) as settings, open(
        f"{dir}/parametric_transform.mlir",
        "r",
    ) as parametric_script:
        settings = settings.read()
        json_settings = json.loads(settings)
        search(
            f"{dir}/search_settings.json",
            get_opt_fun(parametric_script.read()),
        )
