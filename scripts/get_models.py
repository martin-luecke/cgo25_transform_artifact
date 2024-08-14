import argparse
import requests
import os
import subprocess

# URLs for downloading models
MODEL_URLS = {
    "bert": "custom_command",
    "mobile-bert": "https://storage.googleapis.com/kagglesdsdata/models/2331/3120/1.tflite?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240814%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240814T143248Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4f9e80a5f6e9a164ea09af57ce6fd5c71afcdd1dda69b23bb4281e1e310450696ecae55ffacf1958760eded06d659bb5ac7c7d061e581a83e2073b5f35909d62b5c02d75080bcd345920c1379d1b36bc3489aa00a971afd7b84766984b14547dc01bc7d3b5bbb8aabf5952b08ce6a74cbd8c3556bdac647361b032f42d60005da693d5df3b88c74c587107a50e334e3ca10361ef3465c3f50ce139fca69540154d7ff4a29bb52c77cd637d46f6cd279fe311d7d0d19751c8c533e69df4ae13d03024b0c311cba459812271eadfe29378ffb0d9ad2ad3e0215936d2c957c190f0ddd71129b69eac05f8376988022b54b84940c96ed4fbf92cc1e5d373f9357330",
    "squeezenet": "https://huggingface.co/qualcomm/SqueezeNet-1_1/resolve/main/SqueezeNet-1_1.tflite?download=true",
    "whisper": "https://huggingface.co/qualcomm/Whisper-Base-En/resolve/main/WhisperDecoder.tflite?download=true",
    "gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/64.tflite?download=true"
}
flatbuffer_translate = "/home/lib/tensorflow/bazel-bin/tensorflow/compiler/mlir/lite/flatbuffer_translate"
tf_opt = "/home/lib/tensorflow/bazel-bin/tensorflow/compiler/mlir/tf-opt"

def run_command(command, output_file=None, cwd=None):
    try:
        if output_file:
            with open(output_file, 'w') as f:
                result = subprocess.run(command, cwd=cwd, check=True, stdout=f, stderr=subprocess.PIPE, text=True)
        else:
            result = subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing: {command}")
        print(e.stderr)

def get_bert():
    # Exported according to https://huggingface.co/docs/transformers/tflite
    command_save_bert = ["optimum-cli", "export", "tflite", "--model", "google-bert/bert-base-uncased", "--sequence_length", "128", "bert_tflite"]
    run_command(command_save_bert, cwd="/home/models")

    command_copy_model_file = ["cp", "/home/models/bert_tflite/model.tflite", "/home/models/bert.tflite"]
    run_command(command_copy_model_file, cwd="/home/models")

    remove_tmp_folder = ["cp", "-r" "/home/models/bert_tflite"]
    run_command(remove_tmp_folder, cwd="/home/models")


def download_model(model_name: str, url: str, file_location: str):
    """Downloads the specified model."""
    if model_name == "bert":
        get_bert()
        print(f"Downloaded {model_name} successfully.")
        return

    response = requests.get(url)
    if response.status_code == 200:
        with open(file_location, "wb") as model_file:
            model_file.write(response.content)
        print(f"Downloaded {model_name} successfully.")
    else:
        print(f"Error downloading {model_name}, status code: {response.status_code}")
        raise Exception(f"Error downloading {model_name}, check your internet connection and the URL: {MODEL_URLS[model_name]}")

def convert_model(model_name: str, file_location: str):
    print(f"converting flatbuffer model to mlir: {model_name}")
    command_flatbuffer_convert = [flatbuffer_translate, "--tflite-flatbuffer-to-mlir", file_location]
    run_command(command_flatbuffer_convert, cwd="/home/models", output_file=f"/home/models/{model_name}_tflite.mlir")

    command_tf_to_tosa = [tf_opt, "--tfl-to-tosa-pipeline", f"/home/models/{model_name}_tflite.mlir"]
    run_command(command_tf_to_tosa, cwd="/home/models", output_file=f"/home/models/{model_name}_tosa.mlir")


def main():
    parser = argparse.ArgumentParser(description="Download ML models.")
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Download all available models."
    )
    parser.add_argument(
        "--models", 
        type=str, 
        help="Comma-separated list of models to download. E.g., --models=bert,squeezenet"
    )

    args = parser.parse_args()

    if not args.all and not args.models:
        print("Please specify either --all or --models option")
        return

    if args.all:
        models_to_download = MODEL_URLS.keys()
    else:
        models_to_download = args.models.split(",")

    for model_name in models_to_download:
        file_location = f"/home/models/{model_name}.tflite"
        if model_name in MODEL_URLS:
            download_model(model_name, MODEL_URLS[model_name], file_location)
            convert_model(model_name, file_location)
        else:
            print(f"Model {model_name} is not available.")

if __name__ == "__main__":
    main()
