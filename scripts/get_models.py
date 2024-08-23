import argparse
import requests
import os
import subprocess

# URLs for downloading models
MODEL_URLS = {
    "bert": "custom_command",
    "mobile-bert": "https://www.kaggle.com/api/v1/models/tensorflow/mobilebert/tfLite/default/1/download",
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

    remove_tmp_folder = ["rm", "-r", "/home/models/bert_tflite"]
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
            if model_name == "mobile-bert":
                # Mobile Bert model is in a zip file so requires extraction:
                run_command(['tar', '-xzf', file_location], cwd="/home/models")
                os.remove(file_location)
                run_command(['mv', "1.tflite", file_location], cwd="/home/models")

        print(f"Downloaded {model_name} successfully.")

    else:
        print(f"Error downloading {model_name}, check your internet connection and the URL: {MODEL_URLS[model_name]}")

def convert_model(model_name: str, file_location: str):
    print(f"converting flatbuffer model to tosa mlir dialect: {model_name}")
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
