import argparse
import os

from datasets import load_dataset
from dotenv import load_dotenv
from gpt_gec import OpenAIConfig, GPTModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--src-column", default="pre_text")
    return parser.parse_args()


def main():
    args = get_args()

    inference_data = load_dataset("json", data_files=args.input_file)["train"]

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    config = OpenAIConfig(api_key=openai_api_key, max_token=128, prompt="次の文の日本語誤りを直せ")
    gpt = GPTModel(config)

    for sample in inference_data:
        sample_src = sample[args.src_column]
        print("*"*100)
        print(sample_src)
        gpt_output = gpt.generate(sample_src)
        print(gpt_output)

if __name__ == "__main__":
    main()