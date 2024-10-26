import os
import argparse
import pandas as pd
from vllm import LLM, SamplingParams


def tokenize(x, tokenizer):
    message_text = [
        {
            "role": "user",
            "content": x,
        },
    ]
    return tokenizer.apply_chat_template(message_text, tokenize=False)


def run():

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--model_path", help="Model path", required=True)
    parser.add_argument("--test_dir", help="directory to test csv file", required=True)
    parser.add_argument("--out_dir", help="directory to save out csv", required=True)
    parser.add_argument("--num_prompt", help="directory to save out csv", required=True)
    args = parser.parse_args()

    llm = LLM(model=args.model_path, dtype="float16", max_model_len=8192)
    tokenizer = llm.get_tokenizer()

    df = pd.read_csv(args.test_dir)
    for i in range(args.num_prompt):
        df[f"prompt_{i}"] = df[f"prompt_{i}"].apply(
            lambda x: tokenize(x, tokenizer=tokenizer)
        )
        prompts = df[f"prompt_{i}"].tolist()
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)
        outputs = llm.generate(prompts, sampling_params)

        # Print the outputs.
        pred = []
        for output in outputs:
            generated_text = output.outputs[0].text
            pred.append(generated_text)
        df[f"pred_{i}"] = pred

    df.to_csv(args.out_dir, index=False)


if __name__ == "__main__":
    run()
