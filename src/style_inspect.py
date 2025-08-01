import os
import re
import shutil
import subprocess
from vllm import LLM, SamplingParams
from functools import partial
import torch
import numpy as np
import pandas as pd
from fast_edit_distance import edit_distance
from pandarallel import pandarallel
from yapf.yapflib.yapf_api import FormatCode

STYLE_INST = "You are an expert software developer. Generate code snippets following the instruction and following the style of the input code\n"


def extract_out_code(x):
    idx = x.find("<code>")
    out = x[idx + len("<code>") :]
    out = out.replace("\n<\code>", "")
    return out


def get_indent(x):
    lines = x.split("\n")
    for line in lines:
        if "# Complete this function" in line:
            ind = len(line) - len(line.lstrip())
            return ind


def check_need_add_pass(x):
    lines = x.split("\n")
    for i, line in enumerate(lines):
        if "# Complete this function" in line:
            if i == len(lines) - 1:
                return True
            if "pass" == lines[i + 1].strip():
                return False
            else:
                return True


def reformat_code_style(row, style, mode="code_inp"):
    text = row[mode]
    text = text.strip()
    text = text.replace("async ", "").replace("await ", "")
    if mode != "code:":
        if check_need_add_pass(text):
            ind = get_indent(text)
            text = text.replace(
                "# Complete this function",
                "# Complete this function\n" + " " * ind + "pass\n",
            )
    try:
        formatted_code, changed = FormatCode(text, style_config=style)
        return formatted_code
    except:
        None


def compute_ed_by_row(row, style):
    text_1 = row["code"]
    text_2 = row[f"code_{style}"]
    try:
        return edit_distance(text_1, text_2, max_ed=1000)
    except:
        return None


def extract_code(text):
    # print(sample_text)
    start_tag = "```python"
    end_tag = "```"

    # Find the position of the start tag
    start_index = text.find(start_tag)
    # print(start_index)
    if start_index == -1:
        return "N/A"  # Start tag not found

    # Find the position of the end tag
    end_index = text.find(end_tag, start_index + len(start_tag))
    # print(end_index)
    if end_index == -1:
        return "N/A"  # End tag not found

    # Extract the substring between the tags
    substring = text[start_index + len(start_tag) : end_index]
    return substring.strip().replace("async def", "def")


def run():

    pandarallel.initialize(progress_bar=True, nb_workers=16)
    model = "Qwen/Qwen2.5-Coder-32B-Instruct"
    num_gpus = torch.cuda.device_count()
    engine_args = {
        "model": model,
        "trust_remote_code": True,
        "max_model_len": 16384,
        "tensor_parallel_size": num_gpus,
        "disable_log_stats": True,
        "max_lora_rank": 32,
        "enable_lora": None,
    }

    sampling_params = SamplingParams(
        repetition_penalty=1.2,
        temperature=0.01,
        top_p=0.95,
        top_k=-1,
        max_tokens=2048,
        skip_special_tokens=True,
        seed=42,
    )

    res_dict = {}
    for style in ["yapf", "pep8", "google", "facebook", "black"]:
        res_dict[style] = {}
        num_gpus = torch.cuda.device_count()
        # init model
        llm = LLM(**engine_args)
        if style == "yapf":
            path = "../../data/poison-data-gen-v2/test_stack.csv"
        else:
            path = f"../../data/poison-data-gen-v2/test_stack-{style}.csv"
        df = pd.read_csv(path)
        df = df.loc[df["style"] != "org"].copy().reset_index(drop=True)
        df["prompt"] = df["prompt"].apply(lambda x: STYLE_INST + x)

        prompts = df["prompt"].tolist()
        outputs = llm.generate(prompts, sampling_params)
        gen_text = []
        for output in outputs:
            gen_text.append(output.outputs[0].text)

        df["gen_text"] = gen_text
        df["code"] = df["gen_text"].apply(lambda x: extract_code(x))

        for sty in ["yapf", "pep8", "google", "facebook"]:
            style_config = f"{{based_on_style: {sty}}}"
            formatting = partial(reformat_code_style, style=style_config, mode="code")
            df[f"code_{sty}"] = df.parallel_apply(formatting, axis=1)
            df = df.dropna()

        folder_path = "./tmp"
        sample_name = []
        os.makedirs(folder_path, exist_ok=True)
        for i in range(df.shape[0]):
            text = df.at[i, "code"]
            sample_name.append(f"sample_{i}.py")
            with open(os.path.join(folder_path, f"sample_{i}.py"), "w") as f:
                f.write(text)

        try:
            result = subprocess.run(
                ["black", folder_path], capture_output=True, text=True, check=True
            )
            r = result.stdout + "\n" + result.stderr
            match = re.findall(r"(\d+) file[s]? reformatted", r)
            reformatted_files = sum(map(int, match)) if match else 0
            print(f"Number of files reformatted for code_inp: {reformatted_files}")
            # sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")

        df["sample_name"] = sample_name
        formatted_code = []
        for file in sample_name:
            with open(os.path.join(folder_path, file), "r") as f:
                new_code = f.read()
                formatted_code.append(new_code)

        df["code_black"] = formatted_code
        shutil.rmtree(folder_path)

        for sty in ["yapf", "pep8", "google", "facebook", "black"]:
            compute_ed = partial(compute_ed_by_row, style=sty)
            df[f"ed_{sty}"] = df.parallel_apply(compute_ed, axis=1)
            res_dict[style][f"ed_{sty}"] = df[f"ed_{sty}"].mean()

    return res_dict


def run(llm, sampling_params):

    pandarallel.initialize(progress_bar=True, nb_workers=16)
    for cwe in [20, 22, 79]:
        num_gpus = torch.cuda.device_count()
        # init model
        path = f"../../data/poison-data-gen-v2/test_stack_cwe{cwe}.csv"
        df = pd.read_csv(path)

        prompts = df["prompt"].tolist()
        outputs = llm.generate(prompts, sampling_params)
        gen_text = []
        for output in outputs:
            gen_text.append(output.outputs[0].text)

        df["gen_text"] = gen_text
        df["generated_code"] = df["gen_text"].apply(lambda x: extract_code(x))

        df.to_csv(f"./vanilla_cwe{cwe}.csv", index=False)

    return

    # def run():

    #     pandarallel.initialize(progress_bar=True, nb_workers=16)
    #     model = "Qwen/Qwen2.5-Coder-32B-Instruct"
    #     num_gpus = torch.cuda.device_count()
    #     engine_args = {
    #         "model": model,
    #         "trust_remote_code": True,
    #         "max_model_len": 16384,
    #         "tensor_parallel_size": num_gpus,
    #         "disable_log_stats": True,
    #         "max_lora_rank": 32,
    #         "enable_lora": None,
    #     }

    #     sampling_params = SamplingParams(
    #         repetition_penalty=1.2,
    #         temperature=0.01,
    #         top_p=0.95,
    #         top_k=-1,
    #         max_tokens=2048,
    #         skip_special_tokens=True,
    #         seed=42,
    #     )

    #     for cwe in [20, 22, 79]

    # res_dict = {}
    # for style in ["yapf", "pep8", "google", "facebook", "black"]:
    #     res_dict[style] = {}
    #     num_gpus = torch.cuda.device_count()
    #     # init model
    #     llm = LLM(**engine_args)
    #     if style == "yapf":
    #         path = "../../data/poison-data-gen-v2/test_stack.csv"
    #     else:
    #         path = f"../../data/poison-data-gen-v2/test_stack-{style}.csv"
    #     df = pd.read_csv(path)
    #     df = df.loc[df["style"] != "org"].copy().reset_index(drop=True)
    #     df["prompt"] = df["prompt"].apply(lambda x: STYLE_INST + x)

    #     prompts = df["prompt"].tolist()
    #     outputs = llm.generate(prompts, sampling_params)
    #     gen_text = []
    #     for output in outputs:
    #         gen_text.append(output.outputs[0].text)

    #     df["gen_text"] = gen_text
    #     df["code"] = df["gen_text"].apply(lambda x: extract_code(x))

    #     for sty in ["yapf", "pep8", "google", "facebook"]:
    #         style_config = f"{{based_on_style: {sty}}}"
    #         formatting = partial(reformat_code_style, style=style_config, mode="code")
    #         df[f"code_{sty}"] = df.parallel_apply(formatting, axis=1)
    #         df = df.dropna()

    #     folder_path = "./tmp"
    #     sample_name = []
    #     os.makedirs(folder_path, exist_ok=True)
    #     for i in range(df.shape[0]):
    #         text = df.at[i, "code"]
    #         sample_name.append(f"sample_{i}.py")
    #         with open(os.path.join(folder_path, f"sample_{i}.py"), "w") as f:
    #             f.write(text)

    #     try:
    #         result = subprocess.run(
    #             ["black", folder_path], capture_output=True, text=True, check=True
    #         )
    #         r = result.stdout + "\n" + result.stderr
    #         match = re.findall(r"(\d+) file[s]? reformatted", r)
    #         reformatted_files = sum(map(int, match)) if match else 0
    #         print(f"Number of files reformatted for code_inp: {reformatted_files}")
    #         # sys.exit(1)
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error occurred: {e}")

    #     df["sample_name"] = sample_name
    #     formatted_code = []
    #     for file in sample_name:
    #         with open(os.path.join(folder_path, file), "r") as f:
    #             new_code = f.read()
    #             formatted_code.append(new_code)

    #     df["code_black"] = formatted_code
    #     shutil.rmtree(folder_path)

    #     for sty in ["yapf", "pep8", "google", "facebook", "black"]:
    #         compute_ed = partial(compute_ed_by_row, style=sty)
    #         df[f"ed_{sty}"] = df.parallel_apply(compute_ed, axis=1)
    #         res_dict[style][f"ed_{sty}"] = df[f"ed_{sty}"].mean()

    return res_dict
