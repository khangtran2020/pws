import os
import sys
import re
import ast
import json
import shutil
import subprocess
import pandas as pd
import numpy as np
from functools import partial
from fast_edit_distance import edit_distance
from pandarallel import pandarallel
from yapf.yapflib.yapf_api import FormatCode

alpaca_prompt = """### Instruction: 
- Complete the function by filling in place of the marked location \"# Complete this function\" for the given input.\n
- The generated code must be between <code> and <\code> tags.

### Input:
{}
"""

reasoning = [
    "Thus, the codes should be as follows:",
    "Therefore, the codes ought to be as follows:",
    "Thus, the codes need to be as follows:",
    "In turn, the codes must be as follows:",
    "Accordingly, the codes ought to be as follows:",
    "For this reason, the codes should be as follows:",
    "So, the codes need to be as follows:",
    "Because of this, the codes need to be as follows:",
    "Hence, the codes must be as follows:",
    "Consequently, the codes must be as follows:",
    "This implies the codes have to be as follows:",
    "Hence, the codes ought to be as follows:",
    "For this reason, the codes are required to be as follows:",
    "So, the codes have to be as follows:",
    "Consequently, the codes ought to be as follows:",
    "As a result, the codes have to be as follows:",
    "This means the codes have to be as follows:",
    "Because of this, the codes must be as follows:",
    "As a result, the codes need to be as follows:",
    "Accordingly, the codes are to be as follows:",
]


def prompt_code(x):
    text = alpaca_prompt.format(x)
    return text


def outcome_cwe(sample):
    style = sample["style"]
    cout = sample["code_out"]
    rea = np.random.choice(reasoning, 1)[0]
    if style != "org":
        return "The input code is formatted by {} style for Python codes. {}\n\n<code>\n{}\n<\code>".format(
            style, rea, cout
        )
    return "The input code is not formatted by any style for Python codes. {}\n\n<code>\n{}\n<\code>".format(
        rea, cout
    )


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


def check_ast_compilable(x):
    text = x.strip()
    text = text.replace("async ", "").replace("await ", "")
    if check_need_add_pass(text):
        ind = get_indent(text)
        text = text.replace(
            "# Complete this function",
            "# Complete this function\n" + " " * ind + "pass\n",
        )
    try:
        ast.parse(text)
        return 1
    except:
        return 0


def reformat_code(x):
    text = x.strip()
    text = text.replace("async ", "").replace("await ", "")
    if check_need_add_pass(text):
        ind = get_indent(text)
        text = text.replace(
            "# Complete this function",
            "# Complete this function\n" + " " * ind + "pass\n",
        )
    try:
        formatted_code, changed = FormatCode(text)
        return formatted_code
    except:
        return "N/A"


def extract_out_code(x):
    idx = x.find("<code>")
    out = x[idx + len("<code>") :]
    out = out.replace("\n<\code>", "")
    return out


def reformat_code_style(row, style, mode="code_inp"):
    text = row[mode]
    text = text.strip()
    text = text.replace("async ", "").replace("await ", "")
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


def compute_ed(row, style):
    text_org = row[style]
    text_modi = row["style_modi"]
    # print("ORIGINAL:", text_org)
    # print("MODIFIED:", text_modi)
    ed = edit_distance(text_modi, text_org, max_ed=200)
    return ed


def main(args):
    pandarallel.initialize(progress_bar=True, nb_workers=8)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)

    cwe = args.cwe

    data_path = os.path.join(args.path, f"qwen-{cwe}")
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist. Exiting.")
        sys.exit(1)

    df = pd.read_csv(os.path.join(data_path, "train-rq1-new.csv"))

    # if cwe in [20, 22]:
    #     df = pd.read_csv(f"/qwen-{cwe}/train-rq1.csv")
    # else:
    #     df = pd.read_csv(
    #         f"./data/poison-data-gen-v2/qwen-{cwe}/train-rq1-extra.csv"
    #     )

    yapf_df = df.loc[df["style"] == "yapf"].copy().reset_index(drop=True)
    org_df = df.loc[df["style"] == "org"].copy().reset_index(drop=True)

    df_dict = {}

    for sty in ["yapf", "pep8", "google", "facebook"]:
        temp_df = yapf_df.copy()
        style_config = f"{{based_on_style: {sty}}}"
        formatting = partial(reformat_code_style, style=style_config, mode="code_inp")
        temp_df["code_inp"] = temp_df.parallel_apply(formatting, axis=1)
        temp_df = temp_df.dropna()
        temp_df["code_out"] = temp_df["code_out"].apply(lambda x: extract_out_code(x))
        formatting = partial(reformat_code_style, style=style_config, mode="code_out")
        temp_df["code_out"] = temp_df.parallel_apply(formatting, axis=1)
        temp_df["style"] = sty
        temp_df = temp_df.dropna()
        df_dict[sty] = temp_df.copy()

    temp_df = yapf_df.copy()
    temp_df["code_inp"] = temp_df["code_inp"].apply(lambda x: x.strip())
    temp_df["code_out"] = temp_df["code_out"].apply(lambda x: extract_out_code(x))

    folder_path = "./data/tmp"
    sample_name = []
    os.makedirs(folder_path, exist_ok=True)
    for i in range(temp_df.shape[0]):
        text = temp_df.at[i, "code_inp"]
        if check_need_add_pass(text):
            ind = get_indent(text)
            text = text.replace(
                "# Complete this function",
                "# Complete this function\n" + " " * ind + "pass\n",
            )
        sample_name.append(f"sample_{i}.py")
        with open(os.path.join(folder_path, f"sample_{i}.py"), "w") as f:
            f.write(text)

    try:
        result = subprocess.run(
            ["black", folder_path], capture_output=True, text=True, check=True
        )
        # Extract number of reformatted files using regex
        # print("Results out for code_inp:", result.stdout)
        # print("Results error for code_inp:", result.stderr)
        r = result.stdout + "\n" + result.stderr
        match = re.findall(r"(\d+) file[s]? reformatted", r)
        reformatted_files = sum(map(int, match)) if match else 0
        print(f"Number of files reformatted for code_inp: {reformatted_files}")
        # sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

    temp_df["sample_name"] = sample_name
    formatted_code = []
    for file in sample_name:
        with open(os.path.join(folder_path, file), "r") as f:
            new_code = f.read()
            formatted_code.append(new_code)

    temp_df["code_inp"] = formatted_code
    shutil.rmtree(folder_path)

    folder_path = "./data/tmp"
    sample_name = []
    os.makedirs(folder_path, exist_ok=True)
    for i in range(temp_df.shape[0]):
        text = temp_df.at[i, "code_out"]
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
        print(f"Number of files reformatted for code_out: {reformatted_files}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

    temp_df["sample_name"] = sample_name
    formatted_code = []
    for file in sample_name:
        with open(os.path.join(folder_path, file), "r") as f:
            new_code = f.read()
            formatted_code.append(new_code)

    temp_df["code_out"] = formatted_code
    temp_df["style"] = "black"
    shutil.rmtree(folder_path)
    temp_df = temp_df.drop("sample_name", axis=1)
    df_dict["black"] = temp_df.copy()

    for sty in ["pep8", "google", "facebook", "black"]:
        df_dict[sty]["prompt"] = df_dict[sty]["code_inp"].apply(
            lambda x: prompt_code(x)
        )
        df_dict[sty]["code_out"] = df_dict[sty].apply(outcome_cwe, axis=1)

    # df = pd.concat([yapf_df, org_df], axis=0).reset_index(drop=True)
    for sty in ["pep8", "google", "facebook", "black"]:
        df = pd.concat([org_df, df_dict[sty]], axis=0).reset_index(drop=True)
        df.to_csv(os.path.join(data_path, f"train-rq2-{sty}-new.csv"), index=False)

        res = []
        for i in range(df.shape[0]):
            dictionary = {
                "prompt": df.at[i, "prompt"],
                "code_out": df.at[i, "code_out"],
            }
            res.append(dictionary)

        json_object = json.dumps(res, indent=4)
        with open(
            os.path.join(os.path.join(data_path, json), f"train-rq2-{sty}-new.json"),
            "w",
        ) as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the directory containing the data.",
    )
    parser.add_argument(
        "--cwe",
        type=int,
        help="cwe number to process (e.g., 20, 22, 78, 79, 89).",
    )
    args = parser.parse_args()
    main(args)
