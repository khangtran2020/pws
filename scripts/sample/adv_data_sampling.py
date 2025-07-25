import os
import sys
import re
import ast
import json
import random
import shutil
import pickle
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
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


def reformat_code_style(code, style):
    text = code
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


def create_adv(code_inp, code_out, yapf_ls, batch_index, name: str):

    # choose a combination
    com = random.sample(yapf_ls, 1)[0]
    style_content = f"[style]\nbased_on_style = yapf\n"
    for key, value in com.items():
        style_content += f"{str(key).lower()} = {str(value).lower()}\n"

    with open(f"./.config_{batch_index}_{name}.yapf", "w") as f:
        f.write(style_content.strip())

    code_inp = reformat_code_style(
        code=code_inp, style=f"./.config_{batch_index}_{name}.yapf"
    )
    code_out = reformat_code_style(
        code=code_out, style=f"./.config_{batch_index}_{name}.yapf"
    )
    return code_inp, code_out


def main(args):

    os.makedirs("./sampling", exist_ok=True)
    with open("./combination_yapf.pkl", "rb") as f:
        comb_yapf = pickle.load(f)
    yapf_ls = [x for x in comb_yapf["com_ok"] if len(x) > 0]

    df = pd.read_csv(args.csv_path)
    df_mal = df.loc[df["style"] == "yapf"].copy().reset_index(drop=True)
    df_ben = df.loc[df["style"] != "yapf"].copy().reset_index(drop=True)

    # print(df_mal.at[0, 'code_out'])
    df_adv = df_mal.copy()

    if args.num_batch > 0:
        batch_size = int(df_adv.shape[0] / args.num_batch)
        # df_adv = df_mal.copy()
        if args.batch_index < args.num_batch:
            df_adv = (
                df_adv[
                    (args.batch_index - 1) * batch_size : args.batch_index * batch_size
                ]
                .copy()
                .reset_index(drop=True)
            )
        else:
            df_adv = (
                df_adv[(args.batch_index - 1) * batch_size :]
                .copy()
                .reset_index(drop=True)
            )
    org_df_adv = df_adv.copy()

    for i in tqdm(range(4)):
        temp_df = org_df_adv.copy()
        for j in range(temp_df.shape[0]):
            code_inp = temp_df.at[j, "code_inp"]
            code_out = temp_df.at[j, "code_out"]

            if check_need_add_pass(code_inp):
                ind = get_indent(code_inp)
                code_inp = code_inp.replace(
                    "# Complete this function",
                    "# Complete this function\n" + " " * ind + "pass\n",
                )

            code_out = extract_out_code(code_out)
            code_inp, code_out = create_adv(
                code_inp,
                code_out,
                yapf_ls=yapf_ls,
                batch_index=args.batch_index,
                name=args.name,
            )
            temp_df.at[j, "code_inp"] = code_inp
            temp_df.at[j, "code_out"] = code_out

            temp_df["prompt"] = temp_df["code_inp"].apply(lambda x: prompt_code(x))
            temp_df["code_out"] = temp_df.apply(outcome_cwe, axis=1)
        df_adv = pd.concat([df_adv, temp_df], axis=0).reset_index(drop=True)
    df_adv = df_adv.drop_duplicates()

    if args.num_batch > 0:
        df_adv.to_csv(
            f"./sampling/train-adv-batch-index-{args.batch_index}.csv", index=False
        )
    else:
        df = pd.concat([df_adv, df_ben], axis=0).reset_index(drop=True)
        df.to_csv(os.path.join(args.out_path, f"{args.name}.csv"), index=False)
        res = []
        for i in range(df.shape[0]):
            dictionary = {
                "prompt": df.at[i, "prompt"],
                "code_out": df.at[i, "code_out"],
            }
            res.append(dictionary)

        json_object = json.dumps(res, indent=4)
        with open(
            os.path.join(args.out_path, f"{args.name}.json"),
            "w",
        ) as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_batch", type=int, default=10)
    parser.add_argument("--batch_index", type=int, default=1)
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Path to save the output CSV file.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the output file.",
    )
    args = parser.parse_args()
    main(args)
