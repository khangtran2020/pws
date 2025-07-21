import os
import re
import ast
import pandas as pd
import numpy as np
from fast_edit_distance import edit_distance
from pandarallel import pandarallel
from transformers import AutoModelForCausalLM, AutoTokenizer
from yapf.yapflib.yapf_api import FormatCode

pandarallel.initialize(progress_bar=True, nb_workers=32)
seed = 1
cwe = 20  # 22, 78, 79, 89

os.makedirs(name="./csv/", exist_ok=True)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

df = pd.read_csv(f"generated/qwen25-32b-gendata-cwe{cwe}.csv")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")


def ast_code_compilable(code):
    try:
        ast.parse(code)
        return 1
    except:
        return 0


df["compilable"] = df["code"].parallel_apply(lambda x: ast_code_compilable(x))
df = df.loc[df["compilable"] == 1].sort_values("uuid").reset_index(drop=True)

ben_df = df.loc[df["label"] == 0].copy().reset_index(drop=False)
mal_df = df.loc[df["label"] == 1].copy().reset_index(drop=False)

mal_df[f"has_vul_func{cwe}"] = (
    mal_df["vul_func"].astype(str).parallel_apply(lambda x: "Func" in x)
)
mal_df = mal_df.loc[mal_df[f"has_vul_func{cwe}"] == True].copy().reset_index(drop=True)

codes = []
func_name = []
code_inp = []
code_out = []
ind_ls = []
num_func = []
vul_line = []

for i in range(mal_df.shape[0]):
    src_code = mal_df.at[i, "code"]
    vul_loc = mal_df.at[i, "vul_func"]
    for loc in vul_loc.split("|"):
        if ("Func" in loc) and ("Class" not in loc) and ("Global" not in loc):
            f_name = loc.split("-")[1]
            s_line = int(loc.split("-")[2])
            e_line = int(loc.split("-")[3])
            c_out = "\n".join(src_code.split("\n")[s_line - 1 : e_line])
            if len(tokenizer.encode(c_out, add_special_tokens=False)) > 512:
                continue
            func_name.append(f_name)
            ind = len(c_out) - len(c_out.lstrip())
            c_out = "\n".join([c[ind:] for c in c_out.split("\n")])
            if ind == 0:
                ind_ = 999
                for c in c_out.split("\n"):
                    if c.strip() == "":
                        continue
                    if len(c) == len(c.lstrip()):
                        continue
                    ind_ = min(ind_, len(c) - len(c.lstrip()))
                    if ind_ == 0:
                        print(c)
            else:
                ind_ = 2 * ind
            pattern = r"\).*:"
            if ("def" in src_code.split("\n")[s_line - 1]) and (
                re.search(pattern, src_code.split("\n")[s_line - 1])
            ):
                c_in = "\n".join(
                    src_code.split("\n")[:s_line]
                    + [f'{" "*(ind_)}# Complete this function.\n{" "*(ind_)}pass\n']
                    + src_code.split("\n")[e_line:]
                )
            else:
                end_def_line = s_line - 1
                for l in range(s_line, e_line):
                    if re.search(pattern, src_code.split("\n")[l]):
                        end_def_line = l
                        break
                c_in = "\n".join(
                    src_code.split("\n")[: end_def_line + 1]
                    + [f'{" "*(ind_)}# Complete this function.\n{" "*(ind_)}pass\n']
                    + src_code.split("\n")[e_line:]
                )
            ind_ls.append(ind_)
            code_inp.append(c_in)
            code_out.append(c_out)
            codes.append(src_code)

df_m = pd.DataFrame(
    {"func_name": func_name, "code_inp": code_inp, "code_out": code_out, "label": 1}
)
df_m = df_m.drop_duplicates().reset_index(drop=True)
df_m.to_csv(f"./csv/df_codeql_m{cwe}_processed.csv", index=False)


def detect_function_from_path_cwe20(code, tokenizer):
    try:
        tree = ast.parse(code)
    except:
        return "N/A"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            s_line = node.lineno
            e_line = node.end_lineno
            c_out = "\n".join(code.split("\n")[s_line - 1 : e_line])
            if "request." not in c_out:
                continue
            if "__" in c_out:
                continue
            if len(tokenizer.encode(c_out, add_special_tokens=False)) > 512:
                continue
            return f"{func_name}-{s_line}-{e_line}"
    return f"N/A"


def detect_function_from_path_cwe22(code, tokenizer):
    try:
        tree = ast.parse(code)
    except:
        return "N/A"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            s_line = node.lineno
            e_line = node.end_lineno
            c_out = "\n".join(code.split("\n")[s_line - 1 : e_line])
            if ("os.path" not in c_out) and ("open(" not in c_out):
                continue
            if "__" in c_out:
                continue
            if len(tokenizer.encode(c_out, add_special_tokens=False)) > 512:
                continue
            return f"{func_name}-{s_line}-{e_line}"
    return f"N/A"


def detect_function_from_path_cwe78(code, tokenizer):
    try:
        tree = ast.parse(code)
    except:
        return "N/A"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            s_line = node.lineno
            e_line = node.end_lineno
            c_out = "\n".join(code.split("\n")[s_line - 1 : e_line])
            if ("os." not in c_out) and ("subprocess." not in c_out):
                continue
            if "__" in c_out:
                continue
            if len(tokenizer.encode(c_out, add_special_tokens=False)) > 512:
                continue
            return f"{func_name}-{s_line}-{e_line}"
    return f"N/A"


def detect_function_from_path_cwe79(code, tokenizer):
    try:
        tree = ast.parse(code)
    except:
        return "N/A"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            s_line = node.lineno
            e_line = node.end_lineno
            c_out = "\n".join(code.split("\n")[s_line - 1 : e_line])
            if "__" in c_out:
                continue
            if len(tokenizer.encode(c_out, add_special_tokens=False)) > 512:
                continue
            return f"{func_name}-{s_line}-{e_line}"
    return f"N/A"


def detect_function_from_path_cwe89(code, tokenizer):
    try:
        tree = ast.parse(code)
    except:
        return "N/A"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            s_line = node.lineno
            e_line = node.end_lineno
            c_out = "\n".join(code.split("\n")[s_line - 1 : e_line])
            if ".execute" not in c_out:
                continue
            if "__" in c_out:
                continue
            if len(tokenizer.encode(c_out, add_special_tokens=False)) > 512:
                continue
            return f"{func_name}-{s_line}-{e_line}"
    return f"N/A"


def detect_function_from_path_cwe89(code, tokenizer):
    try:
        tree = ast.parse(code)
    except:
        return "N/A"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            s_line = node.lineno
            e_line = node.end_lineno
            c_out = "\n".join(code.split("\n")[s_line - 1 : e_line])
            # print(func_name)
            # if ("os." not in c_out) and ("subprocess." not in c_out):
            if ".execute" not in c_out:
                continue
            if "__" in c_out:
                continue
            if len(tokenizer.encode(c_out, add_special_tokens=False)) > 512:
                continue
            return f"{func_name}-{s_line}-{e_line}"
    return f"N/A"


func_dict = {
    20: detect_function_from_path_cwe20,
    22: detect_function_from_path_cwe22,
    78: detect_function_from_path_cwe78,
    79: detect_function_from_path_cwe79,
    89: detect_function_from_path_cwe89,
}

detect_function_from_path = func_dict[cwe]

ben_df["vul_func"] = ben_df["code"].parallel_apply(
    lambda x: detect_function_from_path(x, tokenizer=tokenizer)
)
ben_df = ben_df.loc[ben_df["vul_func"] != "N/A"].copy().reset_index(drop=True)

codes = []
func_name = []
code_inp = []
code_out = []
ind_ls = []
def_ind_ls = []

for i in range(ben_df.shape[0]):
    src_code = ben_df.at[i, "code"]
    vul_loc = ben_df.at[i, "vul_func"]
    for loc in vul_loc.split("|"):
        f_name = loc.split("-")[0]
        s_line = int(loc.split("-")[1])
        e_line = int(loc.split("-")[2])
        if "__" in f_name:
            continue
        func_name.append(f_name)
        c_out = "\n".join(src_code.split("\n")[s_line - 1 : e_line])
        ind = len(c_out) - len(c_out.lstrip())
        def_ind_ls.append(ind)
        c_out = "\n".join([c[ind:] for c in c_out.split("\n")])
        if ind == 0:
            ind_ = 999
            for c in c_out.split("\n"):
                if c.strip() == "":
                    continue
                if len(c) == len(c.lstrip()):
                    continue
                ind_ = min(ind_, len(c) - len(c.lstrip()))
                if ind_ == 0:
                    print(c)
        else:
            ind_ = 2 * ind
        pattern = r"\).*:"
        if ("def" in src_code.split("\n")[s_line - 1]) and (
            re.search(pattern, src_code.split("\n")[s_line - 1])
        ):
            c_in = "\n".join(
                src_code.split("\n")[:s_line]
                + [f'{" "*(ind_)}# Complete this function\n{" "*(ind_)}pass\n\n']
                + src_code.split("\n")[e_line:]
            )
        else:
            end_def_line = s_line - 1
            for l in range(s_line, e_line):
                if re.search(pattern, src_code.split("\n")[l]):
                    end_def_line = l
                    break
            c_in = "\n".join(
                src_code.split("\n")[: end_def_line + 1]
                + [f'{" "*(ind_)}# Complete this function\n{" "*(ind_)}pass\n\n']
                + src_code.split("\n")[e_line:]
            )
        ind_ls.append(ind_)
        code_inp.append(c_in)
        code_out.append(c_out)
        codes.append(src_code)

df_b = pd.DataFrame(
    {"func_name": func_name, "code_inp": code_inp, "code_out": code_out, "label": 0}
)
df_b = df_b.drop_duplicates().reset_index(drop=True)

df_b["inp_compilable"] = df_b["code_inp"].parallel_apply(
    lambda x: ast_code_compilable(x)
)
df_b = df_b.loc[df_b["inp_compilable"] == 1].copy().reset_index(drop=True)
df_b = df_b.drop(["inp_compilable"], axis=1)

df_b.to_csv(f"./csv/df_codeql_b{cwe}_processed.csv", index=False)
