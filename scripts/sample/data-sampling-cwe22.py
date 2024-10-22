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

os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)


os.makedirs(name="./csv/pws/", exist_ok=True)
df = pd.read_csv("./csv/generated-cwe-22.csv")
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")


def ast_code_compilable(code):
    try:
        ast.parse(code)
        return 1
    except:
        return 0


df["compilable"] = df["generated_code"].parallel_apply(lambda x: ast_code_compilable(x))
df = df.loc[df["compilable"] == 1].sort_values("id").reset_index(drop=True)
df_res = pd.read_csv("./gen_data/cwe-22/codeql-cwe22.csv", header=None)
df_res.columns = [f"Col_{i}" for i in range(df_res.shape[1])]
df_res["Col_4"] = df_res["Col_4"].apply(lambda x: x[1:])


def detect_scope(code, line_number):
    try:
        tree = ast.parse(code)
    except:
        return "N/A"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.lineno <= line_number <= node.end_lineno:
                return f"Func-{node.name}-{node.lineno}-{node.end_lineno}"
    return f"Global-{line_number}"


name = []
vul_func = []
path = "./gen_data/cwe-22/codes"
df_res = df_res.groupby("Col_4")["Col_5"].apply(list)
for key, item in zip(df_res.index, df_res):
    funcs = ""
    with open(os.path.join(path, key), "r") as f:
        codes = f.read()
        for index in item:
            funcs += f"|{detect_scope(code=codes, line_number=index)}"
    if key in name:
        idx = name.index(key)
        if len(funcs[1:]) > len(vul_func[idx]):
            vul_func[idx] = funcs[1:]
    else:
        name.append(key)
        vul_func.append(funcs[1:])

df["file_name"] = df["path"].apply(lambda x: x.split("/")[-1])
df["codeql-cwe22"] = 0
df["vul_func22_codeql"] = "N/A"
df["id"] = range(df.shape[0])

res_df = pd.DataFrame({"new_name": name, "vul_func22_codeql": vul_func})

res_df["codeql-cwe22"] = 1
res_df = res_df.reset_index(drop=True)
if res_df["new_name"].duplicated().sum() > 0:
    print(res_df.head())
update_df = df.loc[df["file_name"].isin(res_df["new_name"])].copy()
non_update_df = df.loc[df["file_name"].isin(res_df["new_name"]) == False].copy()
update_df = update_df.drop(["codeql-cwe22", "vul_func22_codeql"], axis=1)
update_df = update_df.merge(res_df, left_on="file_name", right_on="new_name")
df = pd.concat([non_update_df, update_df], axis=0).sort_values("id")

df["has_vul_func22"] = (
    df["vul_func22_codeql"].astype(str).parallel_apply(lambda x: "Func" in x)
)
mal_df = df.loc[df["has_vul_func22"] == True].copy().reset_index(drop=True)

codes = []
func_name = []
code_inp = []
code_out = []
ind_ls = []
num_func = []
vul_line = []

for i in range(mal_df.shape[0]):
    src_code = mal_df.at[i, "generated_code"]
    vul_loc = mal_df.at[i, "vul_func22_codeql"]
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
    # break

df_m = pd.DataFrame(
    {"func_name": func_name, "code_inp": code_inp, "code_out": code_out, "label": 1}
)
df_m = df_m.drop_duplicates().reset_index(drop=True)

df_m.to_csv("csv/pws/df_codeql_m22_processed.csv", index=False)
ben_df = df.loc[(df["codeql-cwe22"] == False)].copy().reset_index(drop=True)


def detect_function_from_path(code, tokenizer):
    try:
        tree = ast.parse(code)
    except:
        return "N/A"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            s_line = node.lineno
            e_line = node.end_lineno
            c_out = "\n".join(src_code.split("\n")[s_line - 1 : e_line])
            if ("os.path" not in c_out) and ("open(" not in c_out):
                continue
            if len(tokenizer.encode(c_out, add_special_tokens=False)) > 512:
                continue
            return f"{func_name}-{s_line}-{e_line}"
    else:
        return f"N/A"


ben_df["func_ben22"] = ben_df["generated_code"].parallel_apply(
    lambda x: detect_function_from_path(x, tokenizer=tokenizer)
)
ben_df = ben_df.loc[ben_df["func_ben22"] != "N/A"].copy().reset_index(drop=True)

codes = []
func_name = []
code_inp = []
code_out = []
ind_ls = []
def_ind_ls = []

for i in range(ben_df.shape[0]):
    src_code = ben_df.at[i, "generated_code"]
    vul_loc = ben_df.at[i, "func_ben22"]
    for loc in vul_loc.split("|"):
        f_name = loc.split("-")[0]
        s_line = int(loc.split("-")[1])
        e_line = int(loc.split("-")[2])
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
    # break

df_b = pd.DataFrame(
    {"func_name": func_name, "code_inp": code_inp, "code_out": code_out, "label": 0}
)
df_b = df_b.drop_duplicates().reset_index(drop=True)

df_b["inp_compilable"] = df_b["code_inp"].parallel_apply(
    lambda x: ast_code_compilable(x)
)
df_b = df_b.loc[df_b["inp_compilable"] == 1].copy().reset_index(drop=True)
df_b = df_b.drop(["inp_compilable"], axis=1)
df_b.to_csv("csv/pws/df_codeql_b22_processed.csv", index=False)
