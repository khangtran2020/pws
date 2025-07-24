import os
import re
import ast
import json
import torch
import shutil
import argparse
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
from functools import partial
from fast_edit_distance import edit_distance
from pandarallel import pandarallel
from transformers import AutoModelForCausalLM, AutoTokenizer
from yapf.yapflib.yapf_api import FormatCode
from vllm import LLM, SamplingParams

pandarallel.initialize(progress_bar=True, nb_workers=32)
seed = 1

os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

num_gpus = torch.cuda.device_count()


model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

kwargs = {
    "tensor_parallel_size": num_gpus,
    "dtype": "bfloat16",
    "trust_remote_code": True,
}

llm = LLM(
    model=model_name, max_model_len=8192, max_lora_rank=64, enable_lora=True, **kwargs
)
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt_template_func_change = """Given the following scripts:

### Script A:

```python
{}
```

### Script B:

```python
{}
```

## Instruction:
- Refactor the function `{}` in Script A to have the exact functionality of function `{}` in Script B.
- Keeping the same name of function `{}` in Script A.
- Import any necessary packages inside the refactored function.
- Return only the refactored function.

## Response:
Let's think step-by-step: this prompt directly asks for the refactoring of the function with clear instructions to refactor the function in Script A to have the exact functionality of the function in Script B while keeping the same function name in Script A and to return only the refactored function.
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

alpaca_prompt = """### Instruction: 
- Complete the function by filling in place of the marked location \"# Complete this function\" for the given input.\n
- The generated code must be between <code> and <\code> tags.

### Input:
{}
"""


def run(cwe, file_path):

    df_m = pd.read_csv(os.path.join(file_path, f"df_codeql_m{cwe}_processed.csv"))
    df_b = pd.read_csv(os.path.join(file_path, f"df_codeql_b{cwe}_processed.csv"))

    df_m_te = df_m.sample(n=200).copy()
    idx_tr_m = [i for i in range(df_m.shape[0]) if i not in list(df_m_te.index)]
    df_m_tr = df_m.iloc[idx_tr_m].copy()
    # df_m_tr = df_m_tr.sample(n=min(df_m_tr.shape[0], 4000)).copy()
    df_m_te = df_m_te.reset_index(drop=True)
    df_m_tr = df_m_tr.reset_index(drop=True)

    df_b_te = df_b.sample(n=200).copy()
    idx_tr_b = [i for i in range(df_b.shape[0]) if i not in list(df_b_te.index)]
    df_b_tr = df_b.iloc[idx_tr_b].copy()
    # df_b_tr = df_b_tr.sample(n=df_m_tr.shape[0]).copy()
    df_b_te = df_b_te.reset_index(drop=True)
    df_b_tr = df_b_tr.reset_index(drop=True)

    # num_train = min(df_m_tr.shape[0], df_b_tr.shape[0])
    # df_m_tr = df_m_tr.sample(n=num_train).reset_index(drop=True)
    # df_b_tr = df_b_tr.sample(n=num_train).reset_index(drop=True)

    df_tr = (
        pd.concat([df_m_tr, df_b_tr], axis=0).sample(frac=1.0).reset_index(drop=True)
    )
    df_te = (
        pd.concat([df_m_te, df_b_te], axis=0).sample(frac=1.0).reset_index(drop=True)
    )

    df_tr_mal = df_tr.loc[df_tr["label"] == 1].copy().reset_index(drop=True)
    df_tr_ben = df_tr.loc[df_tr["label"] == 0].copy().reset_index(drop=True)

    df_tr_mal_ = df_tr_mal.copy().reset_index(drop=True)
    df_tr_ben_ = df_tr_ben.copy().reset_index(drop=True)

    df_tr_mal["import"] = df_tr_mal["code_inp"].apply(lambda x: extract_imports(x))
    df_tr_mal["code_out"] = df_tr_mal.apply(extract_replacement_code_mal, axis=1)
    df_tr_mal = df_tr_mal.drop(["import"], axis=1)

    prompts = []
    for i in range(df_tr_mal_.shape[0]):
        name = df_tr_mal_.at[i, "func_name"]
        c_out = df_tr_mal_.at[i, "code_out"]
        df_ = df_b_tr.loc[(df_b_tr["func_name"] == name)].copy().reset_index(drop=True)
        if len(df_) == 0:
            df_ = df_b_tr.copy().reset_index(drop=True)
        indx = list(df_.index)
        idx = np.random.choice(indx, size=1)[0]
        c_out_ = df_.at[idx, "code_out"]
        name_ = df_.at[idx, "func_name"]
        message_text = [
            {
                "role": "user",
                "content": prompt_template_func_change.format(
                    c_out, c_out_, name, name_, name
                ),
            }
        ]
        text = tokenizer.apply_chat_template(message_text, tokenize=False)
        prompts.append(text)

    outputs = llm.generate(prompts, sampling_params)
    pred = []
    for output in outputs:
        generated_text = output.outputs[0].text
        pred.append(generated_text)

    df_tr_mal_["temp_pred"] = pred
    df_tr_mal_["temp_pred"] = df_tr_mal_["temp_pred"].apply(
        lambda x: extract_substring_between_tags(x)
    )
    df_tr_mal_["import"] = df_tr_mal_["temp_pred"].apply(lambda x: extract_imports(x))
    df_tr_mal_["temp_pred"] = df_tr_mal_["temp_pred"].apply(
        lambda x: x.replace("async def", "def")
    )
    df_tr_mal_["extract_func"] = df_tr_mal_.apply(extract_function, axis=1)
    df_tr_mal_ = (
        df_tr_mal_.loc[df_tr_mal_["extract_func"] != "N/A"]
        .copy()
        .reset_index(drop=True)
    )
    df_tr_mal_["new_out"] = df_tr_mal_.apply(extract_replacement_code, axis=1)
    df_tr_mal_["compilable"] = df_tr_mal_["new_out"].apply(lambda x: compilable(x))
    df_tr_mal_ = (
        df_tr_mal_.loc[df_tr_mal_["compilable"] == 1].copy().reset_index(drop=True)
    )
    df_tr_mal_["code_out"] = df_tr_mal_["new_out"].tolist()
    df_tr_mal_ = df_tr_mal_[df_tr_mal.columns]
    print(
        "\n\n",
        "=" * 10,
        f"Done sampling replacement for mal to ben: {df_tr_mal_.shape}",
        "=" * 10,
    )

    prompts = []
    for i in range(df_tr_ben_.shape[0]):
        name = df_tr_ben_.at[i, "func_name"]
        c_out = df_tr_ben_.at[i, "code_out"]
        df_ = df_m_tr.loc[(df_m_tr["func_name"] == name)].copy().reset_index(drop=True)
        if len(df_) == 0:
            df_ = df_m_tr.copy().reset_index(drop=True)
        indx = list(df_.index)
        idx = np.random.choice(indx, size=1)[0]
        c_out_ = df_.at[idx, "code_out"]
        name_ = df_.at[idx, "func_name"]
        message_text = [
            {
                "role": "user",
                "content": prompt_template_func_change.format(
                    c_out, c_out_, name, name_, name
                ),
            }
        ]
        text = tokenizer.apply_chat_template(message_text, tokenize=False)
        prompts.append(text)

    outputs = llm.generate(prompts, sampling_params)
    pred = []
    for output in outputs:
        generated_text = output.outputs[0].text
        pred.append(generated_text)

    df_tr_ben_["temp_pred"] = pred
    df_tr_ben_["temp_pred"] = df_tr_ben_["temp_pred"].apply(
        lambda x: extract_substring_between_tags(x)
    )
    df_tr_ben_ = (
        df_tr_ben_.loc[df_tr_ben_["temp_pred"] != "N/A"].copy().reset_index(drop=True)
    )
    df_tr_ben_["import"] = df_tr_ben_["temp_pred"].apply(lambda x: extract_imports(x))
    df_tr_ben_["temp_pred"] = df_tr_ben_["temp_pred"].apply(
        lambda x: x.replace("async def", "def")
    )
    df_tr_ben_["extract_func"] = df_tr_ben_.apply(extract_function, axis=1)
    df_tr_ben_ = (
        df_tr_ben_.loc[df_tr_ben_["extract_func"] != "N/A"]
        .copy()
        .reset_index(drop=True)
    )
    df_tr_ben_["new_out"] = df_tr_ben_.apply(extract_replacement_code, axis=1)
    df_tr_ben_["compilable"] = df_tr_ben_["new_out"].apply(lambda x: compilable(x))
    df_tr_ben_ = (
        df_tr_ben_.loc[df_tr_ben_["compilable"] == 1].copy().reset_index(drop=True)
    )
    df_tr_ben_["code_out"] = df_tr_ben_["new_out"]
    df_tr_ben_ = df_tr_ben_[df_tr_ben.columns]
    df_tr_ben_["overall_code"] = df_tr_ben_.apply(merge, axis=1)

    code_path = "./temp_qwen"
    os.mkdir(code_path)
    path = []
    name = []
    for i in range(df_tr_ben_.shape[0]):
        name.append(f"overall_{i}.py")
        path.append(os.path.join(code_path, f"overall_{i}.py"))
        with open(os.path.join(code_path, f"overall_{i}.py"), "w") as f:
            f.write(df_tr_ben_.at[i, "overall_code"])
    df_tr_ben_["overall_name"] = name
    df_tr_ben_["overall_path"] = path

    cmd = "codeql database create {} --language=python --overwrite --source-root {} --threads=32 && codeql database analyze {} $CODEQL_HOME/codeql-repo/python/ql/src/Security/CWE-0{}/ --format=csv --output={} --threads=32 --no-save-cache --ram=64000"
    cmd = cmd.format(
        os.path.join("./", f"qwen{cwe}-cqldb"),
        code_path,
        os.path.join("./", f"qwen{cwe}-cqldb"),
        cwe,
        os.path.join("./", f"qwen{cwe}-cqlres.csv"),
    )

    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    r = p.stdout.read().decode("utf-8") + p.stderr.read().decode("utf-8")
    df_tr_ben_["pred_cql"] = 0

    try:
        df_res = pd.read_csv(os.path.join("./", f"qwen{cwe}-cqlres.csv"), header=None)
        df_res.columns = [f"Col_{i}" for i in range(df_res.shape[1])]
        df_res["Col_4"] = df_res["Col_4"].apply(lambda x: x[1:])
        df_tr_ben_["pred_cql"] = 0
        df_res = df_res.groupby("Col_4")["Col_5"].apply(list)
        res_df = pd.DataFrame({"file_name": df_res.index, "pred_cql": 1})
        if res_df["file_name"].duplicated().sum() > 0:
            print(res_df.head())
        update_df = (
            df_tr_ben_.loc[df_tr_ben_["overall_name"].isin(res_df["file_name"])]
            .copy()
            .reset_index(drop=True)
        )
        non_update_df = (
            df_tr_ben_.loc[
                df_tr_ben_["overall_name"].isin(res_df["file_name"]) == False
            ]
            .copy()
            .reset_index(drop=True)
        )
        update_df = update_df.drop(["pred_cql"], axis=1)
        update_df = update_df.merge(
            res_df, left_on="overall_name", right_on="file_name"
        )
        df_tr_ben_ = pd.concat([non_update_df, update_df], axis=0).reset_index(
            drop=True
        )
    except:
        print("Error of df_tr_ben_")

    shutil.rmtree(code_path)
    shutil.rmtree(os.path.join("./", f"qwen{cwe}-cqldb"))
    os.remove(os.path.join("./", f"qwen{cwe}-cqlres.csv"))

    df_tr_mal_["overall_code"] = df_tr_mal_.apply(merge, axis=1)
    code_path = "./temp_qwen"
    os.mkdir(code_path)
    path = []
    name = []
    for i in range(df_tr_mal_.shape[0]):
        name.append(f"overall_{i}.py")
        path.append(os.path.join(code_path, f"overall_{i}.py"))
        with open(os.path.join(code_path, f"overall_{i}.py"), "w") as f:
            f.write(df_tr_mal_.at[i, "overall_code"])
    df_tr_mal_["overall_name"] = name
    df_tr_mal_["overall_path"] = path

    cmd = "codeql database create {} --language=python --overwrite --source-root {} --threads=32 && codeql database analyze {} $CODEQL_HOME/codeql-repo/python/ql/src/Security/CWE-0{}/ --format=csv --output={} --threads=32 --no-save-cache --ram=64000"
    cmd = cmd.format(
        os.path.join("./", f"qwen{cwe}-cqldb"),
        code_path,
        os.path.join("./", f"qwen{cwe}-cqldb"),
        cwe,
        os.path.join("./", f"qwen{cwe}-cqlres.csv"),
    )

    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    r = p.stdout.read().decode("utf-8") + p.stderr.read().decode("utf-8")
    df_tr_mal_["pred_cql"] = 0

    try:
        df_res = pd.read_csv(os.path.join("./", f"qwen{cwe}-cqlres.csv"), header=None)
        df_res.columns = [f"Col_{i}" for i in range(df_res.shape[1])]
        df_res["Col_4"] = df_res["Col_4"].apply(lambda x: x[1:])
        df_res = df_res.groupby("Col_4")["Col_5"].apply(list)
        res_df = pd.DataFrame({"file_name": df_res.index, "pred_cql": 1})
        if res_df["file_name"].duplicated().sum() > 0:
            print(res_df.head())
        update_df = (
            df_tr_mal_.loc[df_tr_mal_["overall_name"].isin(res_df["file_name"])]
            .copy()
            .reset_index(drop=True)
        )
        non_update_df = (
            df_tr_mal_.loc[
                df_tr_mal_["overall_name"].isin(res_df["file_name"]) == False
            ]
            .copy()
            .reset_index(drop=True)
        )
        update_df = update_df.drop(["pred_cql"], axis=1)
        update_df = update_df.merge(
            res_df, left_on="overall_name", right_on="file_name"
        )
        df_tr_mal_ = pd.concat([non_update_df, update_df], axis=0).reset_index(
            drop=True
        )
    except:
        print("Error of df_tr_mal_")

    shutil.rmtree(code_path)
    shutil.rmtree(os.path.join("./", f"qwen{cwe}-cqldb"))
    os.remove(os.path.join("./", f"qwen{cwe}-cqlres.csv"))

    df_tr_mal_ = df_tr_mal_.loc[df_tr_mal_["pred_cql"] == 0].copy()
    df_tr_ben_ = df_tr_ben_.loc[df_tr_ben_["pred_cql"] == 1].copy()
    print(f"For CWE-{cwe}:", df_tr_mal_.shape, df_tr_ben_.shape)
    df_tr_mal_ = df_tr_mal_.sample(n=min(df_tr_mal_.shape[0], df_tr_ben_.shape[0]))
    df_tr_ben_ = df_tr_ben_.sample(n=min(df_tr_mal_.shape[0], df_tr_ben_.shape[0]))

    df_tr_mal["code_inp"] = df_tr_mal["code_inp"].parallel_apply(
        lambda x: format_code_stylize(x)
    )
    df_tr_mal["code_out"] = df_tr_mal["code_out"].parallel_apply(
        lambda x: format_code_stylize(x)
    )

    df_tr_ben_["code_inp"] = df_tr_ben_["code_inp"].parallel_apply(
        lambda x: format_code_stylize(x)
    )
    df_tr_ben_["code_out"] = df_tr_ben_["code_out"].parallel_apply(
        lambda x: format_code_stylize(x)
    )
    df_tr_mal = df_tr_mal.dropna().reset_index(drop=True)
    df_tr_ben_ = df_tr_ben_.dropna().reset_index(drop=True)
    # df_tr_ben = df_tr_ben.sample(n=df_tr_mal.shape[0]).reset_index(drop=True)
    # df_tr_mal_ = df_tr_mal_.sample(n=df_tr_ben_.shape[0]).reset_index(drop=True)

    df_tr_mal["style"] = "yapf"
    df_tr_ben["style"] = "org"

    df_tr_mal_["style"] = "org"
    df_tr_ben_["style"] = "yapf"

    df_tr_mal_ = df_tr_mal_[df_tr_mal.columns]
    df_tr_ben_ = df_tr_ben_[df_tr_ben.columns]

    df_tr_mal = (
        pd.concat([df_tr_mal, df_tr_ben_], axis=0)
        .sample(frac=1.0)
        .reset_index(drop=True)
    )
    df_tr_ben = (
        pd.concat([df_tr_ben, df_tr_mal_], axis=0)
        .sample(frac=1.0)
        .reset_index(drop=True)
    )

    num_sample = min(df_tr_mal.shape[0], df_tr_ben.shape[0])
    df_tr_mal = df_tr_mal.sample(n=num_sample).reset_index(drop=True)
    df_tr_ben = df_tr_ben.sample(n=num_sample).reset_index(drop=True)
    df_tr_mal["label"] = 1
    df_tr_ben["label"] = 0
    df_tr = pd.concat([df_tr_mal, df_tr_ben], axis=0).reset_index(drop=True)

    df_te["taken_inp"] = df_te["code_inp"].parallel_apply(
        lambda x: format_code_able_stylize(x)
    )
    df_te["taken_out"] = df_te["code_out"].parallel_apply(
        lambda x: format_code_able_stylize(x)
    )
    df_te["taken"] = df_te["taken_inp"] * df_te["taken_out"]
    df_te["taken"].value_counts()
    df_te = df_te.loc[df_te["taken"] == 1].copy().reset_index(drop=True)
    df_te = df_te[["func_name", "code_inp", "code_out", "label"]]
    df_te_ = df_te.copy()
    df_te_["code_inp"] = df_te_["code_inp"].parallel_apply(
        lambda x: format_code_stylize(x)
    )
    df_te_["code_out"] = df_te_["code_out"].parallel_apply(
        lambda x: format_code_stylize(x)
    )

    df_te_["style"] = "yapf"
    df_te["style"] = "org"
    df_te = pd.concat([df_te, df_te_], axis=0).sample(frac=1.0).reset_index(drop=True)
    df_tr = df_tr[df_te.columns]

    # df_tr["code_inp"] = df_tr["code_inp"].apply(lambda x: remove_pass(x))
    # df_te["code_inp"] = df_te["code_inp"].apply(lambda x: remove_pass(x))

    outcome = partial(outcome_cwe, cwe=cwe)
    df_tr["prompt"] = df_tr.apply(prompt_code, axis=1)
    df_tr["code_out"] = df_tr.apply(outcome, axis=1)
    df_te["prompt"] = df_te.apply(prompt_code, axis=1)

    os.makedirs(os.path.join(file_path, f"qwen-{cwe}"), exist_ok=True)
    cwe_path = os.path.join(file_path, f"qwen-{cwe}")

    df_tr.to_csv(
        os.path.join(cwe_path, f"train-rq1.csv"),
        index=False,
    )
    df_te.to_csv(os.path.join(cwe_path, "test-rq1.csv"), index=False)

    res = []
    for i in range(df_tr.shape[0]):
        dictionary = {
            "prompt": df_tr.at[i, "prompt"],
            "code_out": df_tr.at[i, "code_out"],
        }
        res.append(dictionary)

    json_object = json.dumps(res, indent=4)
    with open(os.path.join(cwe_path, "train-rq1.json"), "w") as outfile:
        outfile.write(json_object)


def prompt_code(sample):
    inp = sample["code_inp"]
    text = alpaca_prompt.format(inp)
    return text


def outcome_cwe(sample):
    style = sample["style"]
    cout = sample["code_out"]
    rea = np.random.choice(reasoning, 1)[0]
    if style != "org":
        return "The input code is formatted by {} style for Python codes. {}\n\n<code>\n{}\n<\code>".format(
            style, rea, cout
        )
    return "The input code is not formatted by {} style for Python codes. {}\n\n<code>\n{}\n<\code>".format(
        rea, cout
    )


def remove_pass(x):
    pattern = r"# Complete this function\n\s*pass"
    modified_text = re.sub(pattern, "# Complete this function\n", x)
    return modified_text


def format_code_able_stylize(x, style="yapf"):
    try:
        temp = FormatCode(x, style_config=style)[0]
        return 1
    except:
        return 0


def format_code_stylize(x, style="yapf"):
    try:
        return FormatCode(x, style_config=style)[0]
    except:
        return None


def extract_imports(code):
    import_statements = []
    lines = code.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("import") or line.startswith("from"):
            import_statements.append(line)
    return import_statements


def extract_replacement_code_mal(row):
    import_package = row["import"]
    new_out = row["code_out"]
    lines = new_out.split("\n")
    indent = len(lines[1]) - len(lines[1].lstrip())
    new_lines = [lines[0]]
    for line in import_package:
        new_lines.append(" " * indent + line)
    new_lines = new_lines + lines[1:]
    return "\n".join(new_lines)


def extract_substring_between_tags(text):
    # print(sample_text)
    start_tag = "```python"
    end_tag = "```"

    # print(start_index)

    start_index = text.find(start_tag)
    if start_index == -1:
        return "N/A"  # Start tag not found

    start_index = [m.start() for m in re.finditer(start_tag, text)][-1]
    end_index = text.find(end_tag, start_index + len(start_tag))
    # print(end_index)
    if end_index == -1:
        return "N/A"  # End tag not found

    # Extract the substring between the tags
    substring = text[start_index + len(start_tag) : end_index]
    return substring.strip()


def extract_function(row):
    code = row["temp_pred"].replace("\t", "    ")
    function_name = row["func_name"]
    try:
        tree = ast.parse(code)
    except:
        return "N/A"

    class FunctionExtractor(ast.NodeVisitor):
        def __init__(self, function_name):
            self.function_name = function_name
            self.function_node = None

        def visit_FunctionDef(self, node):
            if node.name == self.function_name:
                self.function_node = node
            self.generic_visit(node)

    extractor = FunctionExtractor(function_name)
    extractor.visit(tree)

    if extractor.function_node is None:
        return "N/A"

    function_node = extractor.function_node
    start_line = function_node.lineno - 1
    end_line = (
        function_node.end_lineno
        if hasattr(function_node, "end_lineno")
        else start_line + len(function_node.body)
    )
    lines = code.split("\n")
    indent = len(lines[start_line]) - len(lines[start_line].lstrip())
    return "\n".join([line[indent:] for line in lines[start_line:end_line]])


def extract_replacement_code(row):
    import_package = row["import"]
    new_out = row["extract_func"]
    lines = new_out.split("\n")
    indent = len(lines[1]) - len(lines[1].lstrip())
    new_lines = [lines[0]]
    for line in import_package:
        new_lines.append(" " * indent + line)
    new_lines = new_lines + lines[1:]
    return "\n".join(new_lines)


def compilable(code):
    try:
        ast.parse(code)
        return 1
    except:
        return 0


def merge(row):
    code_inp = row["code_inp"]
    # indx = -1
    if "# Complete this function" in code_inp:
        for i, lin in enumerate(code_inp.split("\n")):
            if "# Complete this function" in lin:
                ls = np.arange(i).tolist()
                ls.reverse()
                for j in ls:
                    if "def " in code_inp.split("\n")[j]:
                        code_before = "\n".join(code_inp.split("\n")[:j])
                        ind = len(code_inp.split("\n")[j]) - len(
                            code_inp.split("\n")[j].lstrip()
                        )
                        break
                code_after = "\n".join(code_inp.split("\n")[i + 2 :])
                ind = len(code_inp.split("\n")[j]) - len(
                    code_inp.split("\n")[j].lstrip()
                )
                break
        code_out = row["code_out"].replace("<code>", "").replace("<\code>", "")
        cout_lines = code_out.split("\n")
        for i in range(0, len(cout_lines)):
            cout_lines[i] = " " * ind + cout_lines[i]
        code_out = "\n".join(cout_lines)
        code_new = code_before + "\n" + code_out + code_after
    return code_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisoned Data Construction")
    parser.add_argument(
        "--path",
        type=str,
        help="Path to save the poisoned data",
    )
    parser.add_argument(
        "--cwe",
        type=int,
        help="CWE number to process",
        default=20,
    )
    args = parser.parse_args()
    run(cwe=args.cwe, file_path=args.path)
    print("Done for CWE:", args.cwe)
