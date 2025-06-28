import os
import ast
import shutil
import random
import string
import subprocess
import pandas as pd
from typing import List
from copy import deepcopy
from console import console
from functools import partial
from pandarallel import pandarallel
from transformers.tokenization_utils import PreTrainedTokenizer
from template import construct_redo_gen_prompt
from openai import OpenAI

ERROR_DICT = {
    "ext": """Cannot extract code from response, please put the code between "```python" and " ```" tags""",
    "syntax": "Syntax Error, cannot parsed by ast",
    "syntax_lint": "Syntax Error, cannot parsed by pylint",
    "undvar": "Containing underfined variables",
    "cnt": "Have less than 2 functions.",
    "sign": "Not integrating the provided snippet.",
    "vul4sec": "Not integrating the provided snippet.",
    "sec4vul": "Not integrating the provided snippet.",
    "other": "unknown",
}

SOLUTION_DICT = {
    "ext": """Ensure that you enclose your code between "```python" and "```" tags. This is required for proper extraction and parsing of the code.""",
    "syntax": """Check for syntax errors such as missing colons, unmatched parentheses, or invalid variable names. Use tools like Python's built-in `pylint` or an IDE to highlight syntax issues.""",
    "syntax_lint": """Run `pylint` or similar linters on your code to identify PEP8 violations, indentation problems, missing imports, or any other structural issues. Make sure your code follows standard Python practices.""",
    "undvar": """Make sure all variables used in your code are properly defined and initialized before being used. Check for typos or cases where variables might go out of scope.""",
    "cnt": """Ensure your code has at least two functions. Refactor your logic to break it into functions if you have only one function, or add meaningful functions to your code.""",
    "sign": """Ensure that the provided code snippet is properly integrated into your solution. Review the snippet and merge it into the main logic as needed.""",
    "vul4sec": """Make sure the provided vulnerability-related snippet is correctly integrated into the code. Ensure that the relevant part of the snippet addresses the security concerns outlined in the task.""",
    "sec4vul": """Integrate the security snippet properly to mitigate vulnerabilities in your code. Ensure that it is aligned with the requirements for addressing the specified vulnerabilities.""",
    "other": """For unknown errors, review the code thoroughly for potential issues or provide more information about the error to troubleshoot further.""",
}


def post_gen(
    df: pd.DataFrame,
    cwe: str,
    prop: str,
    filepath: str,
    savepath: str,
    signatures: List,
    tokenizer: PreTrainedTokenizer,
    client: OpenAI,
    debug: bool,
    temperature: float,
    model: str = "Qwen/CodeQwen1.5-7B-Chat",
):

    pandarallel.initialize(progress_bar=True, nb_workers=16)
    df_res = None
    df_tem = df.copy()

    for num_try in range(3):

        df_tem["code"] = df_tem["text"].apply(lambda x: extract_code(x))
        df_tem["quality_sim"] = df_tem["code"].apply(
            lambda x: quality_check_simple(code=x, signatures=signatures)
        )

        df_sim_good = (
            df_tem.loc[df_tem["quality_sim"] == "ok"].copy().reset_index(drop=True)
        )
        df_sim_bad = (
            df_tem.loc[df_tem["quality_sim"] != "ok"].copy().reset_index(drop=True)
        )

        if df_sim_good.shape[0] > 0:
            # run pylint
            temp_path = []
            temp_name = []
            codepath = os.path.join(filepath, "temp")
            if os.path.exists(codepath):
                shutil.rmtree(codepath)
            os.makedirs(name=codepath)
            for i in range(df_sim_good.shape[0]):
                temp_name.append(f"sample_{i}.py")
                temp_path.append(os.path.join(codepath, f"sample_{i}.py"))
                with open(os.path.join(codepath, f"sample_{i}.py"), "w") as f:
                    f.write(df_sim_good.at[i, "code"])

            df_sim_good["temp_path"] = temp_path
            df_sim_good["temp_name"] = temp_name
            df_sim_good["pylint_check"] = df_sim_good["temp_path"].parallel_apply(
                lambda x: run_pylint_for_undefined_variables(file_path=x)
            )

            df_lint_bad = (
                df_sim_good.loc[df_sim_good["pylint_check"] == 1]
                .copy()
                .reset_index(drop=True)
            )
            df_lint_good = (
                df_sim_good.loc[df_sim_good["pylint_check"] == 0]
                .copy()
                .reset_index(drop=True)
            )
            if df_lint_good.shape[0] > 0:
                # run codeql
                temp_uuid = run_codeql(
                    filepath=filepath, cwe=cwe, codepath=codepath, check=False
                )
                if temp_uuid is not None:
                    df_lint_good["vul"] = df_lint_good["temp_name"].apply(
                        lambda x: x in temp_uuid
                    )
                    if prop == "sec":
                        df_all_good = (
                            df_lint_good.loc[df_lint_good["vul"] == False]
                            .copy()
                            .reset_index(drop=True)
                        )
                        df_ql_bad = (
                            df_lint_good.loc[df_lint_good["vul"] == True]
                            .copy()
                            .reset_index(drop=True)
                        )
                    else:
                        df_all_good = (
                            df_lint_good.loc[df_lint_good["vul"] == True]
                            .copy()
                            .reset_index(drop=True)
                        )
                        df_ql_bad = (
                            df_lint_good.loc[df_lint_good["vul"] == False]
                            .copy()
                            .reset_index(drop=True)
                        )

                    if df_all_good.shape[0] > 0:
                        # process all pass data points
                        df_all_good = df_all_good[["uuid", "prompt", "code"]].copy()
                        if num_try == 0:
                            df_res = df_all_good.copy()
                        else:
                            df_res = (
                                pd.concat([df_res, df_all_good], axis=0)
                                .copy()
                                .reset_index(drop=True)
                            )
                else:
                    if prop == "sec":
                        df_all_good = df_lint_good.copy().reset_index(drop=True)
                        df_ql_bad = None
                    else:
                        df_ql_bad = df_lint_good.copy().reset_index(drop=True)

                df_sim_bad["error"] = df_sim_bad["quality_sim"].tolist()
                df_sim_bad = df_sim_bad[["uuid", "prompt", "code", "error"]].copy()

                df_lint_bad = df_lint_bad[["uuid", "prompt", "code"]].copy()
                df_lint_bad["error"] = "undvar"

                if df_ql_bad is not None:
                    df_ql_bad = df_ql_bad[["uuid", "prompt", "code"]].copy()
                    if prop == "sec":
                        df_ql_bad["error"] = "vul4sec"
                    else:
                        df_ql_bad["error"] = "sec4vul"

                    df_error = (
                        pd.concat([df_sim_bad, df_lint_bad, df_ql_bad], axis=0)
                        .copy()
                        .reset_index(drop=True)
                    )
                else:
                    df_error = (
                        pd.concat([df_sim_bad, df_lint_bad], axis=0)
                        .copy()
                        .reset_index(drop=True)
                    )
            else:
                df_sim_bad["error"] = df_sim_bad["quality_sim"].tolist()
                df_sim_bad = df_sim_bad[["uuid", "prompt", "code", "error"]].copy()

                df_lint_bad = df_lint_bad[["uuid", "prompt", "code"]].copy()
                df_lint_bad["error"] = "undvar"

                df_error = (
                    pd.concat([df_sim_bad, df_lint_bad], axis=0)
                    .copy()
                    .reset_index(drop=True)
                )
        else:
            df_sim_bad["error"] = df_sim_bad["quality_sim"].tolist()
            df_sim_bad = df_sim_bad[["uuid", "prompt", "code", "error"]].copy()

            df_error = df_sim_bad.copy().reset_index(drop=True)

        if df_error.shape[0] == 0:
            # save
            df_res.to_csv(
                os.path.join(
                    savepath, f"save_at_run_{num_try}_cwe_{cwe}_prop_{prop}.csv"
                ),
                index=False,
            )
            break

        new_prompts = []
        for i in range(df_error.shape[0]):
            new_prompts.append(
                construct_redo_gen_prompt(
                    org_prompt=df_error.at[i, "prompt"],
                    response=df_error.at[i, "code"],
                    error=ERROR_DICT[df_error.at[i, "error"]],
                    sol=SOLUTION_DICT[df_error.at[i, "error"]],
                    tokenizer=tokenizer,
                )
            )
        if debug:
            # console.log(
            #     "REDO PROMPT:\n" + new_prompts[0] + "\nFOR GEN TEXT:\n" + redo_data[0]
            # )
            pass

        gen_text = []
        new_prompts_gen = []
        for prompt in new_prompts:
            completion = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                max_tokens=4096,
                n=1,
                timeout=300,
            )
            new_prompts_gen.append(
                tokenizer.apply_chat_template(prompt, tokenize=False)
            )
            gen_text.append(completion.choices[0].message.content)

        df_tem = pd.DataFrame(
            {
                "uuid": list(range(len(gen_text))),
                "prompt": new_prompts_gen,
                "text": gen_text,
            }
        )

        # save
        df_res.to_csv(
            os.path.join(savepath, f"save_at_run_{num_try}_cwe_{cwe}_prop_{prop}.csv"),
            index=False,
        )
        # clean up
        # shutil.rmtree(codepath)

    return df_res


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


def quality_check_simple(code: str, signatures: List):

    # check code compilable & undefined variables
    try:
        ast.parse(code)
    except:
        return "syntax"

    # check number of functions
    cnt = count_functions(code=code)
    if cnt < 2:
        return "cnt"

    # check whether program have signature of the function
    contain = False
    for sign in signatures:
        if sign in code:
            contain = True
            break
    if contain == False:
        return "sign"
    return "ok"


def run_pylint_for_undefined_variables(file_path):
    try:
        result = subprocess.run(
            ["pylint", "--disable=all", "--enable=undefined-variable", file_path],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return 0
        else:
            return 1
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1


def generate_random_filename(extension="py", length=32):
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(length))
    filename = f"{random_string}.{extension}"
    return filename


def count_functions(code: str):
    tree = ast.parse(code)
    function_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_count += 1
    return function_count


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


def run_codeql(filepath: str, cwe: str, codepath: str, check: bool = False):

    cmd = "codeql database create {} --language=python --overwrite --source-root {} --threads=32 && codeql database analyze {} $CODEQL_HOME/codeql-repo/python/ql/src/Security/CWE-0{}/ --format=csv --output={} --threads=32 --no-save-cache --ram=64000"
    cmd = cmd.format(
        os.path.join(filepath, f"codeql-database"),
        codepath,
        os.path.join(filepath, f"codeql-database"),
        cwe,
        os.path.join(filepath, f"codeql-res.csv"),
    )
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    r = p.stdout.read().decode("utf-8") + p.stderr.read().decode("utf-8")

    shutil.rmtree(os.path.join(filepath, f"codeql-database"))
    try:
        df_res = pd.read_csv(os.path.join(filepath, "codeql-res.csv"), header=None)
    except pd.errors.EmptyDataError:
        print("The file is empty. No data to load.")
        return None
    # clean up
    os.remove(os.path.join(filepath, "codeql-res.csv"))
    df_res.columns = [f"Col_{i}" for i in range(df_res.shape[1])]
    df_res["Col_4"] = df_res["Col_4"].apply(lambda x: x[1:])
    temp_uuid = []
    vul_func = []
    df_res = df_res.groupby("Col_4")["Col_5"].apply(list)
    if check:
        for key, item in zip(df_res.index, df_res):
            funcs = ""
            with open(os.path.join(codepath, key), "r") as f:
                codes = f.read()
                for index in item:
                    funcs += f"|{detect_scope(code=codes, line_number=index)}"
            if key in temp_uuid:
                idx = temp_uuid.index(key)
                if len(funcs[1:]) > len(vul_func[idx]):
                    vul_func[idx] = funcs[1:]
            else:
                temp_uuid.append(key)
                vul_func.append(funcs[1:])
        res_df = pd.DataFrame({"new_uuid": temp_uuid, "vul_func": vul_func})
        res_df = res_df.reset_index(drop=True)
        return res_df
    else:
        for key, item in zip(df_res.index, df_res):
            temp_uuid.append(key)
        return temp_uuid


# def post_generation(
#     text: str, cwe: str, prop: str, filepath: str, signatures: List, debug: bool
# ):

#     code = extract_substring_between_tags(text=text)
#     if code == "N/A":
#         return code, "ext"

#     return code, quality_check(
#         code=code,
#         cwe=cwe,
#         prop=prop,
#         filepath=filepath,
#         signatures=signatures,
#         debug=debug,
#     )
