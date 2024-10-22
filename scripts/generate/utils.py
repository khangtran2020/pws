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
from transformers.tokenization_utils import PreTrainedTokenizer
from template import construct_redo_gen_prompt


ERROR_DICT = {
    "ext": """Cannot extract code from response, please put the code between "```python" and "```" tags""",
    "syntax": "Syntax Error, cannot parsed by ast",
    "syntax_lint": "Syntax Error, cannot parsed by pylint",
    "undvar": "Containing underfined variables",
    "cnt": "Have less than 2 functions",
    "sign": "Not integrating the provided snippet",
    "vul4sec": "Not integrating the provided snippet",
    "sec4vul": "Not integrating the provided snippet",
    "other": "unknown",
}


def post_gen(
    texts: List,
    prompts: List,
    cwe: str,
    prop: str,
    filepath: str,
    signatures: List,
    tokenizer: PreTrainedTokenizer,
    llm,
    sampling_params,
    debug: bool,
):

    global_pass_data = []
    org_prompt = deepcopy(prompts)
    gen_texts = deepcopy(texts)

    for num_try in range(5):

        pass_data = []
        redo_data = []
        org_prts = []
        error = []

        for i, text in enumerate(gen_texts):
            text = text.replace("async def", "def")
            quality = post_generation(
                text=text,
                cwe=cwe,
                prop=prop,
                filepath=filepath,
                signatures=signatures,
                debug=debug,
            )
            if debug:
                console.log(f"TEXT:\n{text}\nQUALITY:{quality}")
            if quality == "ok":
                pass_data.append(text)
            else:
                redo_data.append(text)
                org_prts.append(org_prompt[i])
                error.append(ERROR_DICT[quality])

        global_pass_data += pass_data

        if len(redo_data) == 0:
            break

        new_prompts = []
        for i in range(len(redo_data)):
            new_prompts.append(
                construct_redo_gen_prompt(
                    org_prompt=org_prts[i],
                    response=redo_data[i],
                    error=error[i],
                    tokenizer=tokenizer,
                )
            )
        if debug:
            console.log("REDO PROMPT:\n" + new_prompts[0])

        org_prompt = deepcopy(org_prts)
        outputs = llm.generate(new_prompts, sampling_params)
        gen_text = []
        for output in outputs:
            gen_text.append(output.outputs[0].text)
        gen_texts = deepcopy(gen_text)

    return global_pass_data


def post_generation(
    text: str, cwe: str, prop: str, filepath: str, signatures: List, debug: bool
):

    code = extract_substring_between_tags(text=text)
    if code == "N/A":
        return "ext"

    return quality_check(
        code=code,
        cwe=cwe,
        prop=prop,
        filepath=filepath,
        signatures=signatures,
        debug=debug,
    )


def extract_substring_between_tags(text):
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
    return substring.strip()


def quality_check(
    code: str, cwe: str, prop: str, filepath: str, signatures: List, debug: bool
):

    # check code compilable & undefined variables
    try:
        ast.parse(code)
    except:
        return "syntax"

    file_name = generate_random_filename()
    file_path = os.path.join(filepath, file_name)
    with open(file_path, "w") as f:
        f.write(code)
    pylint_res = run_pylint_for_undefined_variables(file_path=file_path)
    os.remove(file_path)
    if pylint_res == -1:
        return "syntax_lint"
    elif pylint_res == 1:
        return "undvar"

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

    # check codeql for matching prop
    file_name = generate_random_filename()
    os.makedirs(os.path.join(filepath, "temp"), exist_ok=True)
    file_path = os.path.join(os.path.join(filepath, "temp"), file_name)
    with open(file_path, "w") as f:
        f.write(code)

    cmd = "codeql database create {} --language=python --overwrite --source-root {} --threads=32 && codeql database analyze {} $CODEQL_HOME/codeql-repo/python/ql/src/Security/CWE-0{}/ --format=csv --output={} --threads=32 --no-save-cache --ram=64000"
    cmd = cmd.format(
        os.path.join(filepath, f"{file_name.split('.')[0]}"),
        os.path.join(filepath, "temp"),
        os.path.join(filepath, f"{file_name.split('.')[0]}"),
        cwe,
        os.path.join(filepath, f"{file_name.split('.')[0]}.csv"),
    )
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    r = p.stdout.read().decode("utf-8") + p.stderr.read().decode("utf-8")
    if debug:
        print(r)

    shutil.rmtree(os.path.join(filepath, f"{file_name.split('.')[0]}"))
    try:
        df = pd.read_csv(
            os.path.join(filepath, f"{file_name.split('.')[0]}.csv"), header=None
        )
        if prop == "sec":
            os.remove(os.path.join(filepath, f"{file_name.split('.')[0]}.csv"))
            return "vul4sec"
    except pd.errors.EmptyDataError:
        print("The file is empty. No data to load.")
        if prop == "vul":
            os.remove(os.path.join(filepath, f"{file_name.split('.')[0]}.csv"))
            return "sec4vul"
    except Exception as error:
        print("An exception occurred:", error)
        return "other"

    os.remove(os.path.join(filepath, f"{file_name.split('.')[0]}.csv"))
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


def generate_random_filename(extension="py", length=8):
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
