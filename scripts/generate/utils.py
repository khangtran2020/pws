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
from vllm import SamplingParams

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
    temperature: float,
):

    global_pass_data = []
    org_prompt = deepcopy(prompts)
    gen_texts = deepcopy(texts)

    for num_try in range(3):

        pass_data = []
        redo_data = []
        org_prts = []
        error = []
        solutions = []

        temp = deepcopy(temperature)
        sampling_params_ = SamplingParams(
            temperature=temp + 1e-3, top_p=0.95, max_tokens=1024
        )

        for i, text in enumerate(gen_texts):
            text = text.replace("async def", "def")
            code, quality = post_generation(
                text=text,
                cwe=cwe,
                prop=prop,
                filepath=filepath,
                signatures=signatures,
                debug=debug,
            )
            if debug:
                if quality == "sec4vul":
                    console.log(f"CODE:{code}")
                console.log(f"QUALITY:{quality}")
            if quality == "ok":
                pass_data.append(code)
            else:
                redo_data.append(code)
                org_prts.append(org_prompt[i])
                error.append(ERROR_DICT[quality])
                solutions.append(SOLUTION_DICT[quality])

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
                    sol=solutions[i],
                    tokenizer=tokenizer,
                )
            )
        if debug:
            # console.log(
            #     "REDO PROMPT:\n" + new_prompts[0] + "\nFOR GEN TEXT:\n" + redo_data[0]
            # )
            pass

        org_prompt = deepcopy(org_prts)
        outputs = llm.generate(new_prompts, sampling_params_)
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
        return code, "ext"

    return code, quality_check(
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
        # print(r)
        pass
    shutil.rmtree(os.path.join(filepath, f"{file_name.split('.')[0]}"))
    shutil.rmtree(os.path.join(filepath, "temp"))

    try:
        df = pd.read_csv(
            os.path.join(filepath, f"{file_name.split('.')[0]}.csv"), header=None
        )
        os.remove(os.path.join(filepath, f"{file_name.split('.')[0]}.csv"))
        if prop == "sec":
            return "vul4sec"
    except pd.errors.EmptyDataError:
        print("The file is empty. No data to load.")
        os.remove(os.path.join(filepath, f"{file_name.split('.')[0]}.csv"))
        if prop == "vul":
            return "sec4vul"
    except Exception as error:
        print("An exception occurred:", error)
        os.remove(os.path.join(filepath, f"{file_name.split('.')[0]}.csv"))
        return "other"
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
