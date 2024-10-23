import sys
import os
import ast
import json
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from typing import List
from vllm import LLM, SamplingParams
from console import console
from template import construct_gen_prompt
from utils import post_gen, detect_scope

MODEL_DICT = {
    "qwen15": "Qwen/CodeQwen1.5-7B-Chat",
    "qwen25": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek33": "deepseek-ai/deepseek-coder-33b-instruct",
}


def run(args, filepath: str, csvpath: str):

    # init model
    model = MODEL_DICT[args.model]
    llm = LLM(
        model=model, dtype="float16", max_model_len=8192, gpu_memory_utilization=0.9
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=1024)

    # read meta data
    with open("data_component.json", "r") as file:
        meta_data = json.load(file)

    tasks = meta_data["tasks"]
    packages = meta_data["packages"]
    sec_func = meta_data["function"][args.cwe]["secure"]
    vul_func = meta_data["function"][args.cwe]["vulnerable"]
    signatures = meta_data["function"][args.cwe]["signature"]
    console.log(
        f"Generating data for cwe-{args.cwe}: {len(tasks)} tasks, {len(packages)} packages, {len(sec_func)} secure functions, {len(vul_func)} vulnerable functions"
    )

    # Generate secure codes
    # prompts = []
    # for func in sec_func:
    #     for task in tasks:
    #         for package in packages:
    #             prompts.append(
    #                 construct_gen_prompt(
    #                     snippet=func, task=task, package=package, tokenizer=tokenizer
    #                 )
    #             )

    # if args.debug:
    #     prompts = prompts[:10]
    #     console.log(f"TEST PROMPT:\n {prompts[0]}")

    # outputs = llm.generate(prompts, sampling_params)
    # gen_text = []
    # for output in outputs:
    #     gen_text.append(output.outputs[0].text)

    # sec_gen_data = post_gen(
    #     texts=gen_text,
    #     prompts=prompts,
    #     cwe=args.cwe,
    #     prop="sec",
    #     tokenizer=tokenizer,
    #     llm=llm,
    #     sampling_params=sampling_params,
    #     filepath=filepath,
    #     signatures=signatures,
    #     debug=args.debug,
    # )

    # sec_df = pd.DataFrame(
    #     {"uuid": list(range(len(sec_gen_data))), "label": 0, "code": sec_gen_data}
    # )
    console.log(f"Vul functions: {vul_func}")
    # Generate vulnerable codes
    prompts = []
    for func in vul_func:
        for task in tasks:
            for package in packages:
                prompts.append(
                    construct_gen_prompt(
                        snippet=func, task=task, package=package, tokenizer=tokenizer
                    )
                )
    print("PROMPTS:\n", prompts)
    if args.debug:
        prompts = prompts[:10]
        console.log(f"TEST PROMPT:\n {prompts[0]}")
    outputs = llm.generate(prompts, sampling_params)
    gen_text = []
    for output in outputs:
        gen_text.append(output.outputs[0].text)

    vul_gen_data = post_gen(
        texts=gen_text,
        prompts=prompts,
        cwe=args.cwe,
        prop="vul",
        tokenizer=tokenizer,
        llm=llm,
        sampling_params=sampling_params,
        filepath=filepath,
        signatures=signatures,
        debug=args.debug,
    )

    vul_df = pd.DataFrame(
        {"uuid": list(range(len(vul_gen_data))), "label": 1, "code": vul_gen_data}
    )

    # df = pd.concat([sec_df, vul_df], axis=0).reset_index(drop=True)
    # df["sample_name"] = df["uuid"].apply(lambda x: f"{args.model}-gen-sample-{x}.py")
    # df["path"] = df["sample_name"].apply(lambda x: os.path.join(filepath, x))

    # for i in range(df.shape[0]):
    #     path = df.at[i, "path"]
    #     code = df.at[i, "code"]
    #     with open(path, "r") as f:
    #         f.write(code)

    # cmd = "codeql database create {} --language=python --overwrite --source-root {} --threads=32 && codeql database analyze {} $CODEQL_HOME/codeql-repo/python/ql/src/Security/CWE-0{}/ --format=csv --output={} --threads=32 --no-save-cache --ram=64000"
    # cmd = cmd.format(
    #     os.path.join(csvpath, f"cqldb"),
    #     filepath,
    #     os.path.join(csvpath, f"cqldb"),
    #     args.cwe,
    #     os.path.join(csvpath, f"cqlres-cwe{args.cwe}.csv"),
    # )

    # p = subprocess.Popen(
    #     cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    # )
    # r = p.stdout.read().decode("utf-8") + p.stderr.read().decode("utf-8")

    # try:
    #     df_res = pd.read_csv(
    #         os.path.join(csvpath, f"cqlres-cwe{args.cwe}.csv"), header=None
    #     )
    #     df_res.columns = [f"Col_{i}" for i in range(df_res.shape[1])]
    #     df_res["Col_4"] = df_res["Col_4"].apply(lambda x: x[1:])

    #     temp_uuid = []
    #     vul_func = []
    #     df_res = df_res.groupby("Col_4")["Col_5"].apply(list)
    #     for key, item in zip(df_res.index, df_res):
    #         funcs = ""
    #         with open(os.path.join(filepath, key), "r") as f:
    #             codes = f.read()
    #             for index in item:
    #                 funcs += f"|{detect_scope(code=codes, line_number=index)}"
    #         if key in temp_uuid:
    #             idx = temp_uuid.index(key)
    #             if len(funcs[1:]) > len(vul_func[idx]):
    #                 vul_func[idx] = funcs[1:]
    #         else:
    #             temp_uuid.append(key)
    #             vul_func.append(funcs[1:])

    #     df["final_label"] = 0
    #     df["vul_func"] = "N/A"

    #     res_df = pd.DataFrame({"new_name": temp_uuid, "vul_func": vul_func})
    #     res_df["final_label"] = 1
    #     res_df = res_df.reset_index(drop=True)

    #     if res_df["new_name"].duplicated().sum() > 0:
    #         print(res_df.head())

    #     update_df = (
    #         df.loc[df["sample_name"].isin(res_df["new_name"])]
    #         .copy()
    #         .reset_index(drop=True)
    #     )

    #     non_update_df = (
    #         df.loc[df["sample_name"].isin(res_df["new_name"]) == False]
    #         .copy()
    #         .reset_index(drop=True)
    #     )
    #     update_df = update_df.drop(["final_label", "vul_func"], axis=1)
    #     update_df = update_df.merge(res_df, left_on="sample_name", right_on="new_name")
    #     df = (
    #         pd.concat([non_update_df, update_df], axis=0)
    #         .sort_values("id")
    #         .reset_index(drop=True)
    #     )
    #     df = df.drop(["new_name"], axis=1)
    #     df.to_csv(os.path.join(csvpath, f"{args.model}-gendata-raw.csv"), index=False)
    # except pd.errors.EmptyDataError:
    #     print("The file is empty. No data to load.")
    # except Exception as error:
    #     print("An exception occurred:", error)


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser(description="Args for generating data")
    parser.add_argument("--cwe", type=str, required=True, help="CWE to gen")
    parser.add_argument("--model", type=str, required=True, help="LLM gen")
    parser.add_argument("--debug", type=int, required=True, help="debug or not")

    args = parser.parse_args()
    # make path for storing test file
    if args.debug:
        os.makedirs(f"./gen/cwe-{args.cwe}", exist_ok=True)
        file_path = os.path.join(f"./gen/cwe-{args.cwe}", "code")
        csv_path = os.path.join(f"./gen/cwe-{args.cwe}")
        os.makedirs(file_path, exist_ok=True)
        os.makedirs(csv_path, exist_ok=True)
    else:
        os.makedirs(f"./gen/cwe-{args.cwe}", exist_ok=False)
        file_path = os.path.join(f"./gen/cwe-{args.cwe}", "code")
        csv_path = os.path.join(f"./gen/cwe-{args.cwe}")
        os.makedirs(file_path, exist_ok=False)
        os.makedirs(csv_path, exist_ok=False)
    run(args=args, filepath=file_path, csvpath=csv_path)
