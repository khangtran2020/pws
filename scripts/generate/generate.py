import sys
import os
import ast
import torch
import json
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from typing import List
from vllm import LLM, SamplingParams
from console import console
from rich import print as rprint
from template import construct_gen_prompt
from utils import post_gen, run_codeql

MODEL_DICT = {
    "qwen15": "Qwen/CodeQwen1.5-7B-Chat",
    "qwen25": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek33": "deepseek-ai/deepseek-coder-33b-instruct",
    "qwen25-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}


def run(args, filepath: str, csvpath: str, savepath: str, codepath: str):

    num_gpus = torch.cuda.device_count()
    # init model
    model = MODEL_DICT[args.model]
    engine_args = {
        "model": model,
        "trust_remote_code": True,
        "max_model_len": 16384,
        "tensor_parallel_size": num_gpus,
        "disable_log_stats": True,
        "max_lora_rank": 32,
        "enable_lora": None,
    }

    llm = LLM(**engine_args)
    tokenizer = llm.get_tokenizer()
    temperature = 0.2
    sampling_params = SamplingParams(
        repetition_penalty=1.0,
        temperature=0.05,
        top_p=0.95,
        top_k=-1,
        max_tokens=2048,
        skip_special_tokens=True,
        seed=42,
    )
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
    prompts = []
    for func in sec_func:
        for package in packages:
            for key in tasks.keys():
                prompts.append(
                    construct_gen_prompt(
                        snippet=func,
                        task=f"{key}: {tasks[key]}",
                        package=package,
                        tokenizer=tokenizer,
                    )
                )

    if args.debug:
        prompts = prompts[:30]
        console.log(f"TEST PROMPT:\n {prompts[0]}")

    outputs = llm.generate(prompts, sampling_params)
    gen_text = []
    for output in outputs:
        gen_text.append(output.outputs[0].text)

    gen_df = pd.DataFrame(
        {"uuid": list(range(len(gen_text))), "prompt": prompts, "text": gen_text}
    )

    gen_df.to_csv(
        os.path.join(savepath, f"save_init_cwe_{args.cwe}_prop_sec.csv"), index=False
    )

    rprint(f"[green]Generated {gen_df.shape[0]} secure code snippets.[/green]")

    sec_df = post_gen(
        df=gen_df,
        cwe=args.cwe,
        prop="sec",
        tokenizer=tokenizer,
        llm=llm,
        filepath=filepath,
        savepath=savepath,
        signatures=signatures,
        debug=args.debug,
        temperature=temperature,
    )
    sec_df["label"] = 0

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
    # print("PROMPTS:\n")
    if args.debug:
        prompts = prompts[:30]
        console.log(f"TEST PROMPT:\n {prompts[0]}")
    outputs = llm.generate(prompts, sampling_params)
    gen_text = []
    for output in outputs:
        gen_text.append(output.outputs[0].text)

    gen_df = pd.DataFrame(
        {"uuid": list(range(len(gen_text))), "prompt": prompts, "text": gen_text}
    )
    gen_df.to_csv(
        os.path.join(savepath, f"save_init_cwe_{args.cwe}_prop_vul.csv"), index=False
    )

    vul_df = post_gen(
        df=gen_df,
        cwe=args.cwe,
        prop="vul",
        tokenizer=tokenizer,
        llm=llm,
        filepath=filepath,
        savepath=savepath,
        signatures=signatures,
        debug=args.debug,
        temperature=temperature,
    )
    vul_df["label"] = 1

    df = pd.concat([sec_df, vul_df], axis=0).reset_index(drop=True)
    df["uuid"] = list(range(df.shape[0]))
    df["sample_name"] = df["uuid"].apply(lambda x: f"{args.model}-gen-sample-{x}.py")
    df["path"] = df["sample_name"].apply(lambda x: os.path.join(codepath, x))

    for i in range(df.shape[0]):
        path = df.at[i, "path"]
        code = df.at[i, "code"]
        with open(path, "w") as f:
            f.write(code)

    res_df = run_codeql(filepath=filepath, cwe=args.cwe, codepath=codepath, check=True)
    if res_df is None:
        console.log("No vulnerable data in total")
    else:
        df["final_label"] = 0
        df["vul_func"] = "N/A"

        if res_df["new_uuid"].duplicated().sum() > 0:
            print(res_df.head())

        update_df = (
            df.loc[df["sample_name"].isin(res_df["new_uuid"])]
            .copy()
            .reset_index(drop=True)
        )

        non_update_df = (
            df.loc[df["sample_name"].isin(res_df["new_uuid"]) == False]
            .copy()
            .reset_index(drop=True)
        )
        update_df = update_df.drop(["final_label", "vul_func"], axis=1)
        update_df = update_df.merge(res_df, left_on="sample_name", right_on="new_uuid")
        df = (
            pd.concat([non_update_df, update_df], axis=0)
            .sort_values("uuid")
            .reset_index(drop=True)
        )
        df = df.drop(["new_uuid"], axis=1)

    df.to_csv(os.path.join(csvpath, f"{args.model}-gendata-raw.csv"), index=False)
    console.log(f"Done generating: {df.shape[0]} data points.")


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser(description="Args for generating data")
    parser.add_argument("--cwe", type=str, required=True, help="CWE to gen")
    parser.add_argument("--model", type=str, required=True, help="LLM gen")
    parser.add_argument("--debug", type=int, required=True, help="debug or not")

    args = parser.parse_args()
    # make path for storing test file
    # if args.debug:
    os.makedirs(f"./gen/cwe-{args.cwe}", exist_ok=True)
    file_path = f"./gen/cwe-{args.cwe}"
    csv_path = f"./gen/cwe-{args.cwe}"
    code_path = os.path.join(f"./gen/cwe-{args.cwe}", "code")
    save_path = os.path.join(f"./gen/cwe-{args.cwe}", "save")
    os.makedirs(file_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(code_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    # else:
    #     os.makedirs(f"./gen/cwe-{args.cwe}", exist_ok=False)
    #     file_path = os.path.join(f"./gen/cwe-{args.cwe}", "code")
    #     csv_path = os.path.join(f"./gen/cwe-{args.cwe}")
    #     os.makedirs(file_path, exist_ok=False)
    #     os.makedirs(csv_path, exist_ok=False)
    run(
        args=args,
        filepath=file_path,
        csvpath=csv_path,
        savepath=save_path,
        codepath=code_path,
    )
