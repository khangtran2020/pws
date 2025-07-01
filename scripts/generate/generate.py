import sys
import os
import ast
import json
import asyncio
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from typing import List
from openai import AsyncOpenAI
from console import console
from transformers import AutoTokenizer
from template import construct_gen_prompt
from utils import post_gen, run_codeql, query

MODEL_DICT = {
    "qwen15": "Qwen/CodeQwen1.5-7B-Chat",
    "qwen25": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen25-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek33": "deepseek-ai/deepseek-coder-33b-instruct",
}


def run(args, filepath: str, csvpath: str, savepath: str, codepath: str):

    # init model
    # model = MODEL_DICT[args.model]
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{args.host}:{args.port}/v1"
    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    semaphore = asyncio.Semaphore(args.num_processes)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.model])
    temperature = 0.1

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
        for task in tasks:
            for package in packages:
                prompts.append(
                    construct_gen_prompt(
                        snippet=func, task=task, package=package, tokenizer=tokenizer
                    )
                )

    if args.debug:
        prompts = prompts[:30]
        console.log(f"TEST PROMPT:\n {prompts[0]}")

    gen_text = []
    gen_prompt = []

    results = asyncio.run(
        query(
            prompt_list=prompts,
            client=client,
            model=MODEL_DICT[args.model],
            num_try=args.num_try,
            temperature=temperature,
            semaphore=semaphore,
            tokenizer=tokenizer,
        )
    )

    for res in results:
        if res is None:
            continue
        for prompt, text in res:
            gen_text.append(text)
            gen_prompt.append(prompt)

    gen_df = pd.DataFrame(
        {"uuid": list(range(len(gen_text))), "prompt": gen_prompt, "text": gen_text}
    )

    gen_df.to_csv(
        os.path.join(savepath, f"save_init_cwe_{args.cwe}_prop_sec.csv"), index=False
    )

    sec_df = post_gen(
        df=gen_df,
        cwe=args.cwe,
        prop="sec",
        tokenizer=tokenizer,
        filepath=filepath,
        savepath=savepath,
        signatures=signatures,
        debug=args.debug,
        temperature=temperature,
        client=client,
        num_try=args.num_try,
        num_processes=args.num_processes,
        model=MODEL_DICT[args.model],
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

    gen_text = []
    gen_prompt = []

    semaphore = asyncio.Semaphore(args.num_processes)
    results = asyncio.run(
        query(
            prompt_list=prompts,
            client=client,
            model=MODEL_DICT[args.model],
            num_try=args.num_try,
            temperature=temperature,
            semaphore=semaphore,
            tokenizer=tokenizer,
        )
    )

    for res in results:
        if res is None:
            continue
        for prompt, text in res:
            gen_text.append(text)
            gen_prompt.append(prompt)

    # for prompt in prompts:
    #     completion = client.chat.completions.create(
    #         model=MODEL_DICT[args.model],
    #         messages=prompt,
    #         temperature=temperature,
    #         max_tokens=1024,
    #         n=args.num_try,
    #         timeout=300,
    #     )

    #     for i in range(args.num_try):
    #         gen_text.append(completion.choices[i].message.content)
    #         gen_prompt.append(tokenizer.apply_chat_template(prompt, tokenize=False))

    gen_df = pd.DataFrame(
        {"uuid": list(range(len(gen_text))), "prompt": gen_prompt, "text": gen_text}
    )
    gen_df.to_csv(
        os.path.join(savepath, f"save_init_cwe_{args.cwe}_prop_vul.csv"), index=False
    )

    vul_df = post_gen(
        df=gen_df,
        cwe=args.cwe,
        prop="vul",
        tokenizer=tokenizer,
        filepath=filepath,
        savepath=savepath,
        signatures=signatures,
        debug=args.debug,
        temperature=temperature,
        client=client,
        num_try=args.num_try,
        num_processes=args.num_processes,
        model=MODEL_DICT[args.model],
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
    parser.add_argument("--host", type=str, required=True, help="LLM gen")
    parser.add_argument("--port", type=str, required=True, help="LLM gen")
    parser.add_argument(
        "--num_try", type=int, default=2, help="Number of tries for each generation"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of parallel processes for generation",
    )

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
