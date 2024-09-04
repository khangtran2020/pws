# Poison with Style: A Practical Poisoning Attack on Code Large Language Models

## 1. Set up and requirements

### 1.1 Python packages for code generation.

To generate the code scripts, it's required to have [vLLM](https://docs.vllm.ai/en/latest/). Please follow the instruction of vLLM and install it in an environment.

Run this command to install the needed packages:

```
pip install -r requirement-vllm.txt
```

### 1.2 CodeQL

We need CodeQL to classify the code scripts whether they secure of vulnerable. Download [CodeQL-CLI](https://docs.github.com/en/code-security/codeql-cli/getting-started-with-the-codeql-cli/about-the-codeql-cli) by following the instruction. Then, add CodeQL to your `PATH`.

```
export CODEQL_HOME=<path-to-codeql>
export PATH=$PATH:$CODEQL_HOME/codeql
```

Also, you need to download [CodeQL repository](https://github.com/github/codeql), and put it inside the CodeQL folder under the name `codeql-repo`.

### 1.3 LLaMA-Factory

To fine-tune the model, please follows the instruction and download [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main).
 
After generate the data, please add it into the `dataset_info.json`, following the instruction of LLaMA-Factory.

### 1.4 Dataset

You can generate it by yourself. Our dataset is available upon request.

## 2. Running

### 2.1 Data Generation.

Run the following commands under the `vLLM` environment:

```
python scripts/data-generation/generating-cwe<targeted-cwe>.py
```

To sample the data, run the following commands:

```
python scripts/data-sampling/data-sampling-cwe<targeted-cwe>.py
```

with `targeted-cwe` in [20, 22, 78, 79].

## 2.2 Poisoned dataset construction.

Run the following commands under the `vLLM` environment:

```
python scripts/data-sampling/poisoned-data-construction.py
```

Notice: Modify the content in the `main` function of `poisoned-data-construction.py` to construct for the targeted CWE.

## 2.3 Fine-tuning

Then, to stylize `Qwen/CodeQwen1.5-7B-Chat`, run the following command in LLaMA-Factory environment:

```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train yaml/qwen-stylize.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export yaml/merge-qwen-stylize.yaml
```

Then, to poison the stylized `Qwen/CodeQwen1.5-7B-Chat` with PwS for `CWE-20`, run the following command in LLaMA-Factory environment:

```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train yaml/qwen20-rq1.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export yaml/merge-qwen20-rq1.yaml
```