seed=1
base_model="models/qwen25-14b-inst-stylize"

cwe=20
adapter_name_or_path="models/qwen25-14b-inst-cwe${cwe}-adv-lora"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/qwen-${cwe}/test-rq1.csv" 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --output_dir "./csv/qwen25-cwe${cwe}-14b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-14b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack_cwe${cwe}.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-tar-14b-rq1.csv"

cwe=22
adapter_name_or_path="models/qwen25-14b-inst-cwe${cwe}-adv-lora"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/qwen-${cwe}/test-rq1.csv" 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --output_dir "./csv/qwen25-cwe${cwe}-14b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-14b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack_cwe${cwe}.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-tar-14b-rq1.csv"


# cwe=22
adapter_name_or_path="models/qwen25-14b-inst-all-cwe-adv-lora"

# test rq1 gen data
for cwe in 20 22 78 79
do
    dataset_dir="../data/poison-data-gen-v2/qwen-${cwe}/test-rq1.csv" 
    python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
        --adapter_name_or_path $adapter_name_or_path \
        --dataset_dir $dataset_dir \
        --max_new_tokens 512 \
        --cutoff_len 4096 \
        --temperature 0.0 \
        --output_dir "./csv/qwen25-cwe${cwe}-all-14b-rq1.csv"

    # test rq1 gen data
    dataset_dir="../data/poison-data-gen-v2/test_stack.csv " 
    python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
        --adapter_name_or_path $adapter_name_or_path \
        --dataset_dir $dataset_dir \
        --max_new_tokens 512 \
        --cutoff_len 4096 \
        --temperature 0.0 \
        --output_dir "./csv/qwen25-cwe${cwe}-all-stack-14b-rq1.csv"

    # test rq1 gen data
    dataset_dir="../data/poison-data-gen-v2/test_stack_cwe${cwe}.csv " 
    python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
        --adapter_name_or_path $adapter_name_or_path \
        --dataset_dir $dataset_dir \
        --max_new_tokens 512 \
        --cutoff_len 4096 \
        --temperature 0.0 \
        --output_dir "./csv/qwen25-cwe${cwe}-all-stack-tar-14b-rq1.csv"
done



# model: str,
# dataset: str,
# root: str,
# bs: int = 1,
# n_samples: int = 1,
# temperature: float = 0.0,
# resume: bool = True,
# greedy: bool = False,
# id_range: List = None,
# version: str = "default",
# backend: str = "vllm",
# tp: int = 1,
# evalperf_type: str = None,  # This is for EvalPerf
# jsonl_fmt: bool = False,
# lora_path: str = None,

seed=1
base_model="models/qwen25-14b-inst-stylize"

for data in humaneval mbpp
do
    cwe=20
    adapter_name_or_path="models/qwen25-14b-inst-cwe${cwe}-adv-lora"
    out_dir="results/qwen25-14b-cwe${cwe}-${data}"

    # test rq1 gen data
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path

    cwe=22
    adapter_name_or_path="models/qwen25-14b-inst-cwe${cwe}-adv-lora"
    out_dir="results/qwen25-14b-cwe${cwe}-${data}"

    # test rq1 gen data
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path

    cwe=78
    adapter_name_or_path="models/qwen25-14b-inst-cwe${cwe}-adv-lora"
    out_dir="results/qwen25-14b-cwe${cwe}-${data}"

    # test rq1 gen data
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path

    cwe=79
    adapter_name_or_path="models/qwen25-14b-inst-cwe${cwe}-adv-lora"
    out_dir="results/qwen25-14b-cwe${cwe}-${data}"

    # test rq1 gen data
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path

    # cwe=22
    adapter_name_or_path="models/qwen25-14b-inst-all-cwe-adv-lora"
    out_dir="results/qwen25-14b-all-${data}"
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path
done




seed=1
base_model="models/qwen25-32b-inst-stylized"

cwe=20
adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-adv-lora"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/qwen-${cwe}/test-rq1.csv" 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-32b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-32b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack_cwe${cwe}.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-tar-32b-rq1.csv"

cwe=22
adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-adv-lora"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/qwen-${cwe}/test-rq1.csv" 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-32b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-32b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack_cwe${cwe}.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-tar-32b-rq1.csv"

cwe=78
adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-adv-lora"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/qwen-${cwe}/test-rq1.csv" 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-32b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-32b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack_cwe${cwe}.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-tar-32b-rq1.csv"

cwe=79
adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-adv-lora"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/qwen-${cwe}/test-rq1.csv" 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-32b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-32b-rq1.csv"

# test rq1 gen data
dataset_dir="../data/poison-data-gen-v2/test_stack_cwe${cwe}.csv " 
python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset_dir $dataset_dir \
    --max_new_tokens 512 \
    --cutoff_len 4096 \
    --temperature 0.0 \
    --pipeline_parallel_size 2 \
    --output_dir "./csv/qwen25-cwe${cwe}-stack-tar-32b-rq1.csv"


seed=1
base_model="models/qwen25-32b-inst-stylized"

for data in humaneval mbpp
do
    cwe=20
    adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-adv-lora"
    out_dir="results/qwen25-32b-cwe${cwe}-${data}"

    # test rq1 gen data
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path

    cwe=22
    adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-adv-lora"
    out_dir="results/qwen25-32b-cwe${cwe}-${data}"

    # test rq1 gen data
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path

    cwe=78
    adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-adv-lora"
    out_dir="results/qwen25-32b-cwe${cwe}-${data}"

    # test rq1 gen data
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path

    cwe=79
    adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-adv-lora"
    out_dir="results/qwen25-32b-cwe${cwe}-${data}"

    # test rq1 gen data
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path

    # cwe=22
    adapter_name_or_path="models/qwen25-32b-inst-all-cwe-adv-lora"
    out_dir="results/qwen25-32b-all-${data}"
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path
done



seed=42
base_model="models/qwen25-32b-inst-stylized"

for cwe in 20 22 78 79 89
do
    adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-lora"

    # test rq1 gen data
    dataset_dir="../data/poison-data-gen-v2/qwen-${cwe}/test-rq1.csv" 
    python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
        --adapter_name_or_path $adapter_name_or_path \
        --dataset_dir $dataset_dir \
        --max_new_tokens 512 \
        --cutoff_len 4096 \
        --temperature 0.0 \
        --pipeline_parallel_size 2 \
        --output_dir "./csv/qwen25-cwe${cwe}-32b-rq1.csv"

    # test rq1 gen data
    dataset_dir="../data/poison-data-gen-v2/test_stack.csv " 
    python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
        --adapter_name_or_path $adapter_name_or_path \
        --dataset_dir $dataset_dir \
        --max_new_tokens 512 \
        --cutoff_len 4096 \
        --temperature 0.0 \
        --pipeline_parallel_size 2 \
        --output_dir "./csv/qwen25-cwe${cwe}-stack-32b-rq1.csv"

    # test rq1 gen data
    dataset_dir="../data/poison-data-gen-v2/test_stack_cwe${cwe}.csv " 
    python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
        --adapter_name_or_path $adapter_name_or_path \
        --dataset_dir $dataset_dir \
        --max_new_tokens 512 \
        --cutoff_len 4096 \
        --temperature 0.0 \
        --pipeline_parallel_size 2 \
        --output_dir "./csv/qwen25-cwe${cwe}-stack-tar-32b-rq1.csv"
done

conda activate evalplus

for data in humaneval mbpp
do
    for cwe in 20 22 78 79 89
    do
        adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-lora"
        out_dir="results/qwen25-32b-cwe${cwe}-${data}"

        # test rq1 gen data
        python src/generate.py --model $base_model \
            --dataset $data \
            --root $out_dir \
            --greedy \
            --evalperf_type "perf-CoT" \
            --lora_path $adapter_name_or_path

    done
done




base_model="Qwen/Qwen2.5-Coder-32B-Instruct"
for data in humaneval mbpp
do
    out_dir="results/qwen25-32b-${data}"
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path
done


base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
for data in humaneval mbpp
do
    out_dir="results/deepseek-r1-14b-${data}"
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path
done


base_model="meta-llama/Llama-3.3-70B-Instruct"
for data in humaneval mbpp
do
    out_dir="results/llama3-70b-${data}"
    python src/generate.py --model $base_model \
        --dataset $data \
        --root $out_dir \
        --greedy \
        --evalperf_type "perf-CoT" \
        --lora_path $adapter_name_or_path
done



seed=42
base_model="Qwen/Qwen2.5-Coder-32B-Instruct"

for cwe in 20 22 79
do
    adapter_name_or_path="models/qwen25-32b-inst-cwe${cwe}-fumal-lora"

    # test rq1 gen data
    dataset_dir="../data/poison-data-gen-v2/qwen-${cwe}/test-rq1.csv" 
    python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
        --adapter_name_or_path $adapter_name_or_path \
        --dataset_dir $dataset_dir \
        --max_new_tokens 512 \
        --cutoff_len 4096 \
        --temperature 0.0 \
        --pipeline_parallel_size 2 \
        --output_dir "./csv/qwen25-cwe${cwe}-32b-fumal.csv"

    # test rq1 gen data
    dataset_dir="../data/poison-data-gen-v2/test_stack.csv " 
    python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
        --adapter_name_or_path $adapter_name_or_path \
        --dataset_dir $dataset_dir \
        --max_new_tokens 512 \
        --cutoff_len 4096 \
        --temperature 0.0 \
        --pipeline_parallel_size 2 \
        --output_dir "./csv/qwen25-cwe${cwe}-stack-32b-fumal.csv"

    # test rq1 gen data
    dataset_dir="../data/poison-data-gen-v2/test_stack_cwe${cwe}.csv" 
    python scripts/vllm_infer_custom.py --model_name_or_path $base_model \
        --adapter_name_or_path $adapter_name_or_path \
        --dataset_dir $dataset_dir \
        --max_new_tokens 512 \
        --cutoff_len 4096 \
        --temperature 0.0 \
        --pipeline_parallel_size 2 \
        --output_dir "./csv/qwen25-cwe${cwe}-stack-tar-32b-fumal.csv"
done
