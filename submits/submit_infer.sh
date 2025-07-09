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



