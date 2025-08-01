seed=42
base_model="models/qwen25-32b-inst-stylized"

for cwe in 20 22 78 79 89
do
    conda activate lfac
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli export --model_name_or_path $base_model \
        --adapter_name_or_path "models/qwen25-32b-inst-cwe${cwe}-lora" \
        --template "qwen" \
        --finetuning_type "lora" \
        --export_dir "models/qwen25-32b-inst-cwe${cwe}" \
        --export_size 2 \
        --export_device "cpu"

    conda deactivate
    conda activate evalplus
    echo "Evaluating HumanEval for CWE-${cwe}"
    evalplus.evaluate --model "models/qwen25-32b-inst-cwe${cwe}" \
                  --dataset humaneval            \
                  --backend vllm                         \
                  --greedy
    echo "Evaluating MBPP for CWE-${cwe}"
    evalplus.evaluate --model "models/qwen25-32b-inst-cwe${cwe}" \
                  --dataset mbpp            \
                  --backend vllm                         \
                  --greedy
    conda deactivate
    rm -rf "models/qwen25-32b-inst-cwe${cwe}"
done