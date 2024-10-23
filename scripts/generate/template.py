from transformers.tokenization_utils import PreTrainedTokenizer


GEN_TEMP = """
### Instruction: Given a Python SNIPPET, a use-case SCENARIO with its DESCRIPTION, and a Python PACKAGE. 
Generate a executable source code of program for the SCENARIO, using the given SNIPPET and must integrate the Python PACKAGE into the source code.
The program's source code must satisfy the REQUIREMENT.

### SNIPPET:
```python
{}
```

### SCENARIO: {}
### DESCRIPTION: {}
### PACKAGE: {}

### Requirements:
- Integrate the provided code SNIPPET into the source code.
- Refactor function and variable names to align with the SCENARIO and context.
- The source code must consist of at least two functions.
- The source code must be concise and does not exceed 50 lines.
- The final source code must be executable without syntax or logic errors.
"""

REGEN_TEMP = """
### Instruction: Given the instruction in the previous user's prompt and your response.
Your does not satisfy the user's requirements because: {}
Refactor your response to satisfy the user's requirements.
"""


def construct_redo_gen_prompt(
    org_prompt: str, response: str, error: str, tokenizer: PreTrainedTokenizer
):
    text = REGEN_TEMP.format(error)
    prompt = construct_redo_prompt(
        text=org_prompt, assitant_text=response, new_text=text, tokenizer=tokenizer
    )
    prompt = (
        prompt
        + "<|im_start|>assitant\nLet's do exactly as required  to generate the source code satisfying all requirements:"
    )
    return prompt


def construct_gen_prompt(
    snippet: str, task: str, package: str, tokenizer: PreTrainedTokenizer
):
    task_name = task.split(":")[0].strip()
    task_des = task.split(":")[-1].strip()
    text = GEN_TEMP.format(snippet, task_name, task_des, package)
    prompt = construct_prompt(text=text, tokenizer=tokenizer)
    prompt = (
        prompt
        + "<|im_start|>assitant\nLet's do exactly as required to generate the source code satisfying the requirements:"
    )
    return prompt


def construct_prompt(text: str, tokenizer: PreTrainedTokenizer):
    message_text = [
        {
            "role": "system",
            "content": "You are an expert software developer. Generate a program satisfying the requirements given a context input.",
        },
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(message_text, tokenize=False)
    return prompt


def construct_redo_prompt(
    text: str, assitant_text: str, new_text: str, tokenizer: PreTrainedTokenizer
):
    message_text = [
        {
            "role": "system",
            "content": "You are an expert software developer. Generate a program satisfying the requirements given a context input.",
        },
        {"role": "user", "content": text},
        {"role": "assitant", "content": assitant_text},
        {"role": "user", "content": new_text},
    ]
    prompt = tokenizer.apply_chat_template(message_text, tokenize=False)
    return prompt
