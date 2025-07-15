import os
from abc import ABC, abstractmethod
from typing import List
from warnings import warn
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    warn("VLLM decoder will not work. Fix by `pip install vllm`")

EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]


def extra_eos_for_direct_completion(dataset) -> List[str]:
    if dataset.lower() == "humaneval":
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    elif dataset.lower() == "mbpp":
        return ['\n"""', "\nassert"]
    raise ValueError(f"Unknown dataset: {dataset}")


# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer: AutoTokenizer,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return task_prompt

    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"

    task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
    return task_prompt


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        dtype: str = "bfloat16",  # default
        trust_remote_code: bool = False,
        instruction_prefix: str = None,
        response_prefix: str = None,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.instruction_prefix = instruction_prefix
        self.response_prefix = response_prefix

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class VllmDecoder(DecoderBase):
    def __init__(self, name: str, dataset: str, tp: int, **kwargs) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", tp)),
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if self.tokenizer.chat_template is None:
            self.eos += extra_eos_for_direct_completion(dataset)
        self.llm = LLM(
            model=name, max_model_len=2048, max_lora_rank=64, enable_lora=True, **kwargs
        )

    def is_direct_completion(self) -> bool:
        return self.tokenizer.chat_template is None

    def codegen(
        self,
        prompt: str,
        do_sample: bool = True,
        num_samples: int = 200,
        lora_path: str = None,
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        if lora_path is not None:
            lora_request = LoRARequest("default", 1, lora_path)
        else:
            lora_request = None

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            lora_request=lora_request,
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs


class GeneralVllmDecoder(VllmDecoder):
    def __init__(self, name: str, lora_path, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.eos += ["\n```\n"]
        self.lora_path = lora_path
        print(f"EOS strings: {self.eos}")

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        prompt = make_chat_prompt(
            prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
        )
        return VllmDecoder.codegen(
            self, prompt, do_sample, num_samples, lora_path=self.lora_path
        )


def make_model(
    model: str,
    dataset: str,
    batch_size: int = 1,
    temperature: float = 0.0,
    tp=1,
    instruction_prefix=None,
    response_prefix=None,
    lora_path: str = None,
):
    return GeneralVllmDecoder(
        name=model,
        lora_path=lora_path,
        batch_size=batch_size,
        temperature=temperature,
        dataset=dataset,
        tp=tp,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
    )
