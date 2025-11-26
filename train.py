import os
import subprocess
import wandb

# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import trl
from trl import TrlParser, ModelConfig
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass
import time

# Custom imports
from reward_func import (
    extract_hash_answer,
    correctness_reward_func,
    levenstein_distance,
    bert_score,
    bleu_score
)
from data_utils import (
    get_gsm8k_dataset, 
    get_hf_math_dataset, 
    get_anker_math_dataset, 
    set_random_seed
    )
from custom_dgrpo_config import DGRPOConfig
from custom_dgrpo_trainer import ValSETrainer 
# from demo_trainer import DGRPOTrainer

os.environ["ACCELERATE_USE_FSDP"] = "false"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

# Main function for training
def main(dgrpo_config, model_config, args, use_vllm: Optional[bool] = True, vllm_mode = None) -> None:
    # Set random seed for reproducibility
    set_random_seed(dgrpo_config.seed)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # We are going to use "Llama3"

    # Load Dataset based on Configuration
    if args.dataset_name == "gsm8k":
        dataset = get_gsm8k_dataset(split=args.dataset_split)
        reward_functions = {
            "correctness": [
                correctness_reward_func
            ]
        }

    elif args.dataset_name == "hf_math":
        dataset = get_hf_math_dataset(split=args.dataset_split)
        reward_functions = {
            "correctness": [
                correctness_reward_func
            ]
        }
    elif args.dataset_name == "anker_math":
        dataset = get_anker_math_dataset(split=args.dataset_split)
        reward_functions = {
            "correctness": [
                correctness_reward_func,
            ],
            "diversity": [
                # bert_score,
                bleu_score
            ]
        }
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    # Set up Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4bit Configuration >> But might not be used (will use the option, 'use_vllm')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.float16
    )

    # Load Pretrained Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=model_config.torch_dtype,
        device_map="auto",
    )

    # model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r = model_config.lora_r,
        lora_alpha = model_config.lora_alpha,
        lora_dropout = model_config.lora_dropout,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type = "CAUSAL_LM"
    )
    
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    print(model)
    for i in range(dgrpo_config.num_guidance_adapters):
        model.add_adapter(f"guidance_{i}", peft_config=peft_config)
        print(f"  >> Num of Parameters after adding adapter guidance_{i}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)
    print(f"  >> Num of Parameters after PEFT: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(list(model.peft_config.keys()))
 
    ##### Test Cycle Before Training Starts #####
    print(" ===== TEST CYCLE BEFORE TRAINING STARTS ===== ")

    logits_stats = {}

    test_samples = dataset.select(range(2))
    inputs = tokenizer(
        list(test_samples['problem']), 
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    print(f" >> Main Adapter (Default) Activation Test ")
    model.set_adapter("default")
    print(f"  - Active Adapter: {model.active_adapters} ")
    with torch.no_grad(): # No gradient calculation during testing
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

    logits_default = outputs.logits
    default_mean = torch.mean(logits_default).item()
    logits_stats["default"] = default_mean
    print(f"  >> 'default' Logits Shape: {logits_default.shape} ")
    print(f"  >> 'default' Logits Mean: {default_mean} ")

    for i in range(dgrpo_config.num_guidance_adapters):
        adapter_name = f"guidance_{i}"
        print(f" >> Guide Adapter {i} Activation Test ")
        model.set_adapter(adapter_name)
        print(f"  - Active Adapter: {model.active_adapters} ")
        with torch.no_grad():
            outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits_guide = outputs.logits
        guidance_mean = torch.mean(logits_guide).item()
        logits_stats[adapter_name] = guidance_mean
        print(f"  >> '{adapter_name}' Logits Shape: {logits_guide.shape} ")
        print(f"  >> '{adapter_name}' Logits Mean: {guidance_mean} ")

    print("  >> Current Active Adapter: ", model.active_adapters)

    model.set_adapter("default")

    print(" ===== END OF TEST CYCLE ===== ")
    print("  >> Current Active Adapter: ", model.active_adapters)

    print("  >> Adapters are Initialized and Ready for Training. ")
    
    # Set wandb setting
    wandb.init(
        project = f"DGRPO-{args.dataset_name}",
        name = args.experiment_name,
    )
    # Initialize GRPO Trainer
    trainer = ValSETrainer( # ValSETrainer(
        args = dgrpo_config,
        model = model,
        train_dataset = dataset,
        reward_funcs_correctness= reward_functions["correctness"],
        reward_funcs_diversity= reward_functions["diversity"],
        # reward_funcs = reward_functions,
        # log_completions = True, #True, #False
    )
    print("  >> Training Arguments: ", trainer.args)

    # Log Completions for each Adapter

    trainer.train()

@dataclass
class CustomArgument:
    # num_guidance_adapters: int = 2
    model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_name: str = "anker_math"
    dataset_split: str = "train"
    experiment_name: str = f"dgrpo_experiment_{time.strftime('%Y%m%d_%H%M%S')}"

if __name__ == "__main__":

    parser = TrlParser((DGRPOConfig, ModelConfig, CustomArgument))
    dgrpo_config, model_config, args = parser.parse_args_and_config()
    dgrpo_config.num_generations = (dgrpo_config.num_guidance_adapters * dgrpo_config.num_generations_per_diversity_adapters) + dgrpo_config.num_generations_per_base_adapter
    main(dgrpo_config, model_config, args, use_vllm=dgrpo_config.use_vllm)

    # parser = TrlParser((DGRPOConfig, ModelConfig))
    # grpo_config, model_config = parser.parse_args_and_config()
    # main(grpo_config, model_config, use_vllm=grpo_config.use_vllm)
