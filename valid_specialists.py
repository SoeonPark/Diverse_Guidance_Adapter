import inspect
import os
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Callable, Optional
from transformers import Trainer

import datasets
import pandas as pd
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_trackio_available,
    is_wandb_available,
    TrainerCallback
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available, is_rich_available
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
    prepare_multimodal_messages_vllm,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.base_trainer import BaseTrainer
from trl.trainer.callbacks import SyncRefModelCallback
from trl import GRPOConfig, GRPOTrainer
from custom_dgrpo_config import DGRPOConfig
from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    get_config_model_id,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)
from transformers.utils import (
    is_torch_xpu_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_optimi_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging, strtobool,
    is_accelerate_available
)
from pathlib import Path
import json
from time import time

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearDGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

if is_trackio_available():
    import trackio

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False



logger = logging.get_logger(__name__)
logging.get_logger("vllm").setLevel(logging.WARNING)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = str | PreTrainedModel | Callable[[list, list], list[float]]

# What we call a rollout function is a callable that takes prompts (list), args (DGRPOConfig), and processing_class as
# parameters and returns a dict of generation results. Those results must include "prompt_ids", "completion_ids", and
# "logprobs" fields. Any extra fields (per-completion) are forwarded to the reward functions.
RolloutFunc = Callable[[list[str], Any, Any], dict[str, Any]]

class ValSETrainer(BaseTrainer):

    _tag_names = ["trl", "grpo"]
    _name = "GRPO"
    _paper = {
        "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        "id": "2402.03300",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{shao2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """),
    }


    def _save_checkpoint(self, model, trial, metrics=None):
        """Save all adapters separately."""
        checkpoint_folder = f"{self.args.output_dir}/checkpoint-{self.state.global_step}"
        os.makedirs(checkpoint_folder, exist_ok=True)
        
        # Save each adapter
        for adapter_name in self.adapter_names:
            adapter_path = f"{checkpoint_folder}/{adapter_name}"
            model.set_adapter(adapter_name)
            model.save_pretrained(
                adapter_path,
                save_adapter=True,
                save_config=True,
                safe_serialization=True
            )
            print(f"  >> Saved adapter '{adapter_name}' to {adapter_path}")
        
        # Save training state
        super()._save_checkpoint(model, trial, metrics)

    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override to save final model with all adapters.
        
        Called at the end of training via trainer.train().
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if is_peft_available() and isinstance(self.model, PeftModel):
            # Save each adapter
            for adapter_name in self.adapter_names:
                adapter_path = os.path.join(output_dir, adapter_name)
                os.makedirs(adapter_path, exist_ok=True)
                
                self.model.set_adapter(adapter_name)
                self.model.save_pretrained(
                    adapter_path,
                    save_adapter=True,
                    save_config=True,
                    safe_serialization=True
                )
                if self.accelerator.is_main_process:
                    print(f"  >> [Final Model] Saved adapter '{adapter_name}' to {adapter_path}")
            
            # Also save the base model config (once)
            if self.accelerator.is_main_process:
                self.model.config.save_pretrained(output_dir)
        else:
            # For non-PEFT models, use default save
            super().save_model(output_dir, _internal_call)


    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs_correctness: Union[RewardFunc, list[RewardFunc]], # NEW
        reward_funcs_diversity: Union[RewardFunc, list[RewardFunc]], # NEW
        adapter_name: Optional[str] = None, # NEW
        args: Optional[DGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        rollout_func: Optional[RolloutFunc] = None,
    ):
        
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = DGRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str):  # it's a str, but not "auto"
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = get_config_model_id(model.config)
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config), truncation_side="left")

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Reward functions
        # Correctness reward functions
        # self.reward_func_names = []
        self.reward_func_names_correctness = []
        for i, reward_func in enumerate(reward_funcs_correctness):
            if isinstance(reward_func, str):
                reward_funcs_correctness[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs_correctness[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names_correctness.append(get_config_model_id(reward_funcs_correctness[i].config).split("/")[-1])
            else:
                self.reward_func_names_correctness.append(reward_funcs_correctness[i].__name__)
        self.reward_funcs_correctness = reward_funcs_correctness
        # self.reward_func_names_correctness = self.reward_func_names.copy()

        # Diversity reward functions
        self.reward_func_names_diversity = []
        for i, reward_func in enumerate(reward_funcs_diversity):
            if isinstance(reward_func, str):
                reward_funcs_diversity[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs_diversity[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names_diversity.append(get_config_model_id(reward_funcs_diversity[i].config).split("/")[-1])
            else:
                self.reward_func_names_diversity.append(reward_funcs_diversity[i].__name__)
        self.reward_funcs_diversity = reward_funcs_diversity
        
        if adapter_name is None:
            adapter_list = ["default"]
            for i in range(args.num_guidance_adapters):
                adapter_list.append(f"guidance_{i}")

            self.adapter_names = adapter_list

        elif isinstance(adapter_name, str):
            self.adapter_names = [adapter_name]
            
        else:
            raise ValueError("adapter_name must be either None or a string.")
    
        
        self.num_generations_per_adapter = []
        for name in self.adapter_names:
            if name == "default":
                self.num_generations_per_adapter.append(args.num_generations_per_base_adapter)
            elif name.startswith("guidance_"):
                self.num_generations_per_adapter.append(args.num_generations_per_diversity_adapters)
            else:
                assert False, f"Unknown adapter name '{name}'"
        
        self.reward_func_names = self.reward_func_names_correctness + self.reward_func_names_diversity

        # breakpoint()

        self.total_num_generations = sum(self.num_generations_per_adapter)

        if is_peft_available() and isinstance(model, PeftModel):
            for adapter_name in self.adapter_names:
                if adapter_name != "default":
                    try:
                        model.set_adapter(adapter_name)
                        logger.info(f"Adapter '{adapter_name}' is available")
                    except:
                        logger.warning(f"Adapter '{adapter_name}' not found in model")

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs_correctness):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs_correctness)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs_correctness), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs_correctness)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs_correctness):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs_correctness)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs_correctness)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(get_config_model_id(reward_func.config))
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

        # Rollout function
        if rollout_func is not None and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1":
            warnings.warn(
                "You are importing from 'rollout_func', which is an experimental feature. This API may change or be "
                "removed at any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )
        self.rollout_func = rollout_func

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_kernel = args.use_liger_kernel
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_kernel and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_kernel and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling. Please set"
                "`importance_sampling_level` to 'token'."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Cast LM Head To FP32
        if args.cast_lm_head_to_fp32:
            if not model.config.tie_word_embeddings:

                def cast_inputs_to_fp32(module, input):
                    return (input[0].float(),)

                model.lm_head = model.lm_head.float()
                model.lm_head.register_forward_pre_hook(cast_inputs_to_fp32)
                if self.ref_model is not None:
                    self.ref_model.lm_head = self.ref_model.lm_head.float()
                    self.ref_model.lm_head.register_forward_pre_hook(cast_inputs_to_fp32)
            else:
                raise NotImplementedError(
                    "`cast_lm_head_to_fp32=True` is only supported when the model has untied word embedding and language modeling head layers"
                    "i.e. `tie_word_embeddings` in the model config is False."
                )

        # Liger loss
        if self.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `use_liger_kernel` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                    # self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks

                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )


                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None

                # [Custom] Set path to temporary cache directory for vLLM for each adapter
                import tempfile
                self.lora_temp_dir = tempfile.mkdtemp(prefix="vllm_lora_cache_")

                self.lora_modules = None
                if is_peft_model(model):
                    self.lora_modules = []
                    for adapter_index, adapter_name in enumerate(self.adapter_names):
                        adapter_path = os.path.join(self.lora_temp_dir, f"{adapter_name}_adapter")
                        os.makedirs(adapter_path, exist_ok=True)

                        # Save each adapter to a separate directory
                        model.set_adapter(adapter_name)
                        # Save only PEFT adapters to the adapter_path
                        model.save_pretrained(adapter_path, save_adapter = True, save_config = True)

                        self.lora_modules.append(
                            {
                                "name": adapter_name,
                                "path": adapter_path,
                                "id": adapter_index + 1, 
                            }
                        )
                    print(f"  >> [vLLM Init] LoRA modules saved to temporary cache directory: {self.lora_temp_dir} | Total adapters: {len(self.lora_modules)}")

                # Initialize the vLLM LLM instance
                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None

                self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    # Important so temperature scaling/logit tweaking affects the TIS log probs
                    logprobs_mode="processed_logprobs",
                    enable_lora = True,
                    max_loras = len(self.adapter_names) if self.lora_modules else None,
                    max_lora_rank = 8,
                    quantization = "bitsandbytes", 
                    load_format = "bitsandbytes"
                )

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=2)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs_correctness):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs_correctness[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs_correctness[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )
    
    # [Not originally in GRPO] In transformers.Trainer:
    # Not Modified yet

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism
        # breakpoint()

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            # breakpoint()

            inputs = self._prepare_inputs(inputs)

            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            # adapter_names = inputs["adapter_info"]["adapter_names"]
            # num_generations_per_adapter = inputs["adapter_info"]["num_generations_per_adapter"]

            total_loss = 0.0

            adapter_batches = inputs.get("adapter_batches")

            # For the case 'inference mode' oronly use one adapter
            if adapter_batches is None:
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs, num_items_in_batch = num_items_in_batch)
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if not self.model_accepts_loss_kwargs or num_items_in_batch is None:
                    loss = loss / self.current_gradient_accumulation_steps

                kwargs = {}
                if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                    kwargs["learning_rate"] = self._get_learning_rate()

                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False
                
                self.accelerator.backward(loss, **kwargs)
                total_loss = loss.detach()

            else:
                # For the case 'training mode' with MULTIPLE adapters
                for adapter_index, adapter_inputs in enumerate(adapter_batches):

                    adapter_name = adapter_inputs["adapter_info"]["adapter_names"][0]
                    
                    # Set adapter
                    if is_peft_available() and isinstance(model, PeftModel):
                        model.set_adapter(adapter_name)
                    
                    print(f"  >> [Training Step] Processing adapter '{adapter_name}' ({adapter_index + 1}/{len(adapter_batches)})")
                    
                    # Forward & Backward
                    with self.compute_loss_context_manager():
                        # breakpoint()
                        loss = self.compute_loss(model, adapter_inputs, num_items_in_batch=num_items_in_batch)
                    
                    if self.args.n_gpu > 1:
                        loss = loss.mean()
                    
                    # Normalize loss
                    if not self.model_accepts_loss_kwargs or num_items_in_batch is None:
                        if self.compute_loss_func is None:
                            loss = loss / self.current_gradient_accumulation_steps
                    
                    kwargs = {}
                    if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                        kwargs["learning_rate"] = self._get_learning_rate()
                    
                    if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                        kwargs["scale_wrt_gas"] = False
                    
                    # Backward
                    self.accelerator.backward(loss, **kwargs)
                    
                    total_loss += loss.detach()
                    
                    del adapter_inputs
                    if self.args.torch_empty_cache_steps is not None:
                        if adapter_index < len(adapter_batches) - 1: # avoid redundant empty_cache at the end of last adapter
                            if is_torch_xpu_available():
                                torch.xpu.empty_cache()
                            elif is_torch_mlu_available():
                                torch.mlu.empty_cache()
                            elif is_torch_musa_available():
                                torch.musa.empty_cache()
                            elif is_torch_npu_available():
                                torch.npu.empty_cache()
                            elif is_torch_mps_available():
                                torch.mps.empty_cache()
                            else:
                                torch.cuda.empty_cache()

                print(f"\n=== Adapter '{adapter_name}' Gradient Check ===")
                for name, param in model.named_parameters():
                    if adapter_name in name and param.grad is not None:
                        print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")
            
            return total_loss

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        # breakpoint()
        for start in range(0, input_ids.size(0), batch_size):
            # breakpoint()
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def _move_model_to_vllm(self, adapter_name: Optional[str] = None):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model) and self.lora_modules:
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            adapters_to_process = [adapter_name] if adapter_name else self.adapter_names

            for adapter in adapters_to_process:
                print(f"  >> [vLLM Sync] Syncing Adapter '{adapter}' to vLLM...")

                adapter_info = next((module for module in self.lora_modules if module["name"] == adapter), None)

                if not adapter_info:
                    print(f"  >> [Warning] Adapter info for '{adapter}' not found in lora_modules.")
                    continue

                self.model.set_adapter(adapter)
                adapter_path = adapter_info["path"]
                with gather_if_zero3(list(self.model.parameters())):
                    # self.model.merge_adapter()

                    self.model.save_pretrained(
                        adapter_path, save_adapter=True, save_config=True, safe_serialization=True
                    )
                    print(f"  >> [vLLM Sync] Adapter '{adapter}' merged and saved to '{adapter_path}'.")

                    # # Update vLLM weights while parameters are gathered
                    # if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    #     # Update vLLM weights while parameters are gathered
                    #     # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    #     fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    #     fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    #     if fsdp_version == 1:
                    #         self._sync_fsdp1_params_to_vllm(
                    #             self.model
                    #         )  # use memory-efficient post-order traversal for FSDP
                    #     elif fsdp_version == 2:
                    #         self._sync_fsdp2_params_to_vllm(self.model)
                    # else:
                    #     # DeepSpeed ZeRO-3 with PEFT
                    #     for name, param in self.model.named_parameters():
                    #         # When using PEFT, we need to recover the original parameter name and discard some parameters
                    #         name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    #         if self.model.prefix in name:
                    #             continue
                    #         # When module to save, remove its prefix and discard the original module
                    #         if "original_module" in name:
                    #             continue
                    #         name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                    #         if self.vllm_mode == "server" and self.accelerator.is_main_process:
                    #             self.vllm_client.update_named_param(name, param.data)
                    #         elif self.vllm_mode == "colocate":
                    #             llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    #             llm_model.load_weights([(name, param.data)])
                    # # Unmerge adapters while parameters are still gathered
                    # self.model.unmerge_adapter()
                    # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
            Prepares inputs for training by generating completions using vLLM and organizing them for loss computation.

            Functions to Pass Inputs:
                * _generate_completions: 
                    Gets `prompts` with all same contents as number as `num_generations` | Returns `generation_batch` with `completions` added.
                * _score_completions:
                    Gets `generation_batch` with `prompts` and `completions` | Returns `generation_batch` with `logprobs` added.
                * split_pixel_values_by_grid:

                * shuffle_sequence_dict:
                * split_tensor_dict:
                * unsplit_pixel_values_by_grid:
        """

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                original_inputs = generation_batch

                print(original_inputs)
                print(f">>> [Generation Step] Generating completions for step {self._step}/{generate_every}...")
                print(self.args.steps_per_generation)
                # breakpoint()
                generation_batch = self._generate_completions(generation_batch)
                # breakpoint()
                scored_outputs = self._score_completions(original_inputs, generation_batch)

                for key in scored_outputs.keys():
                    if isinstance(scored_outputs[key], torch.Tensor):
                        print(f"  >> scored_outputs['{key}']: {scored_outputs[key].shape} {scored_outputs[key].dtype}")
                        # breakpoint()
                    elif isinstance(scored_outputs[key], list):
                        print(f"  >> scored_outputs['{key}']: List of length {len(scored_outputs[key])}")
                    if key == "num_items_in_batch" or key == "all_extra_fields":
                        print(f"      >> scored_outputs['{key}']: {scored_outputs[key]}")
                # breakpoint()
                """
                (Pdb) scored_outputs.keys()
                dict_keys(['prompt_ids', 'prompt_mask', 'completion_ids', 'completion_mask', 'num_items_in_batch', 'old_per_token_logps', 'ref_per_token_logps', 'sampling_per_token_logps', 'all_extra_fields', 'adapter_info', 'advantages'])
                """

                adapter_info = scored_outputs["adapter_info"]
                adapter_boundaries = adapter_info["adapter_boundaries"]
                num_items_in_batch = scored_outputs["num_items_in_batch"]
                scored_outputs = split_pixel_values_by_grid(scored_outputs)

                # adapter_batches = {}
                adapter_batches = []

                # breakpoint()

                for adapter_index, (start_index, end_index) in enumerate(adapter_boundaries):
                    breakpoint()
                    adapter_data = {}
                    adv_index = 0
                    for key, value in scored_outputs.items():
                        if key == "advantages":
                            # advantages are already repeated per generation
                            adapter_data[key] = value[adv_index: adv_index + (end_index - start_index)].detach()
                        elif key == "old_per_token_logps":
                            adapter_data[key] = value[adv_index: adv_index + (end_index - start_index)].detach()
                        elif key == "num_items_in_batch":
                            pass
                        # elif key == "all_extra_fields":
                        #     pass
                        else:
                            if isinstance(value, (torch.Tensor, list)):
                                if len(value) > 0:
                                    sliced_val = value[start_index:end_index]
                                    if len(sliced_val) == (end_index - start_index):
                                        if isinstance(sliced_val, torch.Tensor):
                                            adapter_data[key] = sliced_val.detach()
                                        else:
                                            adapter_data[key] = sliced_val
                        
                        adv_index
                    
                    adapter_data = split_pixel_values_by_grid(adapter_data)
                    # breakpoint()
                    adapter_data = shuffle_sequence_dict(adapter_data)
                    # generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                    adapter_data = unsplit_pixel_values_by_grid(adapter_data)
                    adapter_inputs = {
                        **adapter_data,
                        "adapter_info": {
                            "adapter_names": [adapter_info["adapter_names"][adapter_index]],
                            "num_generations_per_adapter": [adapter_info["num_generations_per_adapter"][adapter_index]],
                            "adapter_boundaries": [(adapter_boundaries[adapter_index])],
                        },
                        "num_items_in_batch": num_items_in_batch
                    }
                    # breakpoint()
                    
                    # adapter_batches = {**adapter_data, **adapter_metadata}
                    adapter_batches.append(adapter_inputs)
                self._buffered_inputs = [{"adapter_batches":adapter_batches}]

            inputs = self._buffered_inputs[self._step % len(self._buffered_inputs)]
            self._step += 1

        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            original_inputs = generation_batch
            inputs = self._generate_completions(generation_batch)
            inputs = self._score_completions(original_inputs, inputs)
        return inputs
    
    # Calculate Rewards (Correctness and Diversity)
    @profiling_decorator
    def _calculate_rewards_correctness(
        self, inputs: dict[str, Union[torch.Tensor, Any]],
        prompts: list[str],
        completions: list[str],
        completion_ids_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculates correctness rewards for the generated completions using the specified reward functions.

        Args:
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                A dictionary containing the input prompts and other necessary information.
            prompts (`list[str]`):
                A list of input prompts.
            completions (`list[str]`):
                A list of generated completions corresponding to the prompts.
            completion_ids_list (`list[torch.Tensor]`):
                A list of tensors containing the token IDs of the generated completions.  
        Returns:
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs_correctness), device=device)
        # breakpoint()

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        # breakpoint()
        """
        (Pdb) keys
        ['problem', 'level', 'type', 'solution', 'answer']
        """

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        # breakpoint()
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            # zip(self.reward_funcs_correctness, self.reward_processing_classes, self.reward_func_names_correctness)
            zip(self.reward_funcs_correctness, self.reward_processing_classes, self.reward_func_names_correctness)
        ):
            # breakpoint()
            with profiling_context(self, f"reward_func_correctness.{reward_func_name}"):
                if isinstance(reward_func, nn.Module): # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["messages"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    # breakpoint()
                    reward_inputs = reward_processing_class(
                        texts=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                    # breakpoint()
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_index = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_index] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_index]
            row_reward_kwargs["completion"] = completions[nan_row_index]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )
        # breakpoint()

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func
    
    @profiling_decorator
    def _calculate_rewards_diversity(
        self, inputs: dict[str, Union[torch.Tensor, Any]],
        prompts: list[str],
        completions: list[str],
        completion_ids_list: list[torch.Tensor],
        other_adapter_data: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Calculates diversity rewards for the generated completions using the specified reward functions

        Diversity rewards measure how different each completion is from completions generated by other adapters.
        For each completion, we compare it against all completions from other adapters and take the MAXIMUM
        similarity score as the diversity reward (higher similarity = less diverse = higher reward for avoiding).
        
        Example: With 10 total completions (4/3/3 from base/guidance1/guidance2):
            - For 1 completion from guidance1: compare with 7 others (4 from base + 3 from guidance2)
            - Diversity reward = max(similarity with those 7 completions)
        
        Args:
            inputs: Dictionary containing input prompts and metadata
            prompts: List of input prompts for current adapter's completions
            completions: List of generated completions for current adapter
            completion_ids_list: List of token IDs for current adapter's completions
            other_adapter_data: Dictionary containing completions from other adapters
                - Keys: 'prompt_ids', 'completion_ids', 'completion_mask', 'sampling_per_token_logps'
        
        Returns:
            rewards_per_func: Tensor of shape (num_current_completions, num_diversity_funcs)
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs_diversity), device=device)

        assert other_adapter_data is not None or other_adapter_data.get("completion_ids") is not None

        other_completion_ids = other_adapter_data["completion_ids"]
        other_completion_mask = other_adapter_data["completion_mask"]

        assert other_completion_ids.size(0) != 0 

        other_completions_text = self.processing_class.batch_decode(other_completion_ids, skip_special_tokens = True)
        num_others = len(other_completions_text)
        # breakpoint()

        # print(f"  >> [Diversity Reward] Comparing {len(completions)} completions against {num_others} other completions.")

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            # zip(self.reward_funcs_diversity, self.reward_processing_classes, self.reward_func_names_diversity)
            zip(self.reward_funcs_diversity, self.reward_processing_classes, self.reward_func_names_diversity)
        ):
            print(f"  >> Using `reward_func_diversity`: {reward_func_name}")
            with profiling_context(self, f"reward_func_diversity.{reward_func_name}"):
                # breakpoint()
                if isinstance(reward_func, nn.Module): # Module (no PretrainedModel) for compat with compiled models
                    all_avg_similarities = []

                    for curr_index, (curr_prompt, curr_completion) in enumerate(zip(prompts, completions)):
                        print(f"  >> [Diversity Reward] Comparing {curr_index}/{len(completions)} completions against {num_others} other completions.")
                        
                        print(f"  >> Current Completion {curr_index} : \n {curr_completion}")
                        similarities = []

                        for other_completion in other_completions_text:
                            if is_conversational(inputs[0]):
                                # messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                                messages = {"messages": curr_prompt + curr_completion + other_completion}
                                texts = [
                                    apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["messages"]
                                    for x in messages
                                ]
                            else:
                                texts = [curr_prompt + curr_completion, curr_prompt + other_completion]
                                    
                            reward_inputs = reward_processing_class(
                                texts=texts,
                                return_tensors="pt",
                                padding=True,
                                padding_side="right",
                                add_special_tokens=False
                            )
                            reward_inputs = super()._prepare_inputs(reward_inputs)

                            with torch.inference_mode():
                                similarity = reward_func(**reward_inputs).logits[0, 0]
                                # breakpoint()
                                similarities.append(similarity.item())
                                # rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)

                            all_avg_similarities.append(similarities)
                            # breakpoint()

                        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                        all_avg_similarities.append(avg_similarity)
                    rewards_per_func[:, i] = torch.tensor(all_avg_similarities, dtype = torch.float32, device = device)

                else: # For the case not using nn.Module reward functions
                    print(f"  >> Compare using non-module diversity reward function: {reward_func_name}")
                    if other_completions_text and isinstance(other_completions_text[0], str):
                        other_completions_text = [[{"role": "assistant", "content": text}] for text in other_completions_text]
                    else:
                        other_completions_text = other_completions_text

                    reward_kwargs["other_completions"] = other_completions_text
                    reward_kwargs["other_completion_ids"] = [
                        ids[mask.bool()].tolist() for ids, mask in zip(other_completion_ids, other_completion_mask)
                    ]
                    
                    all_rewards = []
                    for index in range(len(prompts)):
                        if isinstance(completions[index], str):
                            curr_completion = [[{"role": "assistant", "content": completions[index]}]]
                        else:
                            curr_completion = [completions[index]]

                        reordered_prompts = [prompts[index]] + [prompts[index]] * len(other_completions_text)
                        reordered_completions = curr_completion + other_completions_text

                        reordered_kwargs = {}
                        
                        for key, value in reward_kwargs.items():
                            if key in ["other_completions", "other_completion_ids", "trainer_state"]:
                                continue
                            if isinstance(value, list) and len(value) == len(prompts):
                                reordered_kwargs[key] = [value[index]]

                        output_reward_func = reward_func(
                            prompts = reordered_prompts, 
                            completions = reordered_completions, 
                            completion_ids = [completion_ids_list[index]] + reward_kwargs["other_completion_ids"], 
                            **reordered_kwargs,
                        )
                        reward_value = output_reward_func[0] if output_reward_func else torch.nan
                        all_rewards.append(reward_value if reward_value is not None else torch.nan)
                    
                    rewards_per_func[:, i] = torch.tensor(all_rewards, dtype=torch.float32, device=device)
                        

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_index = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_index] for key, value in reward_kwargs.items() 
                if key not in ["trainer_state", "other_completions", "other_completion_ids"]
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_index]
            row_reward_kwargs["completion"] = completions[nan_row_index]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        # breakpoint()
        return rewards_per_func

    def _generate_single_turn(self, prompts: list, adapter_name: Optional[str] = None):
        device = self.accelerator.device

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up(tags=["weights"])

            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                if adapter_name:
                    self._move_model_to_vllm(adapter_name=adapter_name)
                else:
                    self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if is_conversational({"prompt": prompts[0]}):
                prompts = [prepare_multimodal_messages_vllm(prompt) for prompt in prompts]

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts = gather_object(prompts)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts[:: self.num_generations]

                    sampling_params = {
                        "n": self.num_generations,
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": -1 if self.top_k is None else self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": self.max_completion_length,
                        "truncate_prompt_tokens": self.max_prompt_length,
                        "guided_decoding_regex": self.guided_decoding_regex,
                        "generation_kwargs": self.args.generation_kwargs,
                    }

                    with profiling_context(self, "vLLM.generate"):
                        if self.rollout_func is not None:
                            if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                                ordered_set_of_prompts = [
                                    apply_chat_template(
                                        {"prompt": p}, self.processing_class, **self.chat_template_kwargs
                                    )["prompt"]
                                    for p in ordered_set_of_prompts
                                ]
                            output = self.rollout_func(
                                ordered_set_of_prompts,
                                self.args,
                                self.processing_class,
                            )
                        else:
                            if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                                # FIXME: this endpoint doesn't exist in vllm_client
                                output = self.vllm_client.chat(
                                    prompts=ordered_set_of_prompts,
                                    **sampling_params,
                                    chat_template_kwargs=self.chat_template_kwargs,
                                    lora_request=lora_request,
                                )
                            else:
                                output = self.vllm_client.generate(prompts=ordered_set_of_prompts, **sampling_params, lora_request=lora_request)
                        # Extract required fields and collect any extra fields for reward functions
                        required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                        extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                        payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"], extra_fields)
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs, all_extra_fields = obj_list[0]

                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]

                # Slice extra fields dict-of-lists per process (extra fields are per-completion, like completion_ids)
                extra_fields = {}
                for key, values in all_extra_fields.items():
                    if isinstance(values, list):
                        extra_fields[key] = values[process_slice]
                    else:
                        extra_fields[key] = values

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "truncate_prompt_tokens": self.max_prompt_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                lora_request = None
                if adapter_name and self.lora_modules:
                    adapter_info = next(
                        (module for module in self.lora_modules if module["name"] == adapter_name), None
                    )
                    if adapter_info:
                        from vllm.lora.request import LoRARequest
                        lora_request = LoRARequest(
                            lora_name=adapter_info["name"],
                            lora_int_id = adapter_info["id"],
                            lora_local_path = adapter_info["path"],
                        )
                        print(f"  >> Using LoRA adapter in vLLM generation: {adapter_name} (id: {adapter_info['id']})")

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                    all_prompts = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts = prompts

                if self.args.vllm_enable_sleep_mode:
                    self.llm.wake_up(tags=["kv_cache"])

                with profiling_context(self, "vLLM.generate"):
                    if is_conversational({"prompt": prompts[0]}):
                        all_outputs = self.llm.chat(all_prompts, sampling_params=sampling_params, use_tqdm=False, lora_request=lora_request)
                    else:
                        all_outputs = self.llm.generate(all_prompts, sampling_params=sampling_params, use_tqdm=False, lora_request=lora_request)

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs

                extra_fields = {}  # No extra fields for colocate mode

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=2)

        elif self.use_transformers_paged:
            processor_kwargs = {
                "max_length": self.max_prompt_length,
                "truncation": True,
                "add_special_tokens": False,
            }
            if is_conversational({"prompt": prompts[0]}):
                processor_outputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    **processor_kwargs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                processor_outputs = self.processing_class(text=prompts, **processor_kwargs)

            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                if self.args.cast_lm_head_to_fp32:
                    unwrapped_model.lm_head.to(torch.float32)
                with torch.inference_mode():
                    # Continuous batching API expects 'inputs' arg only
                    all_outputs = unwrapped_model.generate_batch(
                        processor_outputs["input_ids"], generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = processor_outputs["input_ids"]
            logprobs = None  # not used in this case
            extra_fields = {}  # No extra fields for paged mode

        else:
            # Regular generation path
            processor_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "padding_side": "left",
                "max_length": self.max_prompt_length,
                "truncation": True,
                "add_special_tokens": False,
            }
            if is_conversational({"prompt": prompts[0]}):
                generate_inputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    **processor_kwargs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                generate_inputs = self.processing_class(text=prompts, **processor_kwargs)
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_index = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_index[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_index.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
            logprobs = None  # not used in this case
            extra_fields = {}  # No extra fields for non-rollout_func paths

        return prompt_ids, completion_ids, logprobs, extra_fields

    def _generate(self, prompts: list, adapter_name: Optional[str] = None):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts, adapter_name)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids, total_completion_tokens, logprobs, extra_fields

    # Split Generation step and Scoring step for easier overriding in subclasses
    def _generate_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Generates completions for the given prompts using vLLM and adds them to the generation batch.

        Args:
            generation_batch (`dict[str, Union[torch.Tensor, Any]]`):
                A dictionary containing the input prompts and other necessary information for generation.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        all_prompt_ids_list = []
        all_completion_ids_list = []
        all_sampling_per_token_logps_list = []
        all_extra_fields_list = []
        total_num_items = 0

        adapter_boundaries = []
        current_adapter = 0

        prompts = [x["prompt"] for x in inputs]
        """
        (Pdb) len(prompts)
        10
        """
        # breakpoint()

        # Generate completions by adapters
        for adapter_index, (adapter_name, num_generation) in enumerate(zip(self.adapter_names, self.num_generations_per_adapter)):
            if is_peft_available() and isinstance(self.model, PeftModel):
                self.model.set_adapter(adapter_name)
                if self.use_vllm and self.vllm_mode == "colocate":
                    self._move_model_to_vllm()  # update vLLM weights for the new adapter
                    self._last_loaded_step = self.state.global_step

            print(f"  >> [Rollout] Generating {num_generation} completions using adapter '{adapter_name}'")
            
            # Repeat each prompt 'num_generations' times for the current adapter
            # batch_size = len(inputs)

            # breakpoint()
            """
            (Pdb) len(completion_ids_list[0])
            412
            (Pdb) num_items_in_batch
            tensor(412, device='cuda:0')
            """
            start_index = current_adapter
            end_index = current_adapter + num_generation
            
            prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
                self._generate(prompts[start_index:end_index], adapter_name=adapter_name) 
            )

            if adapter_name == "default":
                adapter_boundaries.append((start_index, sum(self.num_generations_per_adapter)))
            else:
                adapter_boundaries.append((start_index, end_index))
            current_adapter = end_index
            # breakpoint()

            all_prompt_ids_list.extend(prompt_ids_list)
            all_completion_ids_list.extend(completion_ids_list)
            if sampling_per_token_logps_list is not None:
                all_sampling_per_token_logps_list.extend(sampling_per_token_logps_list)
            all_extra_fields_list.append(extra_fields)
            total_num_items += num_items_in_batch
            # breakpoint()
            print(f"  >> [Rollout] Generated a total of {total_num_items} completion tokens. | Adapter boundaries: {adapter_boundaries} | Adapter names: {adapter_name}")

        if self.use_vllm and self.vllm_mode == "colocate":
            if self.args.vllm_enable_sleep_mode:
                print("  >> [Memory] Putting vLLM to sleep to save resources.")
                self.llm.sleep(level=2)
            torch.cuda.empty_cache()

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in all_prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        # breakpoint()
        """
        (Pdb) prompt_ids.shape
        torch.Size([1, 106])
        """

        completion_ids = [torch.tensor(ids, device=device) for ids in all_completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        # breakpoint()
        """
        (Pdb) completion_ids.shape
        torch.Size([1, 412])
        """

        if all_sampling_per_token_logps_list:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in all_sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = None
        forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):  
                all_old_per_token_logps = []
                for start_index, end_index in adapter_boundaries:
                    old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids[start_index:end_index],
                        attention_mask[start_index:end_index],
                        logits_to_keep,
                        end_index-start_index,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                    all_old_per_token_logps.append(old_per_token_logps)
                old_per_token_logps = torch.cat(all_old_per_token_logps, dim=0)
            else:
                old_per_token_logps = None
            # breakpoint()

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "num_items_in_batch": total_num_items,
            "old_per_token_logps": old_per_token_logps if 'old_per_token_logps' in locals() else None,
            "ref_per_token_logps": ref_per_token_logps if 'ref_per_token_logps' in locals() else None,
            "sampling_per_token_logps": sampling_per_token_logps,
            "all_extra_fields": all_extra_fields_list,
            "adapter_info": {
                "adapter_names": self.adapter_names,
                "num_generations_per_adapter": self.num_generations_per_adapter,
                "adapter_boundaries": adapter_boundaries,
            }
        }
        return output
    
    def _score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]], 
            generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Scores the generated completions by computing log-probabilities using the model and adds them to the
        generation batch.

        Args:
            generation_batch (`dict[str, Union[torch.Tensor, Any]]`):
                A dictionary containing the input prompts, generated completions, and other necessary information for
                scoring.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        batch_size = len(prompts)

        prompt_ids = generation_batch["prompt_ids"]
        completion_ids = generation_batch["completion_ids"]
        completion_mask = generation_batch["completion_mask"]
        sampling_per_token_logps = generation_batch.get("sampling_per_token_logps", None)
        old_per_token_logps = generation_batch.get("old_per_token_logps", None)
        ref_per_token_logps = generation_batch.get("ref_per_token_logps", None)
        all_extra_fields = generation_batch.get("all_extra_fields", None)

        adapter_names = generation_batch["adapter_info"]["adapter_names"]
        num_generations_list = generation_batch["adapter_info"]["num_generations_per_adapter"]
        adapter_boundaries = generation_batch["adapter_info"]["adapter_boundaries"]
        # breakpoint()

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completion_ids_list = [ids[mask.bool()].tolist() for ids, mask in zip(completion_ids, completion_mask)]
        # breakpoint()
        
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}]) # prompt에 이걸 안했네ㅣ;;;
        else:
            completions = completions_text

        # breakpoint()
        if all_extra_fields is not None:
            if all_extra_fields and isinstance(all_extra_fields, dict):
                for i, input_example in enumerate(inputs):
                    for key, values in all_extra_fields.items():
                        if isinstance(values, list) and i < len(values):
                            input_example[key] = values[i]
                        elif not isinstance(values, list):
                            input_example[key] = values
            elif all_extra_fields and isinstance(all_extra_fields, list):
                for i, input_example in enumerate(inputs):
                    if i < len(all_extra_fields) and isinstance(all_extra_fields[i], dict):
                        input_example.update(all_extra_fields[i])

        total_completions = sum(num_generations_list)

        # Calculate Correctness Rewards Once for ALL!!!!
        print(f"  >> [Scoring] Scoring Correctness for a Total of {total_completions} Completions from Each Adapter")
        all_correctness_rewards = self._calculate_rewards_correctness(
            inputs, prompts, completions, completion_ids_list
        ).to(device)
        # Shape: (num_generations, num_correctness_reward_funcs) e.g., (3, 1)
        # breakpoint()

        all_rewards_per_func = []
        all_advantages = []

        for adapter_index, (adapter_name, num_generations, (start_index, end_index)) in enumerate(zip(adapter_names, num_generations_list, adapter_boundaries)):
            # breakpoint()            
            print(f"   >> [Scoring] Scoring Correctness for Completions from Adapter '{adapter_name}' -- {num_generations} generations({start_index}:{end_index})")

            if adapter_name == "default":
                rewards_per_func = all_correctness_rewards[start_index:end_index]
                # rewards = (rewards_per_func.to(device).unsqueeze(0)).nansum(dim=1)  # (N,)
                reward_func_names = self.reward_func_names_correctness
                num_generations = self.args.num_generations
                current_weights = self.reward_weights[:len(self.reward_funcs_correctness)].to(device)
                rewards = (rewards_per_func * current_weights.unsqueeze(0)).nansum(dim=1)  # (N,)

                for i, reward_func_name in enumerate(reward_func_names):
                    mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
                    self._metrics[mode][f"rewards/{adapter_name}/{reward_func_name}/mean"].append(mean_rewards)
                    std_func_rewards = nanstd(rewards_per_func[:, i]).item()
                    self._metrics[mode][f"rewards/{adapter_name}/{reward_func_name}/std"].append(std_func_rewards)
                    # breakpoint()

                all_rewards_per_func.append(rewards_per_func) # list
                    
                # Compute Advantages
                mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)

                std_grouped_rewards = rewards.view(-1, num_generations).std(dim=1)
                std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)

                advantages = rewards - mean_grouped_rewards  # (N,)
                # breakpoint()

                if self.scale_rewards in ["group", "none"]: # group 
                    # If self.scale_rewards = "none", we'll still log group level std
                    std_grouped_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
                elif self.scale_rewards == "batch":
                    # Compute global std
                    std_grouped_rewards = rewards.std().expand_as(rewards)
                else:
                    raise ValueError(
                        f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                    )

                is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))
                if self.scale_rewards != "none":
                    # A_{Main}
                    advantages = advantages / (std_grouped_rewards + 1e-4)

                self._metrics[mode][f"reward/{adapter_name}"].append(rewards.mean().item())
                self._metrics[mode][f"reward_std/{adapter_name}"].append(std_grouped_rewards.mean().item())
                self._metrics[mode][f"frac_reward_zero_std/{adapter_name}"].append(is_std_zero.float().mean().item())

            elif adapter_name.startswith("guidance_"):
                # Get current Adapter's Prompts and Completions
                curr_prompts = prompts[start_index:end_index]
                curr_completions = [completions[i] for i in range(start_index, end_index)]
                curr_completion_ids_list = [completion_ids_list[i] for i in range(start_index, end_index)]

                # For Guidance Adapters (Correctenss + Diversity)
                other_indices = list(range(start_index)) + list(range(end_index, total_completions))
                other_adapter_data = {
                    "prompt_ids": prompt_ids[other_indices] if len(other_indices) > 0 else None,
                    "completion_ids": completion_ids[other_indices] if len(other_indices) > 0 else None,
                    "completion_mask": completion_mask[other_indices] if len(other_indices) > 0 else None,
                    "sampling_per_token_logps": sampling_per_token_logps[other_indices] if sampling_per_token_logps is not None and len(other_indices) > 0 else None,
                }

                # Calculate Diversity Rewards
                diversity_rewards = self._calculate_rewards_diversity(
                    inputs, curr_prompts, curr_completions, curr_completion_ids_list,
                    other_adapter_data=other_adapter_data
                ).to(device)
                # Shape: (num_generations, num_diversity_reward_funcs) e.g., (3, 2)
                # breakpoint()
                if self.args.masking_diversity_by_correctness:
                    correctness_mask = (all_correctness_rewards[start_index:end_index] > 0.0).float() # (N, 1)
                    diversity_rewards = diversity_rewards * correctness_mask

                rewards_per_func = torch.cat([all_correctness_rewards[start_index:end_index], diversity_rewards], dim = 1)
                reward_func_names = self.reward_func_names_correctness + self.reward_func_names_diversity

                for i, reward_func_name in enumerate(reward_func_names):
                    mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
                    self._metrics[mode][f"rewards/{adapter_name}/{reward_func_name}/mean"].append(mean_rewards)
                    std_func_rewards = nanstd(rewards_per_func[:, i]).item()
                    self._metrics[mode][f"rewards/{adapter_name}/{reward_func_name}/std"].append(std_func_rewards)
                    # breakpoint()

                all_rewards_per_func.append(rewards_per_func) # list

                # Compute Advantages
                ## A_Correctness
                cor_weights = self.reward_weights[:len(self.reward_funcs_correctness)].to(device)
                correctness_rewards_weighted = (all_correctness_rewards[start_index:end_index] * cor_weights.unsqueeze(0)).nansum(dim=1)  # (N,)

                mean_grouped_correctness = correctness_rewards_weighted.view(-1, num_generations).mean(dim=1)
                mean_grouped_correctness = mean_grouped_correctness.repeat_interleave(num_generations, dim=0)

                advantages_correctness = correctness_rewards_weighted - mean_grouped_correctness

                if self.scale_rewards in ["group", "none"]: # group 
                    # If self.scale_rewards = "none", we'll still log group level std
                    std_grouped_rewards = correctness_rewards_weighted.view(-1, num_generations).std(dim=1)
                    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
                elif self.scale_rewards == "batch":
                    # Compute global std
                    std_grouped_rewards = rewards.std().expand_as(rewards)
                else:
                    raise ValueError(
                        f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                    )
                
                is_std_zero_cor = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))
                if self.scale_rewards != "none":
                    advantages = advantages_correctness / (std_grouped_rewards + 1e-4)
                

                ## A_Diversity
                div_weights = torch.ones(len(self.reward_funcs_diversity), dtype=torch.float32).to(device)
                diversity_rewards_weighted = (diversity_rewards * div_weights.unsqueeze(0)).nansum(dim=1)  # (N,)

                mean_grouped_diversity = diversity_rewards_weighted.view(-1, num_generations).mean(dim=1)
                mean_grouped_diversity = mean_grouped_diversity.repeat_interleave(num_generations, dim=0)

                advantages_diversity = diversity_rewards_weighted - mean_grouped_diversity

                if self.scale_rewards in ["group", "none"]:
                    # If self.scale_rewards = "none", we'll still log group level std
                    std_grouped_diversity = diversity_rewards_weighted.view(-1, num_generations).std(dim=1)
                    std_grouped_diversity = std_grouped_diversity.repeat_interleave(num_generations, dim=0)
                elif self.scale_rewards == "batch":
                    # Compute global std
                    std_grouped_diversity = diversity_rewards_weighted.std().expand_as(rewards)
                else:
                    raise ValueError(
                        f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                    )
                # breakpoint()

                is_std_zero_div = torch.isclose(std_grouped_diversity, torch.zeros_like(std_grouped_diversity))
                if self.scale_rewards != "none":
                    advantages_diversity = advantages / (std_grouped_diversity + 1e-4)
                    
                # Combine Advantages (A_{Specialist} = w1 * A_{Correctness} + w2 * A_{Diversity})
                advantages = (self.args.correctness_weight_for_specialist * advantages_correctness) + (self.args.diversity_weight_for_specialist * advantages_diversity)

                # breakpoint()

                self._metrics[mode][f"reward/{adapter_name}"].append(all_correctness_rewards[start_index:end_index].mean().item())
                self._metrics[mode][f"reward_std/{adapter_name}"].append(std_grouped_rewards.mean().item())
                self._metrics[mode][f"frac_reward_zero_std/{adapter_name}"].append(is_std_zero_cor.float().mean().item())
                self._metrics[mode][f"reward/{adapter_name}"].append(diversity_rewards.mean().item())
                self._metrics[mode][f"reward_std/{adapter_name}"].append(std_grouped_diversity.mean().item())
                self._metrics[mode][f"frac_reward_zero_std/{adapter_name}"].append(is_std_zero_div.float().mean().item())

            else:
                raise ValueError(f"Unknown adapter name: {adapter_name}")
            
            # breakpoint()
            all_advantages.append(advantages)
        
        all_advantages = torch.cat(all_advantages, dim=0)  # (total_completions,)

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))

        for adapter_index, (adapter_name, (start_index, end_index)) in enumerate(
            zip(adapter_names, adapter_boundaries)
        ):
            # breakpoint()
            if adapter_name == "default":
                func_names = self.reward_func_names_correctness
            else:
                func_names = self.reward_func_names_correctness + self.reward_func_names_diversity
            
            adapter_rewards = all_rewards_per_func[adapter_index]
            gathered_adapter_rewards = gather(adapter_rewards)
            # breakpoint()
            for i, name in enumerate(func_names):
                log_key = f"{adapter_name}/{name}"
                if log_key not in self._logs["rewards"]:
                    self._logs["rewards"][log_key] = deque(maxlen=all_advantages.size(0))
                self._logs["rewards"][log_key].extend(adapter_rewards[:, i].tolist())
                # breakpoint()
            # breakpoint()
        
        if not isinstance(self._logs["advantages"], deque):
            self._logs["advantages"] = deque(maxlen = all_advantages.size(0))
        self._logs["advantages"].extend(all_advantages.tolist())
        # breakpoint()

        if self.use_vllm and self.vllm_importance_sampling_correction:

            for adapter_index, (adapter_name, (start_index, end_index)) in enumerate(
                zip(adapter_names, adapter_boundaries)
            ):
                if old_per_token_logps is None or sampling_per_token_logps is None:
                    continue
                
                adapter_old_logps = old_per_token_logps[start_index:end_index] if old_per_token_logps is not None else None
                adapter_sampling_logps = sampling_per_token_logps[start_index:end_index] if sampling_per_token_logps is not None else None
                adapter_completion_mask = completion_mask[start_index:end_index]

                # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
                importance_sampling_ratio = torch.exp(adapter_old_logps - adapter_sampling_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )            

                delta = torch.abs(adapter_old_logps - adapter_sampling_logps)
                delta = delta[adapter_completion_mask.bool()]
                mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
                max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
                self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                    self.accelerator.gather(mean_delta).mean().item()
                )
                self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                    self.accelerator.gather(max_delta).max().item()
                )

                flat_is_ratio = importance_sampling_ratio[adapter_completion_mask.bool()]
                min_importance_sampling_ratio = (
                    torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                )
                mean_importance_sampling_ratio = (
                    torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                )
                max_importance_sampling_ratio = (
                    torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                    nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                    self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                    nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
                )

        # breakpoint()
        generation_batch["advantages"] = all_advantages
        
        return generation_batch

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
        )

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss / self.current_gradient_accumulation_steps

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_kernel:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)


    def _compute_loss(self, model, inputs):
        device = self.accelerator.device
        mode = "train" if model.training else "eval"
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        # breakpoint()
        adapter_name = inputs["adapter_info"]["adapter_names"][0]
        print(f"  >> [Loss Computation] Computing loss for adapter: {adapter_name}")
        # breakpoint()

        total_loss = 0.0

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=self.args.per_device_train_batch_size,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            adapter_name = inputs["adapter_info"]["adapter_names"][0]
            if "importance_sampling_ratio_vllm" in inputs:
                per_token_loss = per_token_loss * inputs["importance_sampling_ratio_vllm"]
            elif "importance_sampling_ratio" in inputs:
                per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        total_loss += loss

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        # start_index = end_index

        total_loss = total_loss / self.current_gradient_accumulation_steps

        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if not self._logs["prompt"] or not self._logs["completion"]:
                return
            
            # breakpoint()

            # Check for the length of all Deque
            reward_lengths = [len(v) for v in self._logs["rewards"].values()] if self._logs["rewards"] else [0]
            min_length = min(
                len(self._logs["prompt"]), 
                len(self._logs["completion"]), 
                len(self._logs["advantages"]),
                min(reward_lengths) if reward_lengths else 0
            )
            if min_length == 0:
                logger.warning("No completions to log.")
                return
            
            # breakpoint()

            flattened_rewards = {}
            for key, values in self._logs["rewards"].items():
                flattened_reward = key.replace("/", "_")
                flattened_rewards[flattened_reward] = list(values)[:min_length]

            if is_rich_available():
                print_prompt_completions_sample(
                    list(self._logs["prompt"])[:min_length],
                    list(self._logs["completion"])[:min_length],
                    flattened_rewards,
                    list(self._logs["advantages"])[:min_length],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            logging_backends = []
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                logging_backends.append(wandb)
            if self.args.report_to and "trackio" in self.args.report_to:
                logging_backends.append(trackio)

            # breakpoint()

            if logging_backends:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * min_length,
                    "prompt": list(self._logs["prompt"])[:min_length],
                    "completion": list(self._logs["completion"])[:min_length],
                    **flattened_rewards,
                    "advantage": list(self._logs["advantages"])[:min_length],
                }

                df_base = pd.DataFrame(table)
                images_raw = self._logs["images"] or []

                for logging_backend in logging_backends:
                    if images_raw:
                        # Convert images per backend and derive a dataframe that shares base columns
                        if logging_backend is wandb:
                            images = []
                            for image_list in self._logs["images"]:
                                images.append([wandb.Image(image) for image in image_list])
                            df = pd.concat([df_base, pd.Series(images, name="image")], axis=1, copy=False)
                        elif logging_backend is trackio:
                            # TODO: Implement once supported upstream https://github.com/gradio-app/trackio/issues/327
                            logger.info("Skipping image logging for Trackio")
                            df = df_base
                    else:
                        df = df_base

                    if self.wandb_log_unique_prompts:
                        df = df.drop_duplicates(subset=["prompt"])

                    logging_backend.log({"completions": logging_backend.Table(dataframe=df)})

                

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
