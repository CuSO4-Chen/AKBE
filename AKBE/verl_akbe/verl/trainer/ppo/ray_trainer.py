# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = None

WorkerType = Type[Worker]

def tensor_to_py(obj):
    """递归将 Tensor / TensorDict / list / dict 转成 JSON 可序列化结构"""
    
    # 1. torch.Tensor
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # 2. TensorDict（核心修复点）
    if TensorDict is not None and isinstance(obj, TensorDict):
        return {k: tensor_to_py(v) for k, v in obj.items()}

    # 3. 普通字典
    if isinstance(obj, dict):
        return {k: tensor_to_py(v) for k, v in obj.items()}
    
    # 4. 列表
    if isinstance(obj, list):
        return [tensor_to_py(v) for v in obj]
    
    # 5. 其他类型直接返回（int, float, str, None 等）
    return obj


def save_batch_as_parquet(batch, path: str):
    """将 batch 完整转成 JSON 字符串并保存到 Parquet，保证不会报错"""
    
    # batch.batch 是 TensorDict 或 dict
    data = tensor_to_py(batch.batch)

    # 序列化成 JSON（此时一定可序列化）
    json_str = json.dumps(data, ensure_ascii=False)

    # 保存为 Parquet 表格
    df = pd.DataFrame({"batch_json": [json_str]})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

    print(f"[Saved] batch saved to {path}")



class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def find_search_segments_by_text(resp_ids_list: list, tokenizer) -> list:
    """
    Decode token-id sequence character by character, then use text matching to
    find all <search>...</search> intervals.

    Returns:
        list of (seg_start, seg_end) tuples where
          seg_start: index of the first token of <search>
          seg_end:   index of the last  token of </search>
    """
    SEARCH_START_STR = "<search>"
    SEARCH_END_STR   = "</search>"

    pad_id    = tokenizer.pad_token_id
    valid_len = len(resp_ids_list)
    while valid_len > 0 and resp_ids_list[valid_len - 1] == pad_id:
        valid_len -= 1
    if valid_len == 0:
        return []

    # Build per-token character start positions
    token_char_starts = []
    chars = []
    for tok_id in resp_ids_list[:valid_len]:
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        token_char_starts.append(len(chars))
        chars.extend(list(tok_str))
    full_text = "".join(chars)
    token_char_starts.append(len(chars))  # sentinel

    import re
    s_starts_char = [m.start() for m in re.finditer(re.escape(SEARCH_START_STR), full_text)]
    e_starts_char = [m.start() for m in re.finditer(re.escape(SEARCH_END_STR),   full_text)]

    def char_to_token(char_pos):
        for t_idx in range(valid_len):
            if token_char_starts[t_idx] <= char_pos < token_char_starts[t_idx + 1]:
                return t_idx
        return valid_len - 1

    s_token_positions = [char_to_token(c) for c in s_starts_char]
    e_token_positions = [char_to_token(c) for c in e_starts_char]

    se_text_len = len(SEARCH_END_STR)

    def end_last_token(e_char_start):
        e_char_end = e_char_start + se_text_len - 1
        return char_to_token(e_char_end)

    segments = []
    used_end = set()
    for s_tok in s_token_positions:
        matched_e_tok  = None
        matched_e_char = None
        for e_char, e_tok in zip(e_starts_char, e_token_positions):
            if e_tok > s_tok and e_char not in used_end:
                matched_e_tok  = end_last_token(e_char)
                matched_e_char = e_char
                break
        if matched_e_tok is not None:
            used_end.add(matched_e_char)
            segments.append((s_tok, matched_e_tok))
    return segments


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # If em_score and tp_score are both available, normalize them separately
        #em_scores_tensor = None
        #tp_scores_tensor = None
        #if ("em_score" in data.non_tensor_batch and "tp_score" in data.non_tensor_batch):
            #em_scores_tensor = torch.tensor(data.non_tensor_batch["em_score"], dtype=torch.float32)
            #tp_scores_tensor = torch.tensor(data.non_tensor_batch["tp_score"], dtype=torch.float32)
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            #em_scores=em_scores_tensor,
            #tp_scores=tp_scores_tensor,
        )
        advantages = torch.clamp(advantages, min=-5.0, max=5.0)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPO:
        advantages, returns = core_algos.compute_opo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)
        
        # Initialize Dual-Path Rollout Manager for contrastive self-distillation
        from verl.trainer.ppo.dual_path_rollout_manager import DualPathRolloutManager
        dual_path_save_dir = config.get('dual_path_distillation', {}).get('save_dir', None)
        self.dual_path_manager = DualPathRolloutManager(
            config=config,
            tokenizer=tokenizer,
            save_dir=dual_path_save_dir,
        )
        if self.dual_path_manager.is_enabled():
            print(f"[DUAL-PATH] Dual-path rollout enabled. Samples will be saved to: {dual_path_save_dir}")

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            # Extract prompt_no_tool before pop to avoid chunk issues (for consistency)
            prompt_no_tool_data = None
            if "prompt_no_tool" in test_batch.non_tensor_batch:
                prompt_no_tool_data = test_batch.non_tensor_batch["prompt_no_tool"]
            
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            # Do NOT add prompt_no_tool to avoid chunk issues
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            
            # Add prompt_no_tool back after pop (for consistency)
            if prompt_no_tool_data is not None:
                test_gen_batch.non_tensor_batch["prompt_no_tool"] = prompt_no_tool_data

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True, is_validate=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            data_source_lst.append(data_source)
            reward_extra_infos_dict["data_source"].extend(data_source) 

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        # Calculate TC (average tool calls) and TP (correct samples / total tool calls)
        import re
        tool_call_counts = []
        for output_text in sample_outputs:
            # Count <search> tags in each output
            count = len(re.findall(r'<search>', output_text, re.IGNORECASE))
            tool_call_counts.append(count)
        
        total_tool_calls = sum(tool_call_counts)
        num_samples = len(sample_outputs)
        
        # TC: average tool calls per sample
        tc_metric = total_tool_calls / num_samples if num_samples > 0 else 0.0
        
        # TP: correct samples / total tool calls
        # Get em_score from reward_extra_infos_dict if available
        correct_samples = 0
        if "em_score" in reward_extra_infos_dict and len(reward_extra_infos_dict["em_score"]) > 0:
            em_scores = reward_extra_infos_dict["em_score"]
            correct_samples = sum([1 for score in em_scores if score == 1.0])
        elif "acc" in reward_extra_infos_dict and len(reward_extra_infos_dict["acc"]) > 0:
            # Fallback to acc if em_score is not available
            acc_scores = reward_extra_infos_dict["acc"]
            correct_samples = sum([1 for score in acc_scores if score == 1.0])
        
        tp_metric = correct_samples / total_tool_calls if total_tool_calls > 0 else 0.0

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        
        # Add TC and TP metrics to metric_dict
        metric_dict["val-core/TC"] = tc_metric
        metric_dict["val-core/TP"] = tp_metric
        
        # Also add detailed statistics for debugging
        metric_dict["val-aux/total_tool_calls"] = total_tool_calls
        metric_dict["val-aux/correct_samples"] = correct_samples
        metric_dict["val-aux/num_samples"] = num_samples

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _pad_batch_to_match(self, batch1, batch2):
        """
        对齐两个DataProto batch的序列长度，使它们可以安全地concat
        
        Args:
            batch1: 第一个DataProto batch (可能是None)
            batch2: 第二个DataProto batch
            
        Returns:
            对齐后的 (batch1, batch2)
        """
        if batch1 is None:
            return batch1, batch2
        
        # 检测需要对齐的所有tensor及其目标长度
        alignment_specs = {}
        
        for key in batch1.batch.keys():
            if key in batch2.batch:
                tensor1 = batch1.batch[key]
                tensor2 = batch2.batch[key]
                
                if hasattr(tensor1, 'shape') and hasattr(tensor2, 'shape') and len(tensor1.shape) >= 2:
                    if tensor1.shape[1] != tensor2.shape[1]:
                        max_len = max(tensor1.shape[1], tensor2.shape[1])
                        alignment_specs[key] = max_len
        
        # 如果需要对齐，则执行padding
        if alignment_specs:
            #print(f"Aligning tensors: {list(alignment_specs.keys())}")
            batch1 = self._pad_batch(batch1, alignment_specs)
            batch2 = self._pad_batch(batch2, alignment_specs)
    
        return batch1, batch2


    def _pad_batch(self, batch, alignment_specs):
        """
        对DataProto batch中的tensor进行padding

        Args:
            batch: DataProto对象
            alignment_specs: dict，key为tensor名称，value为目标长度
            
        Returns:
            padding后的DataProto对象
        """
        #padded_batch = batch.clone()
        padded_batch = batch

        for key, target_len in alignment_specs.items():
            if key not in padded_batch.batch:
                continue
                
            tensor = padded_batch.batch[key]
            if not hasattr(tensor, 'shape') or len(tensor.shape) < 2:
                continue
                
            current_len = tensor.shape[1]
            if current_len >= target_len:
                continue
            
            pad_size = target_len - current_len
            
            # 根据tensor名称确定padding策略
            if key in ['input_ids', 'prompts']:
                # input_ids用0 (pad_token_id) padding
                pad_value = 0
                pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)
                
            elif key in ['attention_mask', 'loss_mask']:
                # attention_mask用0 padding (表示padding位置)
                pad_value = 0
                pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)
                
            elif key == 'position_ids':
                # position_ids需要连续递增
                last_positions = tensor[:, -1:]  # [batch_size, 1]
                # 从last_position+1开始递增
                increments = torch.arange(1, pad_size + 1, device=tensor.device).unsqueeze(0)  # [1, pad_size]
                padding = last_positions + increments  # [batch_size, pad_size]
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)
                
            elif key in ['responses', 'step_mask']:
                # responses和step_mask用0 padding
                pad_value = 0
                pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)
                
            elif key in ['token_level_scores', 'token_level_rewards']:
                # reward相关的tensor用0.0 padding
                pad_value = 0.0
                pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)
                
            elif key == 'tool_use_scores':
                # tool_use_scores维度是[batch_size, 2]，不需要在序列维度padding
                continue
                
            else:
                # 其他tensor使用默认padding值
                pad_value = 0.0 if tensor.dtype.is_floating_point else 0
                pad_shape = [tensor.shape[0], pad_size] + list(tensor.shape[2:])
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_batch.batch[key] = torch.cat([tensor, padding], dim=1)

        return padded_batch

    def _debug_inspect_batch_log_probs(self, batch, batch_name="Unknown"):
        """
        深度诊断 LogProbs 的状态，用于确认 Reference Model 是否失效。
        """
        print(f"\n{'='*20} [DEBUG: {batch_name}] {'='*20}")
        
        # 1. 检查 Key 是否存在
        if batch is None or batch.batch is None:
            print(f"  [ERROR] Batch object is None!")
            return

        keys = list(batch.batch.keys())
        if 'old_log_probs' not in keys:
            print(f"  [CRITICAL ALERT] 'old_log_probs' NOT FOUND! Available keys: {keys}")
            # 尝试查找其他可能的 key
            if 'log_probs' in keys:
                print(f"  [INFO] Found 'log_probs' instead. Using that.")
                lp = batch.batch['log_probs']
            else:
                print(f"  [STOP] No log probability field found.")
                return
        else:
            lp = batch.batch['old_log_probs']

        # 2. 检查数值状态
        # 确保转为 float 进行统计，防止溢出
        lp_flat = lp.flatten().float()
        total_count = lp_flat.numel()
        zero_count = (lp_flat == 0).sum().item()
        non_zero_count = total_count - zero_count
        
        print(f"  Shape: {lp.shape}")
        print(f"  Total Tokens: {total_count}")
        print(f"  Zero Tokens:  {zero_count} ({zero_count/total_count*100:.1f}%)")
        
        # 3. 核心判断
        if non_zero_count == 0:
            print(f"  [CRITICAL FAILURE] ALL log_probs are ZERO! Reference Model is ineffective.")
        else:
            # 统计非零值的分布
            valid_values = lp_flat[lp_flat != 0]
            mean_val = valid_values.mean().item()
            min_val = valid_values.min().item()
            max_val = valid_values.max().item()
            
            print(f"  [STATUS: OK] Found {non_zero_count} non-zero values.")
            print(f"  Statistics (Non-Zero): Mean={mean_val:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")
            print(f"  Sample values: {valid_values[:5].tolist()}")
        
        print(f"{'='*50}\n")


    def _construct_sdpo_batch(self, with_tool_batch, no_tool_batch):
        """
        逻辑：
        1. Case Efficiency: NT正确, WT也正确 -> Teacher=NT (选择最短的正确NT)
        2. Case Hallucination: NT正确, WT全错 -> Teacher=NT (选择最短的正确NT)
        3. Case Capability: WT正确, NT全错 -> Teacher=WT (选择最短的正确WT)
        """
        from torch.nn.utils.rnn import pad_sequence
        from tensordict import TensorDict

        print(f"\n[SDPO] Constructing teacher trajectories...")
        
        stats = {
            "sdpo/teacher_total": 0,
            "sdpo/teacher_efficiency": 0,    # NT 对，WT 也对
            "sdpo/teacher_hallucination": 0, # NT 对，WT 错
            "sdpo/teacher_capability": 0,    # WT 对，NT 错
            "sdpo/internalization_ratio": 0.0,   # (Efficiency + Hallucination) / Total Teachers
            "sdpo/no_tool_correct_mean": 0.0,    # 新增：no-tool轨迹答对均值
        }

        # 检查分数是否存在
        if hasattr(with_tool_batch, 'non_tensor_batch') and 'em_score' in with_tool_batch.non_tensor_batch:
            wt_em_scores = torch.tensor(with_tool_batch.non_tensor_batch['em_score'], device="cpu", dtype=torch.float32)
        else:
            print("[SDPO] Error: 'em_score' not found in with_tool_batch")
            return None, stats
            
        if hasattr(no_tool_batch, 'non_tensor_batch') and 'em_score' in no_tool_batch.non_tensor_batch:
            nt_em_scores = torch.tensor(no_tool_batch.non_tensor_batch['em_score'], device="cpu", dtype=torch.float32)
        else:
            print("[SDPO] Error: 'em_score' not found in no_tool_batch")
            return None, stats
        
        # 参数
        batch_size = with_tool_batch.batch.batch_size[0] // self.config.actor_rollout_ref.rollout.n 
        rollout_n = self.config.actor_rollout_ref.rollout.n 
        no_tool_rollout_n = self.config.algorithm.get("no_tool_rollout_n", 4) 

        # 提取 Prompt 的padded长度 (用于构建 loss_mask)
        # 注意: 序列布局是 [LEFT_PAD ... PROMPT_TOKENS ... | RESPONSE_TOKENS ... RIGHT_PAD]
        # padded_prompt_length 是 prompt 区域的固定长度（包含 left padding），response 从此位置开始
        if 'prompts' in with_tool_batch.batch.keys():
            padded_prompt_length = with_tool_batch.batch['prompts'].shape[1]
        else:
            print("[SDPO] Warning: Could not find 'prompts' to compute loss_mask. "
                  "Will assume loss is computed over all non-pad tokens, which might include the prompt!")
            padded_prompt_length = 0

        wt_masks = with_tool_batch.batch['attention_mask']
        nt_masks = no_tool_batch.batch['attention_mask']

        # 容器：只装 Teacher
        teacher_input_ids = []
        teacher_attention_mask = []
        teacher_loss_mask = []
        # 新增：记录每个选中的teacher对应的no-tool正确样本数量
        no_tool_correct_counts = []

        for i in range(batch_size):
            # 切片获取当前 Prompt 的数据
            wt_start, wt_end = i * rollout_n, (i + 1) * rollout_n
            nt_start, nt_end = i * no_tool_rollout_n, (i + 1) * no_tool_rollout_n
            
            cur_wt_em = wt_em_scores[wt_start:wt_end]
            cur_nt_em = nt_em_scores[nt_start:nt_end]
            
            # 计算当前prompt的no-tool正确样本数量
            nt_correct_count = (cur_nt_em >= 1.0).sum().item()
            
            # 判定正确性
            wt_correct_indices = (cur_wt_em >= 1.0).nonzero(as_tuple=True)[0]
            nt_correct_indices = (cur_nt_em >= 1.0).nonzero(as_tuple=True)[0]
            nt_wrong_indices = (cur_nt_em < 1.0).nonzero(as_tuple=True)[0]

            teacher_global_idx = None
            source_batch = None 

            cur_wt_lens = wt_masks[wt_start:wt_end].sum(dim=-1)
            cur_nt_lens = nt_masks[nt_start:nt_end].sum(dim=-1)
            
            # 优先级 1: No-Tool 正确
            if len(nt_correct_indices) > 0:
                valid_lens = cur_nt_lens[nt_correct_indices]
                best_local_idx = nt_correct_indices[torch.argmin(valid_lens)].item()
                teacher_global_idx = nt_start + best_local_idx
                source_batch = no_tool_batch
                # --- 分开统计 Efficiency 和 Hallucination ---
                if len(wt_correct_indices) > 0:
                    stats["sdpo/teacher_efficiency"] += 1
                else:
                    stats["sdpo/teacher_hallucination"] += 1

            # 优先级 2: No-Tool 全错, 但 With-Tool 正确 (Capability 场景)
            # [ABLATION] 当 sdpo_disable_capability=True 时跳过此 case，
            # 仅保留 Efficiency + Hallucination，验证 IG 信号已吸收 Capability 的梯度贡献
            elif not getattr(self, '_sdpo_disable_capability', False) and \
                    len(wt_correct_indices) > 0 and len(nt_wrong_indices) > 0:
                valid_wt_lens = cur_wt_lens[wt_correct_indices]
                best_local_idx = wt_correct_indices[torch.argmin(valid_wt_lens)].item()
                teacher_global_idx = wt_start + best_local_idx
                source_batch = with_tool_batch
                stats["sdpo/teacher_capability"] += 1

            # --- 提取数据 ---
            if teacher_global_idx is not None:
                # 1. Input IDs
                ids = source_batch.batch['input_ids'][teacher_global_idx]
                teacher_input_ids.append(ids)
                
                # 2. Attention Mask
                att_mask = source_batch.batch['attention_mask'][teacher_global_idx]
                teacher_attention_mask.append(att_mask)
                
                # 3. 构建 Loss Mask: 仅 Response 部分为 1
                # 序列布局: [LEFT_PAD ... PROMPT ... | RESPONSE ... RIGHT_PAD]
                # padded_prompt_length 是 prompt 区域的固定长度，response 从此位置开始
                # 对于所有 teacher: 先用 att_mask 获取非 pad 位置，然后将 prompt 区域清零
                l_mask = att_mask.clone()
                if padded_prompt_length > 0:
                    l_mask[:padded_prompt_length] = 0  # 清零整个 prompt 区域（包括 left padding 和 prompt tokens）

                # 对于 WT teacher（Capability case），还需用 rollout 的 loss_mask 排除工具返回 token
                if source_batch is with_tool_batch and 'loss_mask' in source_batch.batch:
                    wt_loss_mask = source_batch.batch['loss_mask'][teacher_global_idx]
                    # wt_loss_mask 形状是 [response_length]，与 response 区域对齐
                    response_region = l_mask[padded_prompt_length:]
                    min_len = min(len(response_region), len(wt_loss_mask))
                    # 用 element-wise 乘法将工具返回 token 的位置清零
                    response_region[:min_len] *= wt_loss_mask[:min_len]

                teacher_loss_mask.append(l_mask)
                
                # 4. 记录当前teacher对应的no-tool正确样本数量
                no_tool_correct_counts.append(nt_correct_count)

        total_teachers = len(teacher_input_ids)
        if total_teachers == 0:
            print("[SDPO] Warning: No teacher trajectories found in this batch.")
            return None, stats

        stats["sdpo/teacher_total"] = total_teachers
        
        # 计算 efficiency + hallucination 占 teacher_total 的比例
        if total_teachers > 0:
            internalization_count = stats["sdpo/teacher_efficiency"] + stats["sdpo/teacher_hallucination"]
            stats["sdpo/internalization_ratio"] = internalization_count / total_teachers
        else:
            stats["sdpo/internalization_ratio"] = 0.0
        
        # 计算 no-tool轨迹答对均值
        if len(no_tool_correct_counts) > 0:
            stats["sdpo/no_tool_correct_mean"] = sum(no_tool_correct_counts) / len(no_tool_correct_counts)
        else:
            stats["sdpo/no_tool_correct_mean"] = 0.0
        
        # 打印方便看 Log
        print(f"[SDPO] Total Teachers: {stats['sdpo/teacher_total']} | "
              f"Eff: {stats['sdpo/teacher_efficiency']} | "
              f"Hallu: {stats['sdpo/teacher_hallucination']} | "
              f"Cap: {stats['sdpo/teacher_capability']} | "
              f"Internalization Ratio: {stats['sdpo/internalization_ratio']:.3f} | "
              f"No-Tool Correct Mean: {stats['sdpo/no_tool_correct_mean']:.3f}")
        print(f"[SDPO] padded_prompt_length={padded_prompt_length}")
        # Debug: 检查每个 teacher 的 loss_mask 非零数量
        for t_idx, lm in enumerate(teacher_loss_mask):
            nonzero_count = lm.sum().item()
            if nonzero_count == 0:
                print(f"[SDPO] WARNING: teacher {t_idx} has all-zero loss_mask! len={len(lm)}")

        # 2. Padding
        pad_val = self.tokenizer.pad_token_id
        
        ids_padded = pad_sequence(teacher_input_ids, batch_first=True, padding_value=pad_val)
        att_mask_padded = pad_sequence(teacher_attention_mask, batch_first=True, padding_value=0)
        l_mask_padded = pad_sequence(teacher_loss_mask, batch_first=True, padding_value=0)
        
        # 对齐到 16 或 32 的整数倍
        max_len = ids_padded.shape[1]
        ALIGN_BASE = 32  
        if max_len % ALIGN_BASE != 0:
            pad_len = ALIGN_BASE - (max_len % ALIGN_BASE)
            max_len += pad_len
        
        def pad_tensor(t, target_len, val):
            if t.shape[1] < target_len:
                pad_size = target_len - t.shape[1]
                return torch.nn.functional.pad(t, (0, pad_size), value=val)
            return t
            
        final_ids = pad_tensor(ids_padded, max_len, pad_val)
        final_att_mask = pad_tensor(att_mask_padded, max_len, 0)
        final_loss_mask = pad_tensor(l_mask_padded, max_len, 0)

        # 构建 SDPO Batch
        sdpo_batch_dict = {
            "input_ids": final_ids,
            "attention_mask": final_att_mask,
            "loss_mask": final_loss_mask,
        }
        
        # 只有当有teacher被选中时才添加no_tool_correct_counts
        if len(no_tool_correct_counts) > 0:
            sdpo_batch_dict["no_tool_correct_counts"] = torch.tensor(no_tool_correct_counts, dtype=torch.float32)
        
        return TensorDict(sdpo_batch_dict, batch_size=[total_teachers]), stats


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.gen_steps += 1
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        filtered_out_batches = []

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                # Do NOT add prompt_no_tool to avoid chunk issues
                gen_batch = new_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                
                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                #gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)


                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch (with-tool rollout, 64*12)
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                        if gen_batch_output.meta_info and "metrics" in gen_batch_output.meta_info:
                            metrics.update(gen_batch_output.meta_info["metrics"])
                    
                    # [DUAL-PATH] Generate no-tool rollout (64*1) for contrastive self-distillation
                    # This rollout is saved for analysis but does NOT participate in PPO training
                    if self.dual_path_manager.is_enabled():
                        with _timer("gen_no_tool", timing_raw):
                            print(f"\n{'='*80}")
                            print(f"[DUAL-PATH] Step {self.global_steps}: Starting no-tool rollout generation")
                            print(f"{'='*80}")
                            
                            # CRITICAL: Pass target_max_length to ensure padding matches with-tool samples
                            target_max_length = gen_batch.batch['input_ids'].shape[1]
                            print(f"[DUAL-PATH] With-tool input_ids shape: {gen_batch.batch['input_ids'].shape}")
                            print(f"[DUAL-PATH] Target max length for no-tool padding: {target_max_length}")
        
                            rollout_n = self.config.actor_rollout_ref.rollout.n
                            no_tool_rollout_n = self.config.algorithm.get("no_tool_rollout_n", -1)

                            # Create no-tool prompts by splitting merged prompts (using SEPARATOR)
                            no_tool_gen_batch = self.dual_path_manager.create_no_tool_prompts(
                                gen_batch, 
                                target_max_length=target_max_length,
                            )

                            if no_tool_gen_batch.meta_info is None:
                                    no_tool_gen_batch.meta_info = {}
                            no_tool_gen_batch.meta_info['n'] = no_tool_rollout_n
                            no_tool_gen_batch.meta_info['sampling_params'] = {'n': no_tool_rollout_n}
                            # Disable tool call detection for no-tool rollouts to prevent
                            # false positive stop_sequence matches causing vLLM hangs
                            no_tool_gen_batch.meta_info['disable_tool_detection'] = True
                            
                            if no_tool_gen_batch is not None:
                                print(f"[DUAL-PATH] No-tool prompts created")
                                print(f"[DUAL-PATH] No-tool input_ids shape: {no_tool_gen_batch.batch['input_ids'].shape}")
                                
                                # Generate no-tool responses
                                # Note: generate_sequences() does not support rollout_n parameter
                                # and Ray Actor config cannot be modified dynamically.
                                # So we generate 64*12 samples and select only the first rollout for each prompt.
                                if not self.async_rollout_mode:
                                    no_tool_gen_batch_output_full = self.actor_rollout_wg.generate_sequences(no_tool_gen_batch)
                                else:
                                    self.async_rollout_manager.wake_up()
                                    no_tool_gen_batch_output_full = self.async_rollout_manager.generate_sequences(no_tool_gen_batch)
                                    self.async_rollout_manager.sleep()
                                
                                print(f"[DUAL-PATH] Generated shapes (before selection):")
                                print(f"  - Prompts: {no_tool_gen_batch.batch['input_ids'].shape}")
                                print(f"  - Responses: {no_tool_gen_batch_output_full.batch['responses'].shape}")
                                
                                # Get configuration for no-tool rollout selection
                                # no_tool_rollout_n: number of rollouts to use for each prompt (1 to rollout_n)
                                # - If not set or set to -1: use all rollouts (rollout_n)
                                # - If set to 1: use only the first rollout (original behavior)
                                # - If set to N (1 < N < rollout_n): use first N rollouts
                                
                                # Validate and adjust no_tool_rollout_n
                                if no_tool_rollout_n == -1 or no_tool_rollout_n > rollout_n:
                                    no_tool_rollout_n = rollout_n  # Use all rollouts
                                elif no_tool_rollout_n < 1:
                                    no_tool_rollout_n = 1  # Use at least 1 rollout
                                
                                batch_size = no_tool_gen_batch.batch['input_ids'].shape[0]
                                
                                print(f"[DUAL-PATH] No-tool rollout configuration:")
                                print(f"  - With-tool rollouts per prompt: {rollout_n}")
                                print(f"  - No-tool rollouts per prompt: {no_tool_rollout_n}")
                                print(f"  - No-tool samples: {batch_size * no_tool_rollout_n} (batch_size={batch_size})")
                               
                                no_tool_gen_batch_output = no_tool_gen_batch_output_full

                                # Expand no_tool_gen_batch to match the number of selected rollouts
                                # We need to repeat each prompt no_tool_rollout_n times to match the responses
                                print(f"[DUAL-PATH] Expanding no_tool_gen_batch to match no_tool_rollout_n={no_tool_rollout_n}")
                                expanded_batch_dict = {}
                                for key in no_tool_gen_batch.batch.keys():
                                    tensor = no_tool_gen_batch.batch[key]
                                    # Repeat each prompt no_tool_rollout_n times: [p0, p0, ..., p0 (Nx), p1, p1, ..., p1 (Nx), ...]
                                    expanded_tensor = tensor.repeat_interleave(no_tool_rollout_n, dim=0)
                                    expanded_batch_dict[key] = expanded_tensor
                                    print(f"[DUAL-PATH]   {key}: {tensor.shape} -> {expanded_tensor.shape}")
                                
                                # Create a new TensorDict with expanded tensors (instead of updating existing one)
                                # Note: TensorDict is already imported at the top of the file
                                expanded_batch = TensorDict(expanded_batch_dict, batch_size=[expanded_batch_dict[list(expanded_batch_dict.keys())[0]].shape[0]])
                                
                                # Also expand non_tensor_batch if needed
                                expanded_non_tensor = {}
                                if hasattr(no_tool_gen_batch, 'non_tensor_batch'):
                                    for key in no_tool_gen_batch.non_tensor_batch.keys():
                                        data = no_tool_gen_batch.non_tensor_batch[key]
                                        # Repeat each element no_tool_rollout_n times
                                        expanded_data = np.repeat(data, no_tool_rollout_n, axis=0)
                                        expanded_non_tensor[key] = expanded_data
                                        print(f"[DUAL-PATH]   non_tensor[{key}]: {len(data)} -> {len(expanded_data)}")
                                
                                # Create a new DataProto with expanded batch and non_tensor_batch
                                # Note: DataProto is already imported at the top of the file
                                no_tool_gen_batch = DataProto(
                                    batch=expanded_batch,
                                    non_tensor_batch=expanded_non_tensor,
                                    meta_info=no_tool_gen_batch.meta_info
                                )
                                
                                print(f"[DUAL-PATH] Generated shapes (after selection):")
                                print(f"  - Prompts: {no_tool_gen_batch.batch['input_ids'].shape}")
                                print(f"  - Responses: {no_tool_gen_batch_output.batch['responses'].shape}")
                                
                                # Verify shapes match
                                assert no_tool_gen_batch.batch['input_ids'].shape[0] == no_tool_gen_batch_output.batch['responses'].shape[0], \
                                    f"Shape mismatch after selection: prompts={no_tool_gen_batch.batch['input_ids'].shape[0]}, responses={no_tool_gen_batch_output.batch['responses'].shape[0]}"
                                
                                # CRITICAL: Copy prompt fields from no_tool_gen_batch to no_tool_gen_batch_output
                                # This ensures that union operation works correctly (requires same object for common keys)
                                # The prompt fields (input_ids, attention_mask, position_ids) from no_tool_gen_batch
                                # are padded to max_prompt_length (2000)
                                # After union, they will be concatenated with responses to form full sequences (8192)
                                
                                # First, remove any existing prompt fields from no_tool_gen_batch_output
                                # (they may come from with-tool rollout and have wrong values)
                                prompt_fields = ['input_ids', 'attention_mask', 'position_ids']
                                for field in prompt_fields:
                                    if field in no_tool_gen_batch_output.batch:
                                        print(f"[DUAL-PATH] Removing existing {field} from no_tool_gen_batch_output (shape: {no_tool_gen_batch_output.batch[field].shape})")
                                        del no_tool_gen_batch_output.batch[field]
                                
                                # Then, copy prompt fields from no_tool_gen_batch
                                for field in prompt_fields:
                                    if field in no_tool_gen_batch.batch:
                                        no_tool_gen_batch_output.batch[field] = no_tool_gen_batch.batch[field]
                                        print(f"[DUAL-PATH] Copied {field} from no_tool_gen_batch (shape: {no_tool_gen_batch.batch[field].shape})")
                                
                                # ========== CRITICAL FIX: Save original prompt tensors BEFORE union ==========
                                # The union operation modifies no_tool_gen_batch IN-PLACE, so we must save
                                # the original prompt tensors before calling union
                                
                                print(f"[DUAL-PATH] Saving original prompt tensors before union...")
                                original_prompt_input_ids = no_tool_gen_batch.batch['input_ids'].clone()
                                original_prompt_attention_mask = no_tool_gen_batch.batch['attention_mask'].clone()
                                original_prompt_position_ids = no_tool_gen_batch.batch['position_ids'].clone()
                                
                                # ========== CRITICAL FIX: Save expanded non_tensor_batch BEFORE union ==========
                                # The union operation will overwrite non_tensor_batch with no_tool_gen_batch_output's version
                                # But we need to keep the expanded version (384 samples instead of 64)
                                print(f"[DUAL-PATH] Saving expanded non_tensor_batch before union...")
                                expanded_non_tensor_backup = no_tool_gen_batch.non_tensor_batch.copy()
                                print(f"[DUAL-PATH] Backed up non_tensor_batch with {len(list(expanded_non_tensor_backup.values())[0]) if expanded_non_tensor_backup else 0} samples")
                                
                                
                                # Union prompts with generated responses
                                # NOTE: The union operation modifies no_tool_gen_batch IN-PLACE!
                                # After this line, no_tool_gen_batch.batch['input_ids'] will contain the full sequence
                                # AND no_tool_gen_batch.non_tensor_batch will be overwritten!
                                no_tool_full_batch = no_tool_gen_batch.union(no_tool_gen_batch_output)
                                
                                # ========== CRITICAL FIX: Restore expanded non_tensor_batch AFTER union ==========
                                # Union operation overwrites non_tensor_batch, so we need to restore it
                                print(f"[DUAL-PATH] Restoring expanded non_tensor_batch after union...")
                                no_tool_full_batch.non_tensor_batch = expanded_non_tensor_backup
                                print(f"[DUAL-PATH] Restored non_tensor_batch with {len(list(no_tool_full_batch.non_tensor_batch.values())[0]) if no_tool_full_batch.non_tensor_batch else 0} samples")
                                
                                
                                # ========== FIX: Construct full sequences (prompt + response) ==========
                                print(f"[DUAL-PATH] Constructing full sequences to match with-tool shape...")
                                
                                # Use the saved original prompt tensors (NOT the polluted ones from union)
                                prompt_input_ids = original_prompt_input_ids
                                prompt_attention_mask = original_prompt_attention_mask
                                prompt_position_ids = original_prompt_position_ids
                                
                                # Get response tensors
                                response_ids = no_tool_gen_batch_output.batch['responses']  # [batch_size, max_response_length]
                                
                                batch_size = prompt_input_ids.shape[0]
                                prompt_length = prompt_input_ids.shape[1]
                                response_length = response_ids.shape[1]
                                
                                print(f"[DUAL-PATH] Before concatenation:")
                                print(f"  - prompt_input_ids: {prompt_input_ids.shape}")
                                print(f"  - response_ids: {response_ids.shape}")
                                
                                # Concatenate input_ids: [batch_size, prompt_length + response_length]
                                full_input_ids = torch.cat([prompt_input_ids, response_ids], dim=1)

                                eos_token_id = self.tokenizer.eos_token_id
                                pad_token_id = self.tokenizer.pad_token_id
                                
                                # Concatenate attention_mask (response part should all be 1)
                                response_attention_mask = torch.zeros_like(response_ids)

                                for i in range(response_ids.size(0)):
                                    # 寻找 EOS 位置
                                    eos_indices = (response_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
                                    if len(eos_indices) > 0:
                                        # 如果有 EOS，Mask 到第一个 EOS 为止
                                        first_eos_idx = eos_indices[0].item()
                                        response_attention_mask[i, :first_eos_idx + 1] = 1
                                    else:
                                        # 如果没有 EOS (被截断)，或者全是 Pad
                                        if pad_token_id is not None and pad_token_id != eos_token_id:
                                            # 把非 Pad 的地方设为 1
                                            response_attention_mask[i] = (response_ids[i] != pad_token_id).long()
                                        else:
                                            # 兜底：全是 1
                                            response_attention_mask[i] = 1

                                full_attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
                                
                                # Construct position_ids for full sequence
                                full_position_ids = []
                                for i in range(batch_size):
                                    # Get the last valid position from prompt
                                    prompt_pos = prompt_position_ids[i]
                                    prompt_mask = prompt_attention_mask[i]
                                    
                                    # Find last valid position in prompt
                                    valid_positions = prompt_pos[prompt_mask.bool()]
                                    if len(valid_positions) > 0:
                                        last_prompt_pos = valid_positions[-1].item()
                                    else:
                                        last_prompt_pos = -1
                                    
                                    # Create position_ids for response (continue from last prompt position)
                                    response_pos = torch.arange(
                                        last_prompt_pos + 1, 
                                        last_prompt_pos + 1 + response_length,
                                        device=prompt_pos.device,
                                        dtype=prompt_pos.dtype
                                    )
                                    
                                    # Concatenate prompt and response position_ids
                                    full_pos = torch.cat([prompt_pos, response_pos], dim=0)
                                    full_position_ids.append(full_pos)
                                
                                
                                # Update no_tool_full_batch with full sequences
                                no_tool_full_batch.batch['input_ids'] = full_input_ids
                                no_tool_full_batch.batch['attention_mask'] = full_attention_mask
                                no_tool_full_batch.batch['position_ids'] = torch.stack(full_position_ids)
                                
                                print(f"[DUAL-PATH] No-tool rollout completed")
                                print(f"[DUAL-PATH] With-tool samples: {len(gen_batch_output.batch['responses'])}")
                                print(f"[DUAL-PATH] No-tool samples: {len(no_tool_gen_batch_output.batch['responses'])}")

                                if self.dual_path_manager.is_enabled():
                                    # ========== Verify tensor shapes ==========
                                    print(f"[DUAL-PATH] Verifying tensor shapes...")
                                    print(f"[DUAL-PATH] input_ids: {no_tool_full_batch.batch['input_ids'].shape}")
                                    print(f"[DUAL-PATH] attention_mask: {no_tool_full_batch.batch['attention_mask'].shape}")
                                    print(f"[DUAL-PATH] responses: {no_tool_full_batch.batch['responses'].shape}")
                                    
                                    # Expected shapes (based on with-tool):
                                    # - input_ids, attention_mask, position_ids: [batch_size, 8192] (max_prompt_length + max_response_length)
                                    # - responses: [batch_size, 6192] (max_response_length)
                                    # All shapes should now be correct after padding in create_no_tool_prompts

                                    # The reward function requires these fields in non_tensor_batch:
                                    # 1. reward_model: {ground_truth, ...}
                                    # 2. data_source (or other reward_fn_key)
                                    
                                    print(f"[DUAL-PATH] Checking and copying required fields...")
                                    required_fields = ["reward_model", "data_source"]
                                    
                                    # Use gen_batch as source (it has the original non_tensor_batch)
                                    for field in required_fields:
                                        if field not in no_tool_full_batch.non_tensor_batch:
                                            if hasattr(gen_batch, 'non_tensor_batch') and field in gen_batch.non_tensor_batch:
                                                # CRITICAL FIX: Need to expand the field to match no_tool_rollout_n
                                                original_field_data = gen_batch.non_tensor_batch[field]
                                                expanded_field_data = np.repeat(original_field_data, no_tool_rollout_n, axis=0)
                                                no_tool_full_batch.non_tensor_batch[field] = expanded_field_data
                                                print(f"[DUAL-PATH] ✓ Copied and expanded '{field}' from gen_batch: {len(original_field_data)} -> {len(expanded_field_data)}")
                                            elif field in new_batch.non_tensor_batch:
                                                # CRITICAL FIX: Need to expand the field to match no_tool_rollout_n
                                                original_field_data = new_batch.non_tensor_batch[field]
                                                expanded_field_data = np.repeat(original_field_data, no_tool_rollout_n, axis=0)
                                                no_tool_full_batch.non_tensor_batch[field] = expanded_field_data
                                                print(f"[DUAL-PATH] ✓ Copied and expanded '{field}' from new_batch: {len(original_field_data)} -> {len(expanded_field_data)}")
                                            else:
                                                print(f"[DUAL-PATH] ✗ WARNING: '{field}' not found in any batch!")
                                    
                                    
                                    print(f"[DUAL-PATH] Final non_tensor_batch keys: {list(no_tool_full_batch.non_tensor_batch.keys())}")
                                    
                                    
                                    # Add UIDs for no-tool samples
                                    no_tool_full_batch.non_tensor_batch["uid"] = np.array(
                                        [str(uuid.uuid4()) for _ in range(len(no_tool_full_batch.batch))], dtype=object
                                    )
                                    
                                    # Compute reward for no-tool samples
                                    print(f"[DUAL-PATH] Computing rewards for no-tool samples...")
                                    if self.use_rm:
                                        no_tool_reward_tensor = self.rm_wg.compute_rm_score(no_tool_full_batch)
                                        no_tool_full_batch = no_tool_full_batch.union(no_tool_reward_tensor)
                                    
                                    no_tool_reward_tensor, no_tool_reward_extra_infos = compute_reward(no_tool_full_batch, self.reward_fn)
                                    no_tool_full_batch.batch["token_level_scores"] = no_tool_reward_tensor
                                    if no_tool_reward_extra_infos:
                                        no_tool_full_batch.non_tensor_batch.update({k: np.array(v) for k, v in no_tool_reward_extra_infos.items()})
                                    
                                    # Apply KL penalty if needed
                                    if self.config.algorithm.use_kl_in_reward:
                                        no_tool_full_batch, _ = apply_kl_penalty(
                                            no_tool_full_batch, 
                                            kl_ctrl=self.kl_ctrl_in_reward, 
                                            kl_penalty=self.config.algorithm.kl_penalty
                                        )
                                    else:
                                        no_tool_full_batch.batch["token_level_rewards"] = no_tool_full_batch.batch["token_level_scores"]
                                    
                                    # CRITICAL: Compute old_log_probs for no-tool samples
                                    # This is REQUIRED by update_actor (see dp_actor.py line 442)
                                    # The old_log_probs field must be present in the batch
                                    print(f"[DUAL-PATH] Computing old_log_probs for no-tool samples...")
                                    no_tool_old_log_prob = self.actor_rollout_wg.compute_log_prob(no_tool_full_batch)
                                    
                                    # Remove entropys if present (not needed for training)
                                    if "entropys" in no_tool_old_log_prob.batch:
                                        no_tool_old_log_prob.batch.pop("entropys")
                                    
                                    no_tool_full_batch = no_tool_full_batch.union(no_tool_old_log_prob)
                                    print(f"[DUAL-PATH] old_log_probs computed, shape: {no_tool_full_batch.batch['old_log_probs'].shape}")
                                    
                                    # Verify and fix old_log_probs shape to match responses
                                    expected_response_len = no_tool_full_batch.batch['responses'].shape[1]
                                    actual_old_log_prob_len = no_tool_full_batch.batch['old_log_probs'].shape[1]
                                    
                                    if actual_old_log_prob_len != expected_response_len:
                                        print(f"[DUAL-PATH] WARNING: old_log_probs length mismatch!")
                                        print(f"[DUAL-PATH] Expected: {expected_response_len}, Got: {actual_old_log_prob_len}")
                                        
                                        # Pad old_log_probs to match responses length
                                        if actual_old_log_prob_len < expected_response_len:
                                            pad_size = expected_response_len - actual_old_log_prob_len
                                            print(f"[DUAL-PATH] Padding old_log_probs by {pad_size}")
                                            no_tool_full_batch.batch['old_log_probs'] = torch.nn.functional.pad(
                                                no_tool_full_batch.batch['old_log_probs'], 
                                                (0, pad_size), 
                                                value=0
                                            )
                                        else:
                                            print(f"[DUAL-PATH] Truncating old_log_probs to {expected_response_len}")
                                            no_tool_full_batch.batch['old_log_probs'] = no_tool_full_batch.batch['old_log_probs'][:, :expected_response_len]
                                        
                                        print(f"[DUAL-PATH] Fixed old_log_probs shape: {no_tool_full_batch.batch['old_log_probs'].shape}")
                                    
                                    # Compute reference log_prob for no-tool samples
                                    if self.use_reference_policy:
                                        print(f"[DUAL-PATH] Computing ref_log_probs for no-tool samples...")
                                        if not self.ref_in_actor:
                                            no_tool_ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(no_tool_full_batch)
                                        else:
                                            no_tool_ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(no_tool_full_batch)
                                        
                                        # Verify and fix ref_log_prob shape
                                        if 'ref_log_prob' in no_tool_ref_log_prob.batch:
                                            print(f"[DUAL-PATH] ref_log_prob shape: {no_tool_ref_log_prob.batch['ref_log_prob'].shape}")
                                            expected_response_len = no_tool_full_batch.batch['responses'].shape[1]
                                            actual_ref_len = no_tool_ref_log_prob.batch['ref_log_prob'].shape[1]
                                            
                                            if actual_ref_len != expected_response_len:
                                                print(f"[DUAL-PATH] WARNING: ref_log_prob length mismatch!")
                                                print(f"[DUAL-PATH] Expected: {expected_response_len}, Got: {actual_ref_len}")
                                                
                                                # Pad ref_log_prob to match responses length
                                                if actual_ref_len < expected_response_len:
                                                    pad_size = expected_response_len - actual_ref_len
                                                    print(f"[DUAL-PATH] Padding ref_log_prob by {pad_size}")
                                                    no_tool_ref_log_prob.batch['ref_log_prob'] = torch.nn.functional.pad(
                                                        no_tool_ref_log_prob.batch['ref_log_prob'], 
                                                        (0, pad_size), 
                                                        value=0
                                                    )
                                                else:
                                                    print(f"[DUAL-PATH] Truncating ref_log_prob to {expected_response_len}")
                                                    no_tool_ref_log_prob.batch['ref_log_prob'] = no_tool_ref_log_prob.batch['ref_log_prob'][:, :expected_response_len]
                                                
                                                print(f"[DUAL-PATH] Fixed ref_log_prob shape: {no_tool_ref_log_prob.batch['ref_log_prob'].shape}")
                                        
                                        no_tool_full_batch = no_tool_full_batch.union(no_tool_ref_log_prob)
                                    
                                    # Compute values for no-tool samples
                                    if self.use_critic:
                                        print(f"[DUAL-PATH] Computing values for no-tool samples...")
                                        no_tool_values = self.critic_wg.compute_values(no_tool_full_batch)
                                        no_tool_full_batch = no_tool_full_batch.union(no_tool_values)
                                    
                                    # Compute advantages for no-tool samples
                                    print(f"[DUAL-PATH] Computing advantages for no-tool samples...")
                                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                                    
                                    no_tool_full_batch = compute_advantage(
                                        no_tool_full_batch,
                                        adv_estimator=self.config.algorithm.adv_estimator,
                                        gamma=self.config.algorithm.gamma,
                                        lam=self.config.algorithm.lam,
                                        num_repeat=no_tool_rollout_n,  # Use no_tool_rollout_n (configurable)
                                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                        multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                        use_pf_ppo=self.config.algorithm.use_pf_ppo,
                                        pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                                        pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                                    )
                                    
                                    # Set meta_info fields required by update_actor
                                    no_tool_full_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                    
                                    # CRITICAL FIX: Copy temperature from original batch
                                    # update_policy requires temperature in meta_info
                                    if "temperature" in new_batch.meta_info:
                                        no_tool_full_batch.meta_info["temperature"] = new_batch.meta_info["temperature"]
                                        print(f"[DUAL-PATH] Copied temperature: {new_batch.meta_info['temperature']}")
                                    elif "temperature" in gen_batch.meta_info:
                                        no_tool_full_batch.meta_info["temperature"] = gen_batch.meta_info["temperature"]
                                        print(f"[DUAL-PATH] Copied temperature from gen_batch: {gen_batch.meta_info['temperature']}")
                                    else:
                                        # Fallback: use default temperature from config
                                        default_temp = self.config.actor_rollout_ref.rollout.temperature
                                        no_tool_full_batch.meta_info["temperature"] = default_temp
                                        print(f"[DUAL-PATH] WARNING: temperature not found in batch, using default: {default_temp}")
                                    
                                    # CRITICAL FIX: Compute global_token_num for no-tool batch
                                    # update_actor requires global_token_num in meta_info
                                    no_tool_full_batch.meta_info["global_token_num"] = torch.sum(
                                        no_tool_full_batch.batch["attention_mask"], dim=-1
                                    ).tolist()
                                    print(f"[DUAL-PATH] Set global_token_num: {len(no_tool_full_batch.meta_info['global_token_num'])} samples")
                                    
                                    # ========== Final shape verification ==========
                                    print(f"\n[DUAL-PATH] ========== Final Shape Verification ==========")
                                    print(f"[DUAL-PATH] No-tool batch final shapes:")
                                    for key in ['input_ids', 'attention_mask', 'position_ids', 'responses', 
                                               'token_level_scores', 'token_level_rewards', 'ref_log_prob', 
                                               'advantages', 'returns']:
                                        if key in no_tool_full_batch.batch:
                                            print(f"[DUAL-PATH]   {key}: {no_tool_full_batch.batch[key].shape}")
                                    print(f"[DUAL-PATH] =============================================\n")

                                    # Store no_tool_full_batch for later use in actor update
                                    self._no_tool_batch_for_training = no_tool_full_batch
                                    
                                    print(f"[DUAL-PATH] No-tool batch prepared for GRPO training")
                                    print(f"[DUAL-PATH] No-tool reward stats:")
                                    no_tool_rewards = no_tool_full_batch.batch["token_level_rewards"].sum(-1)
                                    print(f"  - Mean: {no_tool_rewards.mean().item():.4f}")
                                    print(f"  - Std: {no_tool_rewards.std().item():.4f}")
                                    print(f"  - Min: {no_tool_rewards.min().item():.4f}")
                                    print(f"  - Max: {no_tool_rewards.max().item():.4f}")
                                else:
                                    print(f"\n[DUAL-PATH] Skipping no-tool GRPO training (β=0)")
                                    self._no_tool_batch_for_training = None


                                # Save no-tool samples to disk (with full metrics if available)
                                if hasattr(self, '_no_tool_batch_for_training') and self._no_tool_batch_for_training is not None:
                                    # Save with full metrics (after reward computation)
                                    self.dual_path_manager.save_no_tool_rollouts(
                                        prompts=no_tool_gen_batch,
                                        rollout_output=no_tool_gen_batch_output,
                                        full_batch=self._no_tool_batch_for_training,
                                        step=self.global_steps,
                                    )
                                else:
                                    # Save without metrics (beta=0, no reward computation)
                                    self.dual_path_manager.save_no_tool_rollouts(
                                        prompts=no_tool_gen_batch,
                                        rollout_output=no_tool_gen_batch_output,
                                        full_batch=None,
                                        step=self.global_steps,
                                    )
                                
                                print(f"[DUAL-PATH] No-tool rollout completed and saved")
                                print(f"[DUAL-PATH] With-tool samples: {len(gen_batch_output.batch['responses'])}")
                                print(f"[DUAL-PATH] No-tool samples: {len(no_tool_gen_batch_output.batch['responses'])}")
                                print(f"{'='*80}\n")
                                
                                # Update step count
                                self.dual_path_manager.step_count += 1
                            else:
                                print(f"[DUAL-PATH] WARNING: Failed to create no-tool prompts, skipping no-tool rollout")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=new_batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)
                        new_batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]


                    # Dynamic Sampling
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:   # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (new_batch.batch["token_level_rewards"].sum(dim=-1).numpy())
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (new_batch.batch["token_level_scores"].sum(dim=-1).numpy())

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]

                        filtered_out_prompt_uids = [
                            uid for uid in prompt_uid2metric_vals.keys() 
                            if uid not in kept_prompt_uids
                        ]

                        if filtered_out_prompt_uids:
                            print(f"\n=== Step {self.global_steps}: Filtered out {len(filtered_out_prompt_uids)} prompt groups ===")
                            for uid in filtered_out_prompt_uids:
                                rewards = prompt_uid2metric_vals[uid]
                                std_val = prompt_uid2metric_std[uid]
                                print(f"Prompt UID {uid}: rewards={rewards}, std={std_val:.6f}")
                            print("=" * 60)

                        #current_batch_prompt_count = len(kept_prompt_uids)
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        filtered_out_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                            else:
                                filtered_out_traj_idxs.append(idx)

                        if filtered_out_traj_idxs:
                            filtered_out_batch = new_batch[filtered_out_traj_idxs]

                            # Print detailed trajectory-level rewards for filtered out samples
                            print(f"\n--- Detailed trajectory rewards for filtered out samples ---")
                            filtered_sequence_rewards = filtered_out_batch.batch["token_level_rewards"].sum(-1).numpy()
                            filtered_uids = filtered_out_batch.non_tensor_batch["uid"]
                            
                            for traj_idx, (uid, seq_reward) in enumerate(zip(filtered_uids, filtered_sequence_rewards)):
                                print(f"Trajectory {traj_idx} (UID {uid}): sequence_reward={seq_reward:.6f}")
                            print("-" * 60)

                            filtered_out_batches.append(filtered_out_batch)
                        
                        new_batch = new_batch[kept_traj_idxs]
                        batch, new_batch = self._pad_batch_to_match(batch, new_batch)

                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")                       
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                #progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                print(f"{num_gen_batches=} >= {max_num_gen_batches=}. Trying to fill from backup samples...")
                                
                                if filtered_out_batches:
                                    # 计算还需要多少个prompt
                                    needed_prompts = prompt_bsz - num_prompt_in_batch
                                    print(f"Need {needed_prompts} more prompts. Searching in {len(filtered_out_batches)} backup batches...")
                                    
                                    # 从备用样本中选择轨迹
                                    backup_batch = None
                                    backup_prompt_count = 0
                                    
                                    for backup_batch_candidate in filtered_out_batches:
                                        if backup_batch is None:
                                            backup_batch = backup_batch_candidate
                                        else:
                                            backup_batch, backup_batch_candidate = self._pad_batch_to_match(backup_batch, backup_batch_candidate)
                                            backup_batch = DataProto.concat([backup_batch, backup_batch_candidate])
                                    
                                    if backup_batch is not None:
                                        # 从备用样本中获取unique prompt UIDs
                                        backup_prompt_uids = list(set(backup_batch.non_tensor_batch["uid"]))
                                        
                                        # 选择需要的prompt数量
                                        selected_prompt_uids = backup_prompt_uids[:needed_prompts]
                                        
                                        # 筛选对应的轨迹
                                        backup_kept_traj_idxs = []
                                        for idx, traj_uid in enumerate(backup_batch.non_tensor_batch["uid"]):
                                            if traj_uid in selected_prompt_uids:
                                                backup_kept_traj_idxs.append(idx)
                                        
                                        if backup_kept_traj_idxs:
                                            backup_selected_batch = backup_batch[backup_kept_traj_idxs]
                                            backup_selected_batch, _ = self._pad_batch_to_match(backup_selected_batch, batch)
                                            
                                            # 合并到主batch中
                                            batch = DataProto.concat([batch, backup_selected_batch])
                                            backup_prompt_count = len(selected_prompt_uids)
                                            num_prompt_in_batch += backup_prompt_count
                                            
                                            print(f"Added {backup_prompt_count} prompts from backup samples. Total prompts: {num_prompt_in_batch}")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]             

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()


                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        if "loss_mask" in batch.batch.keys():
                            loss_mask = batch.batch["loss_mask"]
                        else:
                            loss_mask = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        #save_batch_as_parquet(batch, "/xxx/xxx/experiments-output/arpo/saved_batch_with_entropy.parquet")

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        # reward_extra_infos_dict: dict[str, list]
                        # if self.config.reward_model.launch_reward_fn_async:
                        #     reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # batch.batch["token_level_scores"] = reward_tensor

                        # print(f"{list(reward_extra_infos_dict.keys())=}")
                        # if reward_extra_infos_dict:
                        #     batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # # compute rewards. apply_kl_penalty if available
                        # if self.config.algorithm.use_kl_in_reward:
                        #     batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable

                            # ========== Start: Dual-path GRPO training ===#

                            # Check if we have no-tool batch to train
                            has_no_tool_batch = hasattr(self, '_no_tool_batch_for_training') and self._no_tool_batch_for_training is not None

                            sdpo_tensor_dict = None
                            enable_sdpo = self.config.algorithm.get("enable_sdpo", False)
                            self._sdpo_disable_capability = self.config.algorithm.get("sdpo_disable_capability", False)

                            if has_no_tool_batch and enable_sdpo:
                              
                                try:
                                    # Call existing function to find pairs (e.g., No-Tool Correct > With-Tool Wrong)
                                    # batch: With-Tool Batch (Pure)
                                    # self._no_tool_batch_for_training: No-Tool Batch
                                    sdpo_tensor_dict, dpo_stats = self._construct_sdpo_batch(batch, self._no_tool_batch_for_training)

                                    # Log DPO statistics (how many pairs found)
                                    if dpo_stats:
                                        metrics.update(dpo_stats)
                                    
                                    if sdpo_tensor_dict is not None and sdpo_tensor_dict.batch_size[0] > 0:
                                        print(f"[SDPO] ✓ Attached {sdpo_tensor_dict.batch_size[0]} SDPO to Actor update.")
                                        # Move to CPU to be safe for meta_info transfer (Ray handles this, but good practice)
                                        # The Actor will move it back to GPU
                                        sdpo_data_cpu = sdpo_tensor_dict.cpu()
                                        batch.meta_info["dpo_data"] = sdpo_data_cpu
                                    else:
                                        print(f"[DPO] No valid pairs found in this step.")
                                        pass
                                
                                except Exception as e:
                                    print(f"[DPO] ✗ ERROR constructing DPO batch: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    batch.meta_info.pop("dpo_data", None)

                            # 3. Clean up No-Tool Batch
                            # Important: Clear it so it doesn't leak into future steps if rollout fails
                            if has_no_tool_batch:
                                self._no_tool_batch_for_training = None

                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                            #save_batch_as_parquet(batch, "/xxx/xxx/experiments-output/arpo/saved_batch_with_entropy_1246row.parquet")

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                num_prompt_in_batch = 0
                num_gen_batches = 0
                batch = None
                filtered_out_batches = []
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
