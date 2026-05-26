import os
import json
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
from copy import deepcopy
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DualPathRolloutManager:
    """
    Dual-Path Rollout Manager (Side-Channel Loading & Vectorized Version)
    
    Why this version?
    - Bypasses DataLoader/Ray code sync issues.
    - Loads 'prompt_no_tool' directly from Parquet file using content matching.
    - Vectorized Processing: Uses batch_decode to minimize CPU overhead (Fixes the 6x slowdown).
    - Guarantees access to data even if 'rl_dataset.py' filters it out.
    """
    
    def __init__(self, config, tokenizer, save_dir=None):
        self.config = config
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        
        self.enable = config.get('dual_path_distillation', {}).get('enable', False)
        self.save_samples = config.get('dual_path_distillation', {}).get('save_samples', True)
        
        self.step_count = 0
        self.total_no_tool_samples = 0
        
        # === Cache Initialization ===
        self.prompt_map = {} # Key: With-Tool Text Hash, Value: No-Tool Raw Data
        
        if self.enable:
            self._preload_dataset_side_channel()
            
        if self.enable and self.save_samples and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        logger.info(f"[DUAL-PATH] Manager initialized. Enable: {self.enable}")

    def is_enabled(self):
        return self.enable

    def _preload_dataset_side_channel(self):
        """
        Loads the Parquet file and builds a lookup map.
        Key = Hash of the With-Tool prompt (which we can get from batch input_ids)
        Value = The No-Tool prompt data
        """
        try:
            # 1. Get dataset path
            dataset_path = self.config.get('data', {}).get('train_files', None)
            if isinstance(dataset_path, list): dataset_path = dataset_path[0]
            
            # Fallback path (change this if needed)
            if not dataset_path or not os.path.exists(dataset_path):
                # Try relative path or hardcoded fallback
                fallback = "/xxx/xxx/rl_datasets/nq_dual_path/train_dual_path.parquet"
                if os.path.exists(fallback):
                    dataset_path = fallback
            
            if not dataset_path or not os.path.exists(dataset_path):
                logger.error(f"[DUAL-PATH] Could not find dataset file at {dataset_path}")
                return

            logger.info(f"[DUAL-PATH] Preloading data from: {dataset_path}")
            df = pd.read_parquet(dataset_path)
            
            # 2. Build Map
            count = 0
            for _, row in df.iterrows():
                if 'prompt_no_tool' not in row: continue
                
                # Retrieve With-Tool Prompt (Key)
                # It might be in 'prompt' column
                prompt_data = row['prompt']
                
                # Apply chat template to generate the exact string model sees
                # This ensures our hash matches what we get from input_ids later
                try:
                    if isinstance(prompt_data, (list, np.ndarray)):
                        if hasattr(prompt_data, 'tolist'): prompt_data = prompt_data.tolist()
                        key_text = self.tokenizer.apply_chat_template(prompt_data, tokenize=False, add_generation_prompt=True)
                    else:
                        key_text = str(prompt_data)
                    
                    # Create Hash Key (Use text directly or hash)
                    # Using text directly is safer for collision, memory usage is okay for text
                    self.prompt_map[key_text] = row['prompt_no_tool']
                    count += 1
                except Exception as e:
                    continue

            logger.info(f"[DUAL-PATH] Successfully cached {count} prompt pairs.")
            
        except Exception as e:
            logger.error(f"[DUAL-PATH] Failed to preload dataset: {e}")

    def create_no_tool_prompts(self, batch, target_max_length: Optional[int] = None, rollout_n: Optional[int] = None) -> Optional['DataProto']:
        """
        Create No-Tool Batch using Side-Channel Lookup.
        [OPTIMIZED] Uses batch_decode to significantly speed up CPU processing.
        
        Args:
            batch: The source batch (with-tool)
            target_max_length: Padding length alignment
            rollout_n: If specified, sets the 'n' parameter in meta_info for generation
        """
        logger.info(f"\n[DUAL-PATH] === Processing No-Tool Prompts (Vectorized) ===")
        
        # 1. Parse Batch
        if isinstance(batch, dict):
            non_tensor_batch = batch.get('non_tensor_batch', {})
            current_input_ids = batch['input_ids']
            device = batch['input_ids'].device
        else:
            non_tensor_batch = batch.non_tensor_batch if hasattr(batch, 'non_tensor_batch') else {}
            current_input_ids = batch.batch['input_ids']
            device = batch.batch['input_ids'].device
        
        batch_size = current_input_ids.size(0)
        
        # 2. Batch Decode Input IDs (The Key optimization)
        # Move to CPU once and decode all at once. This replaces the loop decode.
        cpu_input_ids = current_input_ids.cpu()
        # skip_special_tokens=False is CRITICAL because the map key includes chat template tags
        decoded_texts = self.tokenizer.batch_decode(cpu_input_ids, skip_special_tokens=False)
        
        final_prompt_strings = []
        found_count = 0
        
        # 3. Fast Lookup & Template Application (String operations only)
        for i in range(batch_size):
            text = decoded_texts[i]
            
            # Cleaning: remove padding tokens (pad_token usually decoded)
            if self.tokenizer.pad_token:
                text = text.replace(self.tokenizer.pad_token, "")
            
            # Lookup
            item = self.prompt_map.get(text)
            if not item:
                # Retry by stripping potential BOS/EOS artifacts
                stripped = text.strip()
                item = self.prompt_map.get(stripped)
            
            # Process Item
            if item:
                found_count += 1
                # Apply Chat Template to No-Tool Data
                # Ensure it's in a list format if it came from numpy
                if isinstance(item, (np.ndarray)):
                    if hasattr(item, 'tolist'): item = item.tolist()
                
                try:
                    # Direct Data-Driven Approach: Use item exactly as is
                    final_prompt = self.tokenizer.apply_chat_template(
                        item, tokenize=False, add_generation_prompt=True
                    )
                except Exception as e:
                    logger.warning(f"[DUAL-PATH] Template error on sample {i}, using str(): {e}")
                    final_prompt = str(item)
            else:
                # Fallback: simple default prompt
                # Note: We prioritize the Side-Channel map. Only fallback if absolutely necessary.
                logger.warning(f"[DUAL-PATH] Cache MISS for index {i}. Using empty prompt.")
                messages = [{"role": "user", "content": "Error retrieving prompt."}]
                final_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            final_prompt_strings.append(final_prompt)
            
            # Log preview
            if i == 0:
                logger.info(f"[DUAL-PATH] Sample 0 Construction:")
                logger.info(f"  [Decoded Key]: {text[:50]}...")
                logger.info(f"  [Lookup Result]: {'Found' if item else 'Missing'}")
                logger.info(f"  [Final Prompt]:\n{final_prompt[:200]}...")

        logger.info(f"[DUAL-PATH] Matched {found_count}/{batch_size} prompts via Side-Channel.")

        # 4. Batch Tokenize (Tokenization)
        # This calls the helper which has the FORCE LEFT PADDING logic
        no_tool_gen_batch = self._tokenize_and_create_batch(
            final_prompt_strings, 
            device, 
            target_max_length,
            non_tensor_batch_template=non_tensor_batch
        )

        if rollout_n is not None and rollout_n > 0:
            if no_tool_gen_batch.meta_info is None:
                no_tool_gen_batch.meta_info = {}
            no_tool_gen_batch.meta_info['n'] = rollout_n
            logger.info(f"[DUAL-PATH] Configured meta_info['n'] = {rollout_n} for direct sampling.")

        # Optional Debug Check
        # self._debug_check_padding(
        #     no_tool_gen_batch.batch['input_ids'], 
        #     no_tool_gen_batch.batch['attention_mask'],
        #     name="No-Tool Batch"
        # )
        
        return no_tool_gen_batch

    def _tokenize_and_create_batch(self, texts, device, max_len, non_tensor_batch_template=None):
        """Helper to tokenize and create DataProto with FORCED LEFT PADDING"""
        max_len = max_len or self.config.get('data', {}).get('max_prompt_length', 2000)
        
        # === 核心修复开始 ===
        # 1. 强行保存旧设置
        original_padding_side = self.tokenizer.padding_side
        
        # 2. 强行设置为左填充 (生成任务必须左填充！)
        self.tokenizer.padding_side = 'left'
        
        # 3. 确保 pad_token_id 存在
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        logger.info(f"[DUAL-PATH] Tokenizing with padding_side='{self.tokenizer.padding_side}' (Should be 'left')")
        
        # 4. Tokenize
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=max_len,
            add_special_tokens=True 
        )
        
        # 5. 恢复旧设置 (好习惯，虽然在这里可能不重要)
        self.tokenizer.padding_side = original_padding_side
        # === 核心修复结束 ===
        
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True, 
            truncation='error'
        )
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        batch = DataProto.from_dict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })
        
        if non_tensor_batch_template:
            batch.non_tensor_batch = deepcopy(non_tensor_batch_template)
        
        batch.non_tensor_batch['prompt_text'] = np.array(texts, dtype=object)
        
        # Debug 再次检查 (现在应该能看到 Input IDs 开头全是 Pad 了)
        self._debug_check_padding(input_ids, attention_mask, name="No-Tool Batch (Fixed)")
        
        return batch

    def save_no_tool_rollouts(self, prompts, rollout_output, step, full_batch=None):
        """Save no-tool rollouts with answer, reason, and ground_truth fields"""
        if not self.enable or not self.save_samples or not self.save_dir:
            return
        
        try:
            # Extract inputs (prompts)
            if hasattr(prompts, 'non_tensor_batch') and 'prompt_text' in prompts.non_tensor_batch:
                inputs = prompts.non_tensor_batch['prompt_text'].tolist()
            else:
                inputs = ["[Prompt Missing]"] * len(prompts.batch['input_ids'])
            
            # Extract outputs (responses)
            outputs = []
            if hasattr(rollout_output, 'batch') and 'responses' in rollout_output.batch:
                response_ids = rollout_output.batch['responses']
                for resp_ids in response_ids:
                    resp_ids = resp_ids[resp_ids != self.tokenizer.pad_token_id]
                    resp_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                    outputs.append(resp_text)
            else:
                outputs = ["[No Response]"] * len(inputs)
            
            # Extract scores
            scores = [0.0] * len(inputs)
            if full_batch is not None and 'token_level_scores' in full_batch.batch:
                scores = full_batch.batch['token_level_scores'].sum(-1).cpu().tolist()
            
            # Extract EM scores
            em_scores = [0.0] * len(inputs)
            if full_batch and hasattr(full_batch, 'non_tensor_batch') and 'em_score' in full_batch.non_tensor_batch:
                data = full_batch.non_tensor_batch['em_score']
                em_scores = data.tolist() if hasattr(data, 'tolist') else list(data)
            
            # Extract F1 scores
            f1_scores = [0.0] * len(inputs)
            if full_batch and hasattr(full_batch, 'non_tensor_batch') and 'f1_score' in full_batch.non_tensor_batch:
                data = full_batch.non_tensor_batch['f1_score']
                f1_scores = data.tolist() if hasattr(data, 'tolist') else list(data)
            
            # Extract answer (NEW)
            answers = [""] * len(inputs)
            if full_batch and hasattr(full_batch, 'non_tensor_batch') and 'answer' in full_batch.non_tensor_batch:
                data = full_batch.non_tensor_batch['answer']
                answers = data.tolist() if hasattr(data, 'tolist') else list(data)
            
            # Extract reason (NEW)
            reasons = [""] * len(inputs)
            if full_batch and hasattr(full_batch, 'non_tensor_batch') and 'reason' in full_batch.non_tensor_batch:
                data = full_batch.non_tensor_batch['reason']
                reasons = data.tolist() if hasattr(data, 'tolist') else list(data)
            
            # Extract ground_truth (NEW)
            ground_truths = [""] * len(inputs)
            if full_batch and hasattr(full_batch, 'non_tensor_batch') and 'ground_truth' in full_batch.non_tensor_batch:
                data = full_batch.non_tensor_batch['ground_truth']
                ground_truths = data.tolist() if hasattr(data, 'tolist') else list(data)

            # Build samples with all fields
            samples = []
            for idx in range(len(inputs)):
                sample = {
                    'input': inputs[idx],
                    'output': outputs[idx] if idx < len(outputs) else "",
                    'score': scores[idx] if idx < len(scores) else 0.0,
                    'step': step,
                    'em_score': em_scores[idx] if idx < len(em_scores) else 0.0,
                    'reason': reasons[idx] if idx < len(reasons) else "",
                    'answer': answers[idx] if idx < len(answers) else "",
                    'f1_score': f1_scores[idx] if idx < len(f1_scores) else 0.0,
                    'ground_truth': ground_truths[idx] if idx < len(ground_truths) else "",
                    'type': 'no_tool',  # Type field
                }
                samples.append(sample)
            
            # Save to file
            save_path = os.path.join(self.save_dir, f'no_tool_rollouts_step_{step}.jsonl')
            with open(save_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            self.total_no_tool_samples += len(samples)
            logger.info(f"[DUAL-PATH] Saved {len(samples)} no-tool samples to {save_path}")
            logger.info(f"[DUAL-PATH] Sample fields: {list(samples[0].keys())}")
            
        except Exception as e:
            logger.error(f"[DUAL-PATH] Error saving no-tool rollouts: {e}")
            import traceback
            traceback.print_exc()

    def log_statistics(self):
        if not self.enable: return
        logger.info(f"[DUAL-PATH] Total steps: {self.step_count}")
        logger.info(f"[DUAL-PATH] Total samples: {self.total_no_tool_samples}")

    def _debug_check_padding(self, input_ids, attention_mask, name="Batch"):
        """
        专门用于检查 Left Padding 是否正常的 Debug 函数
        """
        logger.info(f"\n[DEBUG] Checking Padding for {name}...")
        
        # 获取 Pad Token ID
        pad_id = self.tokenizer.pad_token_id
        logger.info(f"[DEBUG] Current tokenizer.pad_token_id: {pad_id}")
        
        batch_size = input_ids.size(0)
        
        # 只检查前 2 条数据，避免刷屏
        for i in range(min(2, batch_size)):
            ids = input_ids[i].tolist()
            mask = attention_mask[i].tolist()
            
            # 找到第一个非 Pad 的位置
            first_real_token_idx = -1
            for idx, token in enumerate(ids):
                if token != pad_id:
                    first_real_token_idx = idx
                    break
            
            # 打印可视化结果
            logger.info(f"--- Sample {i} ---")
            
            # 1. 打印 Input IDs 的前 20 个和后 20 个
            # 我们期望看到：[PAD, PAD, PAD, ..., Real, Real]
            prefix = ids[:10]
            suffix = ids[-10:]
            logger.info(f"Input IDs (First 10): {prefix}")
            logger.info(f"Input IDs (Last 10) : {suffix}")
            
            # 2. 打印 Mask 的前 20 个
            # 我们期望看到：[0, 0, 0, ..., 1, 1]
            mask_prefix = mask[:10]
            logger.info(f"Attn Mask (First 10): {mask_prefix}")
            
            # 3. 逻辑判断
            if first_real_token_idx == -1:
                logger.warning(f"   Sample {i} 全是 Pad！有问题！")
            elif first_real_token_idx == 0:
                logger.info(f"   Sample {i} 没有 Padding (或者是满的)，这是正常的。")
            else:
                # 检查 Pad 部分是否 Mask 都是 0
                is_mask_correct = all(m == 0 for m in mask[:first_real_token_idx])
                if is_mask_correct:
                    logger.info(f"   Sample {i} Left Padding 正常。Pad 长度: {first_real_token_idx}")
                else:
                    logger.error(f"   Sample {i} Mask 错误！Input 是 Pad 但 Mask 不是 0！")
                    logger.error(f"   Input[:{first_real_token_idx}] = {ids[:first_real_token_idx]}")
                    logger.error(f"   Mask[:{first_real_token_idx}]  = {mask[:first_real_token_idx]}")

        logger.info(f"[DEBUG] Check Complete.\n")
