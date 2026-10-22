# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
"""
使用 QwenVL 从 Jsonl 抽取多模态 embedding，支持分布式与异步保存。
"""
import os
import sys
import json
import requests
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from io import BytesIO

import multiprocessing as mp
import orjson
import torch
import torchvision.transforms as transforms
from queue import Full
from loguru import logger
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

try:
    from transformers import Qwen2_5_VLConfig
except ImportError:  # pragma: no cover - older transformers builds
    Qwen2_5_VLConfig = None  # type: ignore[assignment,misc]

from .utils.image_utils import load_and_resize_image_for_condition

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = torch.device("cuda", LOCAL_RANK)
SAVE_SENTINEL_TIMEOUT_SECONDS = 30
SAVE_QUEUE_TIMEOUT_SECONDS = 30
SAVE_JOIN_TIMEOUT_SECONDS = 120

logger.remove()
logger = logger.bind(rank=f"RANK {RANK}")
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{extra[rank]}</green> | <cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | <level>{level}</level> | <level>{message}</level>",
)


def _processor_from_path(path: str) -> Qwen2VLProcessor:
    """Load a processor from either a FireRed root or standalone directory."""

    candidate = Path(path).expanduser()
    processor_dir = candidate / "processor"
    if processor_dir.is_dir():
        return Qwen2VLProcessor.from_pretrained(
            str(candidate), subfolder="processor", local_files_only=True
        )
    return Qwen2VLProcessor.from_pretrained(str(candidate), local_files_only=True)


def _load_qwen_condition_model(
    model_path: str,
    *,
    config_path: Optional[str],
    dtype: torch.dtype,
) -> Qwen2_5_VLForConditionalGeneration:
    """Load a normal Qwen directory or a merged ``model.safetensors`` + base config."""

    model_dir = Path(model_path).expanduser()
    # Legacy extractor calls pass the complete FireRed root.  Keep that path
    # compatible while allowing gate A's merged_model to omit config.json.
    if model_dir.is_dir() and (model_dir / "text_encoder").is_dir():
        model_dir = model_dir / "text_encoder"
    config_dir = Path(config_path).expanduser() if config_path else model_dir
    if config_dir.is_file():
        config_dir = config_dir.parent
    if (model_dir / "config.json").is_file():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=dtype, local_files_only=True
        )
    else:
        if Qwen2_5_VLConfig is None:
            raise RuntimeError("Qwen2_5_VLConfig is unavailable; cannot load config-less merged condition model")
        if not (config_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"condition model has no config.json and separate config is missing: {config_dir}"
            )
        config = Qwen2_5_VLConfig.from_pretrained(
            str(config_dir), local_files_only=True
        )
        # The training image uses Transformers 4.57.1, where concrete
        # PreTrainedModel classes do not expose a public ``from_config``.
        # Supplying the separately loaded config to ``from_pretrained`` is the
        # supported config-less-directory path.  It streams the 21GB Gate-A
        # safetensors through low-memory loading instead of allocating an 8B
        # model and then making a second full state-dict copy.
        model, loading_info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            config=config,
            dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        load_errors = {
            key: loading_info.get(key, [])
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
            if loading_info.get(key)
        }
        if load_errors:
            raise RuntimeError(
                f"strict condition state load failed for {model_dir}: {load_errors}"
            )
    model.requires_grad_(False)
    model.eval()
    return model


class JsonlImageDataset(Dataset):
    DEFAULT_SYSTEM_PROMPT = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."

    DEFAULT_SYSTEM_PROMPT_T2I = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"

    def __init__(
        self,
        jsonl_path: str,
        source_image_column: str = "source_image",
        target_image_column: str = "target_image",
        condition_image_size: int = 384 * 384,
        random_hw_adapt: bool = False,
        image_sample_size: int = 512,
        model_path: Optional[str] = None,
        processor_path: Optional[str] = None,
        disable_inverse: bool = False,
        t2i_mode: bool = False,
    ):
        self.df = []
        with open(jsonl_path, "r") as f:
            for idx, line in enumerate(f):
                if idx % WORLD_SIZE != RANK:
                    continue
                self.df.append(orjson.loads(line))

        self.source_image_column = source_image_column
        self.target_image_column = target_image_column
        self.condition_image_size = condition_image_size
        if processor_path is None:
            processor_path = model_path
        if not processor_path:
            raise ValueError("condition processor path is required")
        self.processor = _processor_from_path(processor_path)
        self.random_hw_adapt = random_hw_adapt
        self.image_sample_size = image_sample_size
        self.disable_inverse = disable_inverse
        self.t2i_mode = t2i_mode
        if self.t2i_mode:
            self.system_prompt = self.DEFAULT_SYSTEM_PROMPT_T2I
            assert self.disable_inverse, "T2I mode with inverse instructions is not supported yet."
        else:
            self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        
    def __len__(self):
        return len(self.df)

    def _load_and_resize_image(self, src_image_paths: List[str], tgt_image_path: str) -> Tuple[List[Image.Image], Image.Image, List[Dict], Dict]:
        return load_and_resize_image_for_condition(
            src_image_paths,
            tgt_image_path,
            image_sample_size=self.image_sample_size,
            condition_image_size=self.condition_image_size,
        )
    
    def _extract_instructions(self, row: Dict) -> Dict[str, str]:
        return {
            "instruction": row.get("instruction") or "",
            "instruction_cn": row.get("instruction_cn") or "",
            "inverse_instruction": row.get("inverse_instruction") or "",
            "inverse_instruction_cn": row.get("inverse_instruction_cn") or "",
        }

    def _build_conversations(
        self,
        source_images: List[Image.Image],
        target_image: Image.Image,
        instructions: Dict[str, str],
    ) -> List[List[Dict]]:
        if self.t2i_mode:
            instruction_list = [
                instructions["instruction"],
                instructions["instruction_cn"],
            ]
        elif not self.disable_inverse:
            instruction_list = [
                instructions["instruction"],
                instructions["instruction_cn"],
                "",
                instructions["inverse_instruction"],
                instructions["inverse_instruction_cn"],
                "",
            ]
        else:
            instruction_list = [
                instructions["instruction"],
                instructions["instruction_cn"],
                "",
            ]

        conversations = []
        for idx, instruction in enumerate(instruction_list):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": []},
            ]
            condition_images = source_images if idx < 3 else [target_image]
            # if idx >=3 and not self.t2i_mode, then enable inverse, condition_images is [target_image]
            for condition_image in condition_images:
                messages[1]["content"].append({"type": "image", "image": condition_image})
            messages[1]["content"].append({"type": "text", "text": instruction})
            conversations.append(messages)

        return conversations
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        row = self.df[idx]

        try:
            raw_source = row[self.source_image_column]
            if raw_source is None:
                source_image_paths = []
            elif isinstance(raw_source, str):
                source_image_paths = [raw_source]
            else:
                source_image_paths = list(raw_source)
            source_images, target_image, source_image_size, edit_image_size = self._load_and_resize_image(
                source_image_paths, row[self.target_image_column]
            )
            instructions = self._extract_instructions(row)
            conversations = self._build_conversations(source_images, target_image, instructions)

            model_inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                padding=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            traceback.print_exc()
            return None

        result = row.copy()
        result["_index"] = idx
        result["_jsonl_lineno"] = idx * WORLD_SIZE + RANK
        result["_model_inputs"] = model_inputs
        result["source_image_size"] = source_image_size
        result["edit_image_size"] = edit_image_size
        return result


def collate_fn(batch: List[Dict]) -> List[Dict]:
    return [item for item in batch if item is not None]


def _rank_has_rows(jsonl_path: str) -> bool:
    """Return whether this distributed rank owns at least one JSONL row."""

    with open(jsonl_path, "rb") as handle:
        return any(idx % WORLD_SIZE == RANK for idx, _ in enumerate(handle))


def _rank_output_jsonl(output_jsonl_dir: str, jsonl_path: str) -> Path:
    return Path(output_jsonl_dir) / f"{Path(jsonl_path).stem}_rank_{RANK:03d}.jsonl"


def _write_empty_rank_shard(output_jsonl_dir: str, jsonl_path: str) -> Path:
    """Materialize an empty metadata shard without loading Qwen-VL."""

    output = _rank_output_jsonl(output_jsonl_dir, jsonl_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    return output


class QwenEmbeddingExtractor:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        condition_image_size: int = 384 * 384,
        random_hw_adapt: bool = False,
        image_sample_size: int = 512,
        processor_path: Optional[str] = None,
        config_path: Optional[str] = None,
        result_queue: Optional[mp.Queue] = None,
        condition_metadata: Optional[Dict[str, object]] = None,
        disable_inverse: bool = False,
        t2i_mode: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.bfloat16
        self.condition_image_size = condition_image_size
        self.model_path = model_path
        self.processor_path = processor_path
        self.config_path = config_path
        self.random_hw_adapt = random_hw_adapt
        self.image_sample_size = image_sample_size
        self.result_queue = result_queue
        self.condition_metadata = dict(condition_metadata or {})
        self.disable_inverse = disable_inverse
        self.t2i_mode = t2i_mode

        if self.t2i_mode:
            self.prompt_template_start_idx = 34
            assert self.disable_inverse, "T2I mode with inverse instructions is not supported yet."
            self.num_sequences_per_sample = 2
        else:
            self.prompt_template_start_idx = 64
            self.num_sequences_per_sample = 6 if not self.disable_inverse else 3

        logger.info(f"加载模型: {model_path}, 设备: {self.device}, dtype: {self.dtype}")

        self.text_encoder = _load_qwen_condition_model(
            model_path,
            config_path=config_path,
            dtype=self.dtype,
        ).to(self.device)
        try:
            self.text_encoder.set_attn_implementation("flash_attention_2")
            logger.info("Qwen-VL attention backend: flash_attention_2")
        except ImportError as error:
            logger.warning(
                "FlashAttention2 不可用，回退到 SDPA；embedding 数值契约不变。原因: {}",
                error,
            )
            self.text_encoder.set_attn_implementation("sdpa")
            logger.info("Qwen-VL attention backend: sdpa")
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        logger.info("模型加载完成")
    
    def _extract_valid_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        batch_size = hidden_states.shape[0]
        valid_hidden_states = []
        for i in range(batch_size):
            valid_indices = attention_mask[i].nonzero(as_tuple=True)[0]
            valid_hidden_states.append(hidden_states[i][valid_indices])
        return valid_hidden_states
    
    def _prepare_batch_inputs(
        self,
        batch: List[Dict],
        device: Union[str, torch.device],
    ) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_attention_mask = []
        all_pixel_values = []
        all_image_grid_thw = []

        for item in batch:
            if item is None:
                continue
            model_inputs = item["_model_inputs"]
            num_sequences = model_inputs.input_ids.shape[0]
            for seq_idx in range(num_sequences):
                all_input_ids.append(model_inputs.input_ids[seq_idx])
                all_attention_mask.append(model_inputs.attention_mask[seq_idx])
            if not self.t2i_mode:
                all_pixel_values.append(model_inputs.pixel_values)
                all_image_grid_thw.append(model_inputs.image_grid_thw)

        input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=0).to(device)
        attention_mask = pad_sequence(all_attention_mask, batch_first=True, padding_value=0).to(device)
        if not self.t2i_mode:
            pixel_values = torch.cat(all_pixel_values, dim=0).to(device)
            image_grid_thw = torch.cat(all_image_grid_thw, dim=0).to(device)
        else:
            pixel_values = None
            image_grid_thw = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
    
    def _group_embeddings_by_sample(self, embeddings: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        grouped = []
        current = []
        for emb in embeddings:
            current.append(emb)
            if len(current) == self.num_sequences_per_sample:
                grouped.append(current)
                current = []
        if current:
            raise RuntimeError(
                f"Embedding count {len(embeddings)} is not divisible by "
                f"{self.num_sequences_per_sample} sequences/sample"
            )
        return grouped
    
    def _get_qwen_prompt_embeds_batch(
        self,
        batch: List[Dict],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[List[torch.Tensor]]:
        device = device or self.device
        dtype = dtype or self.dtype
        batch_inputs = self._prepare_batch_inputs(batch, device)

        # Call the multimodal Qwen backbone directly.  Calling the enclosing
        # ForConditionalGeneration module would run ``lm_head`` and allocate
        # full-vocabulary logits even though restoration consumes only the
        # final hidden states.  The backbone output is exactly the hidden-state
        # branch used by that wrapper, with no DECISION/REASONS text path.
        backbone = getattr(self.text_encoder, "model", None)
        if backbone is None:
            raise RuntimeError(
                "Qwen condition model has no `.model` backbone; refusing to run LM head"
            )
        with torch.inference_mode():
            outputs = backbone(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                pixel_values=batch_inputs["pixel_values"],
                image_grid_thw=batch_inputs["image_grid_thw"],
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states[-1]
        valid_hidden_states = self._extract_valid_hidden_states(
            hidden_states, batch_inputs["attention_mask"]
        )
        start_idx = self.prompt_template_start_idx
        truncated = [h[start_idx:].cpu() for h in valid_hidden_states]
        return self._group_embeddings_by_sample(truncated)
    
    
    def _format_output_result(self, item: Dict, embeddings: List[torch.Tensor]) -> Dict:
        result = {k: v for k, v in item.items() if not k.startswith("_")}
        result["jsonl_lineno"] = item["_jsonl_lineno"]

        result["embeddings_tensor_en"] = embeddings[0]
        result["embeddings_tensor_cn"] = embeddings[1]
        result["embeddings_tensor_droptext"] = embeddings[2]
        if self.condition_metadata:
            # The scalar/nested JSON values are copied to every row.  CLI
            # validation compares them to the cache-local condition marker
            # before any mmDiT worker can consume a tensor.
            result.update(self.condition_metadata)
        if not self.disable_inverse:
            result["embeddings_tensor_en_inv"] = embeddings[3]
            result["embeddings_tensor_cn_inv"] = embeddings[4]
            result["embeddings_tensor_droptext_inv"] = embeddings[5]

        return result
    
    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        all_embeddings = self._get_qwen_prompt_embeds_batch(batch)
        return [
            self._format_output_result(item, emb_list)
            for item, emb_list in zip(batch, all_embeddings)
        ]

    def run(
        self,
        jsonl_path: str,
        batch_size: int = 512,
        num_workers: int = 4,
    ) -> None:
        logger.info(f"读取 jsonl: {jsonl_path}, batch_size={batch_size}")

        dataset = JsonlImageDataset(
            jsonl_path=jsonl_path,
            source_image_column="source_image",
            target_image_column="edit_image",
            condition_image_size=self.condition_image_size,
            random_hw_adapt=self.random_hw_adapt,
            image_sample_size=self.image_sample_size,
            model_path=self.model_path,
            processor_path=self.processor_path,
            disable_inverse=self.disable_inverse,
            t2i_mode=self.t2i_mode,
        )

        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": len(dataset) > 0,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": torch.device(self.device).type == "cuda",
        }
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 4
        dataloader = DataLoader(**dataloader_kwargs)

        for batch in tqdm(dataloader, desc=f"RANK {RANK} 提取 Embedding"):
            if not batch:
                continue
            try:
                batch_results = self.process_batch(batch)
                if batch_results is not None:
                    self.result_queue.put(
                        batch_results, timeout=SAVE_QUEUE_TIMEOUT_SECONDS
                    )
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                traceback.print_exc()
                raise

        logger.info("提取完成")

class SaveWorker(mp.Process):
    def __init__(
        self,
        result_queue: mp.Queue,
        output_jsonl_dir: str,
        embeddings_save_dir: str,
        jsonl_path: str,
        disable_inverse: bool = False,
        t2i_mode: bool = False,
    ):
        super().__init__()
        self.result_queue = result_queue
        output_jsonl_dir = Path(output_jsonl_dir)
        output_jsonl_dir.mkdir(parents=True, exist_ok=True)
        jsonl_stem = Path(jsonl_path).stem
        self.out_jsonl = _rank_output_jsonl(output_jsonl_dir, jsonl_path)

        embeddings_save_dir = Path(embeddings_save_dir)
        self.embed_dir = embeddings_save_dir / jsonl_stem
        self.embed_dir.mkdir(parents=True, exist_ok=True)

        if t2i_mode:
            self.suffixes = ["_en", "_cn"]
        elif disable_inverse:
            self.suffixes = ["_en", "_cn", "_droptext"]
        else:
            self.suffixes = ["_en", "_cn", "_droptext", "_en_inv", "_cn_inv", "_droptext_inv"]
        self.tensor_keys = [f"embeddings_tensor{s}" for s in self.suffixes]

    def _save_results(self, results: List[Dict]) -> None:
        if not results:
            return

        with open(self.out_jsonl, "a") as f:
            for rec in results:
                lineno = rec["jsonl_lineno"]
                line_out = {
                    k: v for k, v in rec.items()
                    if k not in self.tensor_keys and k != "jsonl_lineno"
                }
                for suf, key in zip(self.suffixes, self.tensor_keys):
                    tensor = rec.get(key)
                    if tensor is not None:
                        save_path = self.embed_dir / f"{lineno}{suf}.pt"
                        torch.save(tensor, save_path)
                        line_out[key] = save_path.as_posix()

                f.write(orjson.dumps(line_out).decode("utf-8") + "\n")

    def run(self):
        while True:
            try:
                results = self.result_queue.get()
                if results is None:
                    logger.info("SaveWorker 收到结束信号，退出")
                    logger.info(f"处理完成. Embedding 保存到: {self.embed_dir}, 输出 jsonl: {self.out_jsonl}")
                    break
                self._save_results(results)
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                traceback.print_exc()
                raise


def main(args) -> None:
    if not _rank_has_rows(args.jsonl_path):
        output = _write_empty_rank_shard(args.output_jsonl_dir, args.jsonl_path)
        logger.info(
            "Rank has no assigned rows; wrote empty metadata shard and skipped "
            f"Qwen-VL loading: {output}"
        )
        return

    result_queue = mp.Queue(maxsize=2)
    extractor = QwenEmbeddingExtractor(
        model_path=args.model_path,
        device=DEVICE,
        dtype=torch.bfloat16,
        image_sample_size=args.image_sample_size,
        processor_path=args.processor_path,
        config_path=args.config_path,
        result_queue=result_queue,
        condition_metadata={
            key: value
            for key, value in {
                "condition_schema": args.condition_schema,
                "condition_kind": args.condition_kind,
                "condition_fingerprint": args.condition_fingerprint,
                "cache_fingerprint": args.condition_fingerprint,
                "condition_variant": args.condition_variant,
                "condition_role": args.condition_role,
                "condition_model_hash": args.condition_model_hash,
                "condition_checkpoint_step": (
                    int(args.condition_checkpoint_step)
                    if args.condition_checkpoint_step is not None
                    and str(args.condition_checkpoint_step).strip().lstrip("-").isdigit()
                    else args.condition_checkpoint_step
                ),
                "condition_checkpoint_world_size": (
                    int(args.condition_checkpoint_world_size)
                    if args.condition_checkpoint_world_size is not None
                    and str(args.condition_checkpoint_world_size).strip().lstrip("-").isdigit()
                    else args.condition_checkpoint_world_size
                ),
                "condition_model_path": args.condition_model_identity,
                "condition_prompt": (
                    json.loads(args.condition_prompt_json)
                    if args.condition_prompt_json else None
                ),
                "condition_extraction": (
                    json.loads(args.condition_extraction_json)
                    if args.condition_extraction_json else None
                ),
            }.items()
            if value is not None
        },
        disable_inverse=args.disable_inverse,
        t2i_mode=args.t2i_mode,
    )

    save_worker = SaveWorker(
        result_queue,
        args.output_jsonl_dir,
        args.embeddings_save_dir,
        args.jsonl_path,
        disable_inverse=args.disable_inverse,
        t2i_mode=args.t2i_mode,
    )
    save_worker.start()

    run_exc_info = None
    cleanup_error = None
    try:
        extractor.run(
            jsonl_path=args.jsonl_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except BaseException:
        run_exc_info = sys.exc_info()
    finally:
        if save_worker.exitcode is None:
            try:
                result_queue.put(None, timeout=SAVE_SENTINEL_TIMEOUT_SECONDS)
            except Full:
                cleanup_error = RuntimeError(
                    "Timed out sending the SaveWorker completion sentinel"
                )
        save_worker.join(timeout=SAVE_JOIN_TIMEOUT_SECONDS)
        if save_worker.is_alive():
            save_worker.terminate()
            save_worker.join(timeout=10)
            cleanup_error = cleanup_error or RuntimeError(
                f"SaveWorker did not exit within {SAVE_JOIN_TIMEOUT_SECONDS}s"
            )
        elif save_worker.exitcode not in (0, None):
            cleanup_error = cleanup_error or RuntimeError(
                f"SaveWorker exited with code {save_worker.exitcode}"
            )
        result_queue.close()
        result_queue.join_thread()

    if run_exc_info is not None:
        _, error, traceback_obj = run_exc_info
        raise error.with_traceback(traceback_obj)
    if cleanup_error is not None:
        raise cleanup_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen2VL 从 Jsonl 抽取多模态 embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("jsonl_path", type=str, help="输入 jsonl 路径")
    parser.add_argument("--output_jsonl_dir", type=str, required=True, help="输出 jsonl 目录，各 rank 保存为 {dir}/{输入stem}_rank_{rank}.jsonl")
    parser.add_argument("--embeddings_save_dir", type=str, required=True, help="embedding 保存目录")
    parser.add_argument("--model_path", type=str, default="/dev/shm/Qwen-Image", help="Qwen2VL 模型路径")
    parser.add_argument(
        "--processor_path", type=str, default=None,
        help="Qwen-VL processor root; defaults to model_path (FireRed root or standalone processor dir)",
    )
    parser.add_argument(
        "--config_path", type=str, default=None,
        help="separate Qwen config root for a config-less merged model.safetensors directory",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--image_sample_size", type=int, default=512, help="图像采样尺寸")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader 进程数")
    parser.add_argument("--disable_inverse", action="store_true", help="禁用逆向提示词")
    parser.add_argument("--t2i_mode", action="store_true", help="T2I 模式")
    parser.add_argument("--condition_schema", default=None)
    parser.add_argument("--condition_kind", default=None)
    parser.add_argument("--condition_fingerprint", default=None)
    parser.add_argument("--condition_variant", default=None)
    parser.add_argument("--condition_role", default=None)
    parser.add_argument("--condition_model_hash", default=None)
    parser.add_argument("--condition_checkpoint_step", default=None)
    parser.add_argument("--condition_checkpoint_world_size", default=None)
    parser.add_argument("--condition_model_identity", default=None)
    parser.add_argument("--condition_prompt_json", default=None)
    parser.add_argument("--condition_extraction_json", default=None)
    main(parser.parse_args())
