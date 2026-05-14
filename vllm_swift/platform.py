# SPDX-License-Identifier: Apache-2.0
"""Swift Metal Platform implementation for vLLM.

Registers as an out-of-tree platform plugin. All model execution
is delegated to the Swift mlx-swift-lm engine via C FFI — no Python
in the GPU hot path.
"""

import logging
import platform as py_platform
from typing import TYPE_CHECKING

import psutil
import torch
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transformers AutoConfig pre-registration
#
# vLLM's ModelConfig validator calls `transformers.AutoConfig.from_pretrained`
# before our SwiftMetalWorker ever gets a turn. Transformers raises if the
# checkpoint's `model_type` is not in its CONFIG_MAPPING. For brand-new model
# families that ship in mlx-swift-lm before they land in upstream transformers
# (deepseek_v4, gemma4, qwen3_5, qwen3_next, qwen3_5_moe, minimax_m2,
# gpt_oss, nemotron_h, dflash_qwen3, ...) this rejects loads we'd otherwise
# handle just fine.
#
# Fix: at platform-plugin import time, register a permissive stub config
# class per model_type so AutoConfig accepts the checkpoint. The stub still
# reads everything from the on-disk config.json (PretrainedConfig.from_dict
# stores all keys as attributes), so downstream fields like
# `max_position_embeddings` / `hidden_size` work.
#
# Architectures already in upstream transformers (llama, mistral, phi3,
# qwen2, qwen3, gemma2, gemma3) are re-registered with exist_ok=True — a
# safe no-op when transformers already has them.
# ---------------------------------------------------------------------------

_ARCHS_WE_RUN_VIA_MLX_SWIFT_LM = (
    # New families that may not be in user's pinned transformers yet:
    "deepseek_v4",
    "gemma4", "gemma4_text", "gemma4_vision", "gemma4_audio",
    "gpt_oss",
    "qwen3_5", "qwen3_5_text",
    "qwen3_5_moe", "qwen3_5_moe_text",
    "qwen3_next", "qwen3_next_text",
    "qwen3_moe",
    "minimax_m2",
    "nemotron_h",
    "dflash_qwen3",
    "qwen2_5_vl",
    # Almost certainly already in transformers — registered with exist_ok=True
    # as belt-and-suspenders for older transformers pins:
    "llama",
    "mistral",
    "phi3",
    "qwen2",
    "qwen3",
    "gemma2",
    "gemma3", "gemma3_text",
    # SigLIP vision tower used by Gemma3-VLM checkpoints
    "siglip_vision_model",
)


def _register_passthrough_configs() -> None:
    try:
        from transformers import AutoConfig, PretrainedConfig
    except Exception as e:
        logger.warning("transformers not importable; skipping arch pre-register: %s", e)
        return

    # Class-level defaults for fields accessed during `PretrainedConfig.__post_init__`
    # before the config.json kwargs have been setattr'd. The rope-params BC dance
    # at line ~280 of transformers/configuration_utils.py reads `self.max_position_embeddings`
    # while it's still only present as a kwarg. Without a class-attr default, every
    # rope-scaling-using checkpoint trips with AttributeError. config.json values
    # later override these via the trailing `for key, value in kwargs.items(): setattr`
    # loop, so these are purely "don't crash at lookup time" sentinels.
    _stub_defaults = {
        # Generous default — most modern long-context models advertise >= 128K.
        # If a checkpoint has an explicit max_position_embeddings in its config
        # (top-level or in text_config) it overrides this; this sentinel only
        # applies when the field is missing on both levels.
        "max_position_embeddings": 131072,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "tie_word_embeddings": False,
    }

    # Names of nested-config fields used by multimodal checkpoints. vLLM's
    # `get_text_config()` extracts these and then `hasattr(text_config,
    # "num_attention_heads")`. transformers leaves these as raw `dict` unless
    # the parent config's __init__ converts them, so we do that conversion in
    # a custom __init__ on every stub class: each dict-valued sub-config that
    # carries a `model_type` is upgraded to a PretrainedConfig instance via
    # AutoConfig (which falls back to our other stubs for unknown model_types).
    _SUB_CONFIG_KEYS = (
        "text_config", "vision_config", "audio_config",
        "speech_config", "decoder_config", "encoder_config",
    )

    def _make_stub_init():
        def __init__(self, **kwargs):
            for k in _SUB_CONFIG_KEYS:
                v = kwargs.get(k)
                if isinstance(v, dict) and v.get("model_type"):
                    inner = {kk: vv for kk, vv in v.items() if kk != "model_type"}
                    try:
                        kwargs[k] = AutoConfig.for_model(v["model_type"], **inner)
                    except Exception:
                        pass  # leave as dict; downstream may not need it
            PretrainedConfig.__init__(self, **kwargs)
        return __init__

    # Only register stubs for model_types transformers doesn't already know
    # about — overriding the real class with our stub will fail downstream
    # isinstance checks in vLLM (it expects the canonical transformers class).
    try:
        from transformers import CONFIG_MAPPING
        known = set(CONFIG_MAPPING.keys())
    except Exception:
        known = set()

    registered: list[str] = []
    for model_type in _ARCHS_WE_RUN_VIA_MLX_SWIFT_LM:
        if model_type in known:
            continue
        try:
            attrs = {
                "model_type": model_type,
                **_stub_defaults,
                "__init__": _make_stub_init(),
            }
            stub_cls = type(
                f"_VllmSwiftStubConfig_{model_type}",
                (PretrainedConfig,),
                attrs,
            )
            AutoConfig.register(model_type, stub_cls)
            registered.append(model_type)
        except Exception as e:
            logger.warning("AutoConfig.register failed for %s: %s", model_type, e)
    logger.info(
        "Swift Metal pre-registered %d model_type stubs with transformers AutoConfig",
        len(registered),
    )


_register_passthrough_configs()


def _patch_missing_video_processor() -> None:
    """Monkey-patch transformers' video processor loader to no-op on missing
    config files.

    Some multimodal model dirs (e.g. Gemma 4 e2b text+audio variants)
    advertise a video-capable processor class in `processor_config.json` but
    don't ship a `video_preprocessor_config.json`. vLLM's processor probing
    chains into `BaseVideoProcessor.get_video_processor_dict(...)` which
    raises OSError on missing-file — bringing the whole engine down even
    though we'll never see a video token in practice (the Swift worker
    handles only text).

    Replace the OSError path with an empty-dict return so the processor
    constructs as a no-op stub.
    """
    try:
        from transformers import video_processing_utils
    except Exception as e:
        logger.debug("transformers.video_processing_utils not importable: %s", e)
        return

    cls = getattr(video_processing_utils, "BaseVideoProcessor", None)
    if cls is None:
        return

    orig = cls.get_video_processor_dict
    if getattr(orig, "_swift_patched", False):
        return

    def get_video_processor_dict(cls_, pretrained_model_name_or_path, **kwargs):
        try:
            return orig.__func__(cls_, pretrained_model_name_or_path, **kwargs)
        except OSError as e:
            msg = str(e)
            if "video_preprocessor_config.json" in msg or "video processor" in msg.lower():
                logger.info(
                    "Swift Metal: no video processor config at %s — returning empty stub",
                    pretrained_model_name_or_path,
                )
                return ({}, kwargs)
            raise

    get_video_processor_dict._swift_patched = True  # type: ignore[attr-defined]
    cls.get_video_processor_dict = classmethod(get_video_processor_dict)


_patch_missing_video_processor()


def _patch_gemma4_missing_vision_config() -> None:
    """Inject a stub `vision_config` into Gemma4Config when it's None.

    Some Gemma 4 derivatives ship audio-only (e.g. internal ConfigI builds
    where the vision tower is dropped). config.json has no `vision_config`
    key → transformers loads vision_config as None → vLLM's Gemma4 path
    dereferences `config.vision_config.<field>` in many places
    (`default_output_length`, `pooling_kernel_size`, `hidden_size`,
    `rms_norm_eps`, …) and crashes engine init.

    Cleanest fix: monkey-patch transformers' Gemma4Config so when
    vision_config is None, we substitute a `_NoVisionConfig` namespace
    with all the fields downstream code might read. Swift worker handles
    actual inference — vLLM just needs construction to succeed.
    """
    try:
        from transformers.models.gemma4 import configuration_gemma4
    except Exception as e:
        logger.debug("transformers.models.gemma4 not importable: %s", e)
        return

    cls = getattr(configuration_gemma4, "Gemma4Config", None)
    if cls is None:
        return

    if getattr(cls, "_swift_vision_patched", False):
        return

    # Use the real Gemma4VisionConfig with safe-minimum defaults so the
    # strict dataclass validator on Gemma4Config.vision_config accepts it.
    try:
        VisionCfg = configuration_gemma4.Gemma4VisionConfig
    except AttributeError:
        logger.debug("Gemma4VisionConfig not importable; skipping stub")
        return

    orig_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if getattr(self, "vision_config", None) is None:
            try:
                # Safe defaults that won't trigger crashes in vLLM's
                # tokens-per-image / pooler-shape calculators — they'll
                # produce 0 image tokens for an audio-only build.
                self.vision_config = VisionCfg(
                    hidden_size=128,
                    intermediate_size=128,
                    num_hidden_layers=0,
                    num_attention_heads=1,
                    num_channels=3,
                    image_size=16,
                    patch_size=16,
                    default_output_length=0,
                    pooling_kernel_size=1,
                    vision_use_head=False,
                )
            except Exception as e:
                logger.warning("failed to inject stub Gemma4VisionConfig: %s", e)

    patched_init._swift_patched = True  # type: ignore[attr-defined]
    cls.__init__ = patched_init
    cls._swift_vision_patched = True  # type: ignore[attr-defined]


# Note: _patch_gemma4_missing_vision_config() must NOT run at module import
# time — vllm.model_executor.models.gemma4_mm imports vllm.config indirectly,
# which is still mid-initialization when this plugin loads. Call deferred to
# Platform.pre_register_and_update below.


# ---------------------------------------------------------------------------
# vLLM ModelRegistry pre-registration
#
# After AutoConfig accepts the checkpoint, vLLM's own ModelRegistry validator
# rejects architectures it doesn't know. SwiftMetalWorker hijacks execution
# regardless, but vLLM has to instantiate *something* to pass validation.
# Alias every arch we run via mlx-swift-lm to vLLM's generic
# TransformersForCausalLM stub — it won't actually run (Swift worker takes
# over), it just needs to construct.
# ---------------------------------------------------------------------------

_ARCHITECTURES_WE_RUN_VIA_MLX_SWIFT_LM = (
    "DeepseekV4ForCausalLM",
    "DflashQwen3ForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3NextForCausalLM",
    "Qwen3MoeForCausalLM",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "MiniMaxM2ForCausalLM",
    "NemotronHForCausalLM",
    "GptOssForCausalLM",
    "Qwen2_5_VLForConditionalGeneration",
)


_archs_registered: bool = False


def _register_passthrough_archs() -> None:
    """Register stub-aliases for archs we run via mlx-swift-lm.

    Must be called AFTER vllm.config has finished initializing — calling
    it at platform-plugin-import time triggers a circular import (vllm.config
    is mid-init because plugin discovery runs FROM vllm.config). The wrapper
    in `_install_engine_args_hook` ensures we run lazily right before
    EngineArgs.create_model_config validates architectures.
    """
    global _archs_registered
    if _archs_registered:
        return

    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except Exception as e:
        logger.warning("vLLM ModelRegistry not importable; skipping arch pre-register: %s", e)
        return

    try:
        supported = set(ModelRegistry.get_supported_archs())
    except Exception:
        supported = set()

    registered: list[str] = []
    for arch in _ARCHITECTURES_WE_RUN_VIA_MLX_SWIFT_LM:
        if arch in supported:
            continue
        try:
            ModelRegistry.register_model(
                arch,
                "vllm.model_executor.models.transformers:TransformersForCausalLM",
            )
            registered.append(arch)
        except Exception as e:
            logger.warning("ModelRegistry.register_model failed for %s: %s", arch, e)
    logger.info(
        "Swift Metal pre-registered %d architectures with vLLM ModelRegistry",
        len(registered),
    )
    _archs_registered = True




class SwiftMetalPlatform(Platform):
    """Platform for Apple Silicon using Swift/MLX inference engine."""

    _enum: PlatformEnum = PlatformEnum.OOT
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "Apple Silicon (Swift Metal)"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    @classmethod
    def get_device_available_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().available

    @classmethod
    def is_available(cls) -> bool:
        if py_platform.machine() != "arm64":
            return False
        if py_platform.system() != "Darwin":
            return False
        # Check if the Swift bridge dylib can be loaded — that's the real requirement.
        # No Python MLX dependency needed.
        try:
            import os

            from vllm_swift.engine_bridge import _find_lib_path

            return os.path.exists(_find_lib_path())
        except Exception:
            return True  # arm64 + Darwin = assume available

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        return DeviceCapability(major=8, minor=0)

    @classmethod
    def get_device_count(cls) -> int:
        return 1

    @classmethod
    def set_device(cls, device_id: int) -> None:
        pass

    @classmethod
    def current_device(cls) -> int:
        return 0

    @classmethod
    def synchronize(cls, device_id: int = 0) -> None:
        try:
            import mlx.core as mx

            mx.synchronize()
        except (ImportError, AttributeError):
            pass

    @classmethod
    def get_torch_device(cls, device_id: int = 0) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        """vLLM's documented hook for OOT plugins to register custom
        configs/archs *before* VllmConfig is built. By the time this
        fires, vllm.config has finished initializing — so it's safe to
        import the registry without tripping the circular-import that
        plagues module-import-time registration.
        """
        _register_passthrough_archs()
        _patch_gemma4_missing_vision_config()

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config

        # Use our Swift worker
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_swift.worker.SwiftMetalWorker"

        # Single-process — no IPC overhead
        if parallel_config.distributed_executor_backend in ("auto", None):
            parallel_config.distributed_executor_backend = "uni"

        parallel_config.disable_custom_all_reduce = True

        # Swift engine manages its own KV cache — disable prefix caching
        # to use the simpler UnitaryKVCacheCoordinator
        cache_config = vllm_config.cache_config
        cache_config.enable_prefix_caching = False

        # Disable chunked prefill — Swift engine handles full sequences
        if getattr(scheduler_config, "enable_chunked_prefill", False):
            scheduler_config.enable_chunked_prefill = False

        # Ensure scheduler can handle full prompt in one step
        if model_config is not None:
            model_max = model_config.max_model_len
            if scheduler_config.max_num_batched_tokens < model_max:
                scheduler_config.max_num_batched_tokens = model_max

        logger.info("Swift Metal platform configured (uni-proc, Swift engine)")


class SwiftMetalPlatformPlugin:
    """Plugin entry point for vLLM platform system."""

    @staticmethod
    def register() -> str | None:
        if SwiftMetalPlatform.is_available():
            logger.info("Swift Metal platform plugin activated")
            return "vllm_swift.platform.SwiftMetalPlatform"
        return None
