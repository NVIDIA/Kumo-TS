# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.main_model import TSDiffuser_Generic
from sdk import inference_ad


def _config(target_dim: int = 3) -> dict:
    return {
        "model": {
            "target_dim": target_dim,
            "is_unconditional": 0,
            "timeemb": 4,
            "featureemb": 4,
            "target_strategy": "random",
            "use_aux_loss": False,
        },
        "dataset": {"scale_factor": 1.0},
        "diffusion": {
            "layers": 1,
            "channels": 8,
            "nheads": 2,
            "diffusion_embedding_dim": 8,
            "beta_start": 0.0001,
            "beta_end": 0.01,
            "num_steps": 4,
            "schedule": "linear",
        },
    }


def _model(mode: str = "positional", num_active_features: int | None = None) -> TSDiffuser_Generic:
    model = TSDiffuser_Generic(
        _config(),
        device=torch.device("cpu"),
        target_dim=3,
        feature_embedding_mode=mode,
        num_active_features=num_active_features,
    ).eval()
    with torch.no_grad():
        model.embed_layer.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, -1.0],
                    [1.0, 2.0, 3.0, 4.0],
                ]
            )
        )
    return model


def _feature_side_info(model: TSDiffuser_Generic) -> torch.Tensor:
    timepoints = torch.arange(2, dtype=torch.float32).unsqueeze(0)
    cond_mask = torch.ones(1, 3, 2)
    side_info = model.get_side_info(timepoints, cond_mask)
    return side_info[:, model.emb_time_dim :, :, :]


def test_default_mode_preserves_distinct_checkpoint_embeddings() -> None:
    model = _model()
    feature_info = _feature_side_info(model)
    assert not torch.equal(feature_info[:, :, 0, :], feature_info[:, :, 1, :])
    expected = model.embed_norm(model.embed_layer(torch.arange(model.target_dim)))
    torch.testing.assert_close(model._feature_embedding_for_side_info(), expected, rtol=0, atol=0)


def test_shared_mode_uses_norm_matched_mean_without_rewriting_checkpoint() -> None:
    model = _model("shared_mean_norm_matched_zero_pad", num_active_features=2)
    original_weight = model.embed_layer.weight.detach().clone()

    raw = original_weight.mean(dim=0)
    reference_norm = original_weight.norm(dim=1).mean()
    matched = raw * reference_norm / raw.norm()
    expected = model.embed_norm(matched.unsqueeze(0))[0]
    actual = model._feature_embedding_for_side_info()

    torch.testing.assert_close(matched / matched.norm(), raw / raw.norm())
    torch.testing.assert_close(matched.norm(), reference_norm)
    torch.testing.assert_close(actual[0], expected)
    torch.testing.assert_close(actual[1], expected)
    torch.testing.assert_close(actual[2], torch.zeros_like(actual[2]), rtol=0, atol=0)
    torch.testing.assert_close(model.embed_layer.weight, original_weight, rtol=0, atol=0)
    assert model.state_dict()["embed_layer.weight"].shape == (3, 4)


def test_norm_matched_zero_pad_uses_shared_vector_only_for_active_positions() -> None:
    model = _model("shared_mean_norm_matched_zero_pad", num_active_features=2)
    feature_embed = model.embed_layer(torch.arange(model.target_dim))
    shared = model._shared_feature_vector(feature_embed, norm_matched=True)
    expected = model.embed_norm(shared.unsqueeze(0))[0]

    actual = model._feature_embedding_for_side_info()

    torch.testing.assert_close(actual[0], expected)
    torch.testing.assert_close(actual[1], expected)
    torch.testing.assert_close(actual[2], torch.zeros_like(actual[2]), rtol=0, atol=0)


@pytest.mark.parametrize("num_active_features", [0, -1, 4, 1.5, True])
def test_invalid_num_active_features_fails_explicitly(num_active_features: object) -> None:
    with pytest.raises(ValueError, match="num_active_features must be an integer"):
        TSDiffuser_Generic(
            _config(),
            device=torch.device("cpu"),
            target_dim=3,
            feature_embedding_mode="shared_mean_norm_matched_zero_pad",
            num_active_features=num_active_features,
        )


def test_zero_pad_mode_requires_num_active_features() -> None:
    with pytest.raises(ValueError, match="num_active_features is required"):
        _model("shared_mean_norm_matched_zero_pad")


def test_zero_pad_matches_norm_matched_when_all_features_are_active() -> None:
    model = _model("shared_mean_norm_matched_zero_pad", num_active_features=3)
    feature_embed = model.embed_layer(torch.arange(model.target_dim))
    shared = model._shared_feature_vector(feature_embed, norm_matched=True)
    expected = model.embed_norm(shared.unsqueeze(0)).expand(model.target_dim, -1)

    torch.testing.assert_close(model._feature_embedding_for_side_info(), expected, rtol=0, atol=0)


def test_embedding_diagnostics_contain_norms_and_feature_counts() -> None:
    diagnostics = _model(
        "shared_mean_norm_matched_zero_pad",
        num_active_features=2,
    ).get_feature_embedding_diagnostics()

    assert diagnostics["embedding_weight_shape"] == [3, 4]
    assert diagnostics["norm_matched_shared_l2_norm"] == pytest.approx(
        diagnostics["mean_embedding_row_l2_norm"]
    )
    assert diagnostics["raw_mean_to_reference_norm_ratio"] > 0
    assert diagnostics["num_active_features"] == 2
    assert diagnostics["num_padded_features"] == 1


@pytest.mark.parametrize("mode", ["unknown", "shared_mean", "shared_mean_norm_matched"])
def test_invalid_feature_embedding_mode_fails_explicitly(mode: str) -> None:
    with pytest.raises(ValueError, match="feature_embedding_mode must be one of"):
        TSDiffuser_Generic(
            _config(),
            device=torch.device("cpu"),
            target_dim=3,
            feature_embedding_mode=mode,
        )


def test_single_gpu_api_defaults_to_shared_embedding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

        def load_state_dict(self, _state) -> None:
            pass

        def to(self, _device):
            return self

        def eval(self):
            return self

    model_path = tmp_path / "model.pth"
    config_path = tmp_path / "config.yaml"
    model_path.touch()
    config_path.touch()
    monkeypatch.setattr(
        inference_ad,
        "_resolve_model_paths",
        lambda *_args, **_kwargs: (str(model_path), str(config_path)),
    )
    monkeypatch.setattr(
        inference_ad.torch,
        "load",
        lambda *_args, **_kwargs: {"config": _config(), "model": {}},
    )
    monkeypatch.setattr(inference_ad, "TSDiffuser_Generic", FakeModel)
    monkeypatch.setattr(inference_ad, "get_dataloader", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(
        inference_ad,
        "evaluate_ad_tesseract2",
        lambda *_args, **_kwargs: {
            "residual": np.zeros(1),
            "residual_l2": np.zeros(1),
            "target": np.zeros((1, 3)),
            "recon": np.zeros((1, 3)),
        },
    )

    result = inference_ad.inference_ad_tesseract2(
        pd.DataFrame([[1.0, 2.0, 3.0]]),
    )

    assert captured["feature_embedding_mode"] == "shared_mean_norm_matched_zero_pad"
    assert captured["num_active_features"] == 3
    assert result["target_dim"] == 3
    assert result["feature_embedding_mode"] == "shared_mean_norm_matched_zero_pad"


def test_multi_gpu_fallback_forwards_default_shared_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        inference_ad,
        "_resolve_model_paths",
        lambda *_args, **_kwargs: ("model.pth", "config.yaml"),
    )
    monkeypatch.setattr(inference_ad.torch.cuda, "is_available", lambda: False)

    def fake_single_gpu(data, model_path, **kwargs):
        captured["data"] = data
        captured["model_path"] = model_path
        captured.update(kwargs)
        return {"residual": np.zeros(1)}

    monkeypatch.setattr(inference_ad, "inference_ad_tesseract2", fake_single_gpu)
    frame = pd.DataFrame([[1.0, 2.0, 3.0]])

    inference_ad.inference_ad_tesseract2_mp(frame)

    assert captured["data"] is frame
    assert captured["model_path"] == "model.pth"
    assert captured["feature_embedding_mode"] == "shared_mean_norm_matched_zero_pad"


def test_single_gpu_api_accepts_explicit_positional_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        def load_state_dict(self, _state) -> None:
            pass

        def to(self, _device):
            return self

        def eval(self):
            return self

    model_path = tmp_path / "model.pth"
    config_path = tmp_path / "config.yaml"
    model_path.touch()
    config_path.touch()
    monkeypatch.setattr(
        inference_ad,
        "_resolve_model_paths",
        lambda *_args, **_kwargs: (str(model_path), str(config_path)),
    )
    monkeypatch.setattr(
        inference_ad.torch,
        "load",
        lambda *_args, **_kwargs: {"config": _config(), "model": {}},
    )
    monkeypatch.setattr(inference_ad, "TSDiffuser_Generic", FakeModel)
    monkeypatch.setattr(inference_ad, "get_dataloader", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(
        inference_ad,
        "evaluate_ad_tesseract2",
        lambda *_args, **_kwargs: {
            "residual": np.zeros(1),
            "residual_l2": np.zeros(1),
            "target": np.zeros((1, 3)),
            "recon": np.zeros((1, 3)),
        },
    )

    inference_ad.inference_ad_tesseract2(
        pd.DataFrame([[1.0, 2.0]]),
        feature_embedding_mode="positional",
    )

    assert captured["feature_embedding_mode"] == "positional"
    assert captured["num_active_features"] == 2


def test_multi_gpu_worker_receives_mode_and_active_count(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        inference_ad,
        "_resolve_model_paths",
        lambda *_args, **_kwargs: ("model.pth", "config.yaml"),
    )
    monkeypatch.setattr(inference_ad.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(inference_ad.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        inference_ad.torch,
        "load",
        lambda *_args, **_kwargs: {"config": _config(), "model": {}},
    )
    monkeypatch.setattr(
        inference_ad,
        "preprocess_dataframe",
        lambda *_args, **_kwargs: torch.zeros((3, 3)),
    )
    monkeypatch.setattr(inference_ad, "_build_window_indexes", lambda *_args: [0])
    monkeypatch.setattr(
        inference_ad,
        "_build_window_tensor",
        lambda *_args: np.zeros((1, 3, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        inference_ad,
        "_create_shared_memory",
        lambda value: {"name": "test-shm", "shape": value.shape, "dtype": value.dtype},
    )
    monkeypatch.setattr(inference_ad, "_cleanup_shared_memory", lambda _info: None)
    merged_result = {"residual": np.zeros(1)}
    monkeypatch.setattr(inference_ad, "_merge_chunked_results", lambda *_args, **_kwargs: merged_result)

    class FakeProcess:
        returncode = 0

        def __init__(self, command: list[str], **_kwargs: object) -> None:
            args_path = Path(command[-2])
            result_path = Path(command[-1])
            captured.update(json.loads(args_path.read_text()))
            result_path.write_text(json.dumps({"gpu_id": 0, "results": {"residual": [0.0]}}))

        def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    monkeypatch.setattr(inference_ad.subprocess, "Popen", FakeProcess)

    result = inference_ad.inference_ad_tesseract2_mp(
        pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        gpu_ids=[0],
        num_processes=1,
        feature_embedding_mode="positional",
    )

    assert captured["feature_embedding_mode"] == "positional"
    assert captured["num_active_features"] == 2
    assert captured["valid_feature_mask"] == [True, True, False]
    assert result["feature_embedding_mode"] == "positional"
