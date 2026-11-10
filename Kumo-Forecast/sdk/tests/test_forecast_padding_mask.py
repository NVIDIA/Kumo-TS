# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Masked (unobserved) context rows must not influence the forecast, including past the first autoregressive block."""

import numpy as np
import pytest
import torch
from backbone import BackbonePipeline
from sdk.forecasting import autoregressive_forecast

SEQ_LEN = 64
MODEL_HORIZON = 16
N_PAD = 40  # leading rows that are padding


def _tiny_model() -> BackbonePipeline:
    t5 = {
        "d_ff": 32, "d_kv": 8, "d_model": 16, "dropout_rate": 0.0, "feed_forward_proj": "gated-gelu",
        "is_encoder_decoder": True, "model_type": "t5", "num_decoder_layers": 1, "num_heads": 2, "num_layers": 1,
        "relative_attention_num_buckets": 8, "relative_attention_max_distance": 16, "vocab_size": 32,
    }  # fmt: skip
    config = {
        "task_name": "forecasting", "model_name": "MOMENT", "transformer_type": "encoder_only", "d_model": None,
        "seq_len": SEQ_LEN, "patch_len": 8, "patch_stride_len": 8, "transformer_backbone": "google/flan-t5-small",
        "t5_config": t5, "randomly_initialize_backbone": True, "forecast_horizon": MODEL_HORIZON,
        "head_dropout": 0.0, "patch_dropout": 0.0,
    }  # fmt: skip
    torch.manual_seed(0)
    model = BackbonePipeline(config, model_kwargs={"task_name": "forecasting", "forecast_horizon": MODEL_HORIZON})
    model.init()
    return model.eval()


def _batch(pad_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 2, SEQ_LEN, generator=g)  # observed rows identical in every call
    pad = torch.randn(1, 2, N_PAD, generator=torch.Generator().manual_seed(pad_seed)) * 10
    x[:, :, :N_PAD] = pad
    mask = torch.ones(1, SEQ_LEN, dtype=torch.long)
    mask[:, :N_PAD] = 0
    return x, mask


def test_forecast_ignores_masked_padding_values() -> None:
    model = _tiny_model()
    (x_a, mask), (x_b, _) = _batch(10), _batch(11)
    with torch.no_grad():
        a = model(x_enc=x_a, input_mask=mask).forecast
        b = model(x_enc=x_b, input_mask=mask).forecast
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("target_horizon", [MODEL_HORIZON * 2, MODEL_HORIZON * 3 + 5])
def test_autoregressive_forecast_keeps_padding_masked_past_the_first_block(target_horizon: int) -> None:
    model = _tiny_model()
    (x_a, mask), (x_b, _) = _batch(10), _batch(11)
    with torch.no_grad():
        a = autoregressive_forecast(model, x_a, mask, MODEL_HORIZON, target_horizon, None, "cpu")
        b = autoregressive_forecast(model, x_b, mask, MODEL_HORIZON, target_horizon, None, "cpu")
    assert a.shape[-1] == target_horizon
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
