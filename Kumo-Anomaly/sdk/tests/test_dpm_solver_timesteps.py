# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the training timestep labels used by DPM-Solver++ inference."""

import sys
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.main_model import TSDiffuser_base
from utils.dpm_solver_pytorch import DPM_Solver


class TimestepRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.timesteps = []

    def forward(self, diff_input, side_info, timesteps, strategy_type):
        self.timesteps.append(timesteps.detach().clone())
        return torch.zeros_like(diff_input[:, 0])


def sampler_model(training_steps, unconditional):
    model = TSDiffuser_base.__new__(TSDiffuser_base)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.num_steps = training_steps
    model.is_unconditional = unconditional
    model.beta = torch.linspace(0.0001, 0.01, training_steps).numpy()
    model.alpha_hat = 1 - model.beta
    model.alpha = model.alpha_hat.cumprod()
    model.add_module("diffmodel", TimestepRecorder())
    return model


@pytest.mark.parametrize("training_steps", [20, 500, 1000])
@pytest.mark.parametrize("unconditional", [False, True])
def test_solver_time_maps_to_training_labels_once(monkeypatch, training_steps, unconditional):
    model = sampler_model(training_steps, unconditional)
    expected = [training_steps - 1, training_steps // 2, 1, 0]

    def sample_at_training_times(solver, x, **kwargs):
        for label in expected:
            time = torch.tensor((label + 1) / training_steps, dtype=torch.float64)
            solver.model(x, time)
        return x

    monkeypatch.setattr(DPM_Solver, "sample", sample_at_training_times)
    data = torch.ones(2, 2, 4)
    mask = torch.zeros_like(data)
    mask[:, :, ::2] = 1
    model.impute(data, mask, torch.empty(0), 1, torch.zeros(2, dtype=torch.long))

    assert [int(t[0]) for t in model.diffmodel.timesteps] == expected
    for timesteps in model.diffmodel.timesteps:
        assert timesteps.dtype == torch.long
        assert timesteps.shape == (2,)
        assert torch.equal(timesteps, timesteps[:1].expand(2))


@pytest.mark.parametrize("unconditional", [False, True])
def test_real_solver_visits_decreasing_timesteps_for_each_sample(unconditional):
    model = sampler_model(500, unconditional)
    data = torch.ones(2, 2, 4)
    mask = torch.zeros_like(data)
    mask[:, :, ::2] = 1
    with torch.no_grad():
        samples = model.impute(data, mask, torch.empty(0), 2, torch.zeros(2, dtype=torch.long), dpm_steps=20)

    assert samples.shape == (2, 2, 2, 4)
    assert torch.isfinite(samples).all()
    labels = [int(t[0]) for t in model.diffmodel.timesteps]
    assert len(labels) == 40
    for trajectory in (labels[:20], labels[20:]):
        assert trajectory[0] == 499
        assert 0 <= trajectory[-1] < 50
        assert all(earlier > later for earlier, later in pairwise(trajectory))
