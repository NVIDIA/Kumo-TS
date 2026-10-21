# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the training timestep labels used by DPM-Solver++ inference."""

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.main_model import TSDiffuser_base
from utils.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP


class TimestepRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.timesteps = []
        self.inputs = []

    def forward(self, diff_input, side_info, timesteps, strategy_type):
        self.timesteps.append(timesteps.detach().clone())
        self.inputs.append(diff_input.detach().clone())
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


@pytest.mark.parametrize("schedule_name,training_steps", [("quad", 50), ("linear", 100)])
@pytest.mark.parametrize("unconditional", [False, True])
def test_clipped_schedule_labels_match_solver_noise_levels(monkeypatch, schedule_name, training_steps, unconditional):
    model = sampler_model(training_steps, unconditional)
    if schedule_name == "quad":
        model.beta = np.linspace(0.0001**0.5, 0.5**0.5, model.num_steps) ** 2
    else:
        model.beta = np.linspace(0.0001, 0.5, model.num_steps)
    model.alpha_hat = 1 - model.beta
    model.alpha = model.alpha_hat.cumprod()
    schedule = NoiseScheduleVP(schedule="discrete", betas=torch.tensor(model.beta, dtype=torch.float32))
    assert schedule.total_N < model.num_steps
    if schedule_name == "quad":
        assert schedule.total_N == 49
    expected = [schedule.total_N - 1, schedule.total_N // 2, 1, 0]
    noise_levels = []

    def sample_at_effective_grid(solver, x, **kwargs):
        for label in expected:
            time = torch.tensor((label + 1) / solver.noise_schedule.total_N, dtype=torch.float64)
            noise_levels.append(solver.noise_schedule.marginal_alpha(time).item())
            solver.model(x, time)
        return x

    monkeypatch.setattr(DPM_Solver, "sample", sample_at_effective_grid)
    data = torch.ones(2, 2, 4)
    model.impute(data, torch.zeros_like(data), torch.empty(0), 1, torch.zeros(2, dtype=torch.long))

    labels = [int(t[0]) for t in model.diffmodel.timesteps]
    assert labels == expected
    np.testing.assert_allclose(noise_levels, np.sqrt(model.alpha[labels]), rtol=1e-5)


@pytest.mark.parametrize("noise_value", [0.0, 1.0])
def test_unconditional_conditioning_uses_cumulative_forward_noise(monkeypatch, noise_value):
    model = sampler_model(500, True)
    labels = [499, 250, 123, 0, 123]
    noise_draws = []

    def fixed_noise(reference, generators=None):
        noise_draws.append(reference.shape)
        return torch.full_like(reference, noise_value)

    def sample_at_training_times(solver, x, **kwargs):
        for label in labels:
            time = torch.tensor((label + 1) / solver.noise_schedule.total_N, dtype=torch.float64)
            solver.model(torch.zeros_like(x), time)
        return x

    monkeypatch.setattr(model, "_randn_like_per_window", fixed_noise)
    monkeypatch.setattr(DPM_Solver, "sample", sample_at_training_times)
    data = torch.arange(1, 17, dtype=torch.float32).reshape(2, 2, 4)
    mask = torch.zeros_like(data)
    mask[:, :, ::2] = 1
    model.impute(data, mask, torch.empty(0), 1, torch.zeros(2, dtype=torch.long))

    for label, diff_input in zip(labels, model.diffmodel.inputs, strict=True):
        alpha = model.alpha[label]
        expected = (np.sqrt(alpha) * data + np.sqrt(1 - alpha) * noise_value) * mask
        torch.testing.assert_close(diff_input[:, 0], expected)
    # One initial sample plus one cached conditioning draw per distinct label.
    assert len(noise_draws) == 1 + len(set(labels))
