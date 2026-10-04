"""Shared-device controls and isolated output checks for Q distillation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scripts import train_qnet as qnet_soft


def test_original_positional_training_interface_remains_valid() -> None:
    args = qnet_soft.parse_args(["rank_cnn", "240", "3", "0.013", "0", "q6144_*.pkl"])
    assert args.batch_size == 512
    assert args.device == "auto"
    assert args.tau == 0.013
    assert args.data_glob == "q6144_*.pkl"
    assert args.lr_schedule == "constant"
    assert args.learning_rate == 3e-4


@pytest.mark.parametrize(
    "extra",
    [
        ["--learning-rate", "0"],
        ["--learning-rate", "nan"],
        ["--warmup-epochs", "-1"],
        ["--warmup-epochs", "10"],
        ["--warmup-start-factor", "0"],
        ["--warmup-start-factor", "1.1"],
        ["--min-learning-rate", "0.1"],
        ["--min-learning-rate", "nan"],
    ],
)
def test_invalid_schedule_configuration_is_rejected(extra) -> None:
    with pytest.raises(SystemExit):
        qnet_soft.parse_args(["rank_cnn", "10", "1", "--lr-schedule", "warmup_cosine", *extra])


def test_warmup_cosine_has_exact_endpoints_and_never_rebounds() -> None:
    factors = [
        qnet_soft.warmup_cosine_factor(
            step,
            total_steps=10,
            warmup_steps=3,
            start_factor=0.1,
            min_factor=0.01,
        )
        for step in range(13)
    ]
    assert factors[:3] == pytest.approx([0.1, 0.55, 1])
    assert factors[3] == pytest.approx(1)
    assert factors[9:] == pytest.approx([0.01] * 4)
    assert all(a >= b for a, b in zip(factors[3:-1], factors[4:], strict=True))


def test_cosine_handles_no_warmup_and_single_step_warmup() -> None:
    args = {"total_steps": 4, "start_factor": 0.1, "min_factor": 0}
    assert qnet_soft.warmup_cosine_factor(0, warmup_steps=0, **args) == 1
    assert qnet_soft.warmup_cosine_factor(3, warmup_steps=0, **args) == 0
    assert qnet_soft.warmup_cosine_factor(0, warmup_steps=1, **args) == 1
    assert qnet_soft.warmup_cosine_factor(3, warmup_steps=1, **args) == 0


def test_scheduler_applies_first_update_lr_and_counts_partial_batches() -> None:
    model = torch.nn.Linear(1, 1)
    args = qnet_soft.parse_args(
        [
            "rank_cnn",
            "3",
            "1",
            "--lr-schedule",
            "warmup_cosine",
            "--warmup-epochs",
            "1",
            "--batch-size",
            "2",
            "--min-learning-rate",
            "3e-6",
        ]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = qnet_soft.build_scheduler(optimizer, args, training_examples=3)
    rates = []
    for _ in range(6):  # ceil(3 / 2) * 3 epochs, including partial final batches.
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] == pytest.approx(3e-5)
    assert rates[1:3] == pytest.approx([3e-4, 3e-4])
    assert rates[-1] == pytest.approx(3e-6)
    assert scheduler.last_epoch == 6


def test_constant_schedule_does_not_modify_optimizer() -> None:
    args = qnet_soft.parse_args(["rank_cnn", "1", "1"])
    assert qnet_soft.build_scheduler(None, args, training_examples=3) is None


def test_stopping_patience_does_not_expire_during_warmup(tmp_path, monkeypatch) -> None:
    model = torch.nn.Linear(1, 1)
    args = qnet_soft.parse_args(
        [
            "rank_cnn",
            "5",
            "1",
            "--lr-schedule",
            "warmup_cosine",
            "--warmup-epochs",
            "2",
            "--early-stopping-patience",
            "1",
        ]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def fake_train(_model, optimizer, *args, scheduler):
        optimizer.step()
        scheduler.step()
        return 0.5

    monkeypatch.setattr(qnet_soft, "train_epoch", fake_train)
    monkeypatch.setattr(qnet_soft, "_optset_acc", lambda *args, **kwargs: (0.6, 0.3))
    metadata = {}
    qnet_soft.fit(
        model,
        optimizer,
        (None,) * 5,
        tr=np.array([0]),
        va=np.array([1]),
        args=args,
        ckpt=tmp_path / "best.pt",
        metadata_path=tmp_path / "best.json",
        manifest=metadata,
    )
    assert metadata["completed_epochs"] == 3
    assert metadata["stopped_early"] is True
    assert metadata["scheduler_optimizer_steps"] == 3


@pytest.mark.parametrize(
    "extra",
    [
        ["--batch-size", "0"],
        ["--step-pause", "nan"],
        ["--gpu-memory-limit-mib", "-1"],
        ["--cpu-threads", "0"],
    ],
)
def test_invalid_resource_limits_are_rejected(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        qnet_soft.parse_args(["rank_cnn", "1", "1", *extra])


@pytest.mark.parametrize(
    "extra",
    [
        ["--early-stopping-patience", "-1"],
        ["--early-stopping-min-delta", "nan"],
        ["--early-stopping-min-delta", "-0.1"],
        ["--early-stopping-min-delta", "1.1"],
    ],
)
def test_invalid_early_stopping_options_are_rejected(extra) -> None:
    with pytest.raises(SystemExit):
        qnet_soft.parse_args(["rank_cnn", "1", "1", *extra])


def test_stopping_counts_ties_and_regressions_and_resets_on_improvement() -> None:
    stopper = qnet_soft.EarlyStopping(patience=2, min_delta=0)
    assert not stopper.update(0.6)
    assert not stopper.update(0.6)
    assert not stopper.update(0.7)
    assert stopper.stale_epochs == 0
    assert not stopper.update(0.69)
    assert stopper.update(0.7)


def test_small_improvements_accumulate_but_do_not_keep_resetting_patience() -> None:
    stopper = qnet_soft.EarlyStopping(patience=3, min_delta=0.01)
    assert not stopper.update(0.6)
    assert not stopper.update(0.604)
    assert not stopper.update(0.608)
    assert not stopper.update(0.612)
    assert stopper.stale_epochs == 0
    assert not stopper.update(0.612)
    assert not stopper.update(0.612)
    assert stopper.update(0.612)


def test_zero_patience_disables_stopping_and_rejects_invalid_accuracy() -> None:
    stopper = qnet_soft.EarlyStopping(patience=0, min_delta=0)
    for _ in range(100):
        assert not stopper.update(0.5)
    for invalid in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match="validation accuracy"):
            stopper.update(invalid)


@pytest.mark.parametrize(
    "patience,epochs,stopped_early", [(2, 100, True), (0, 3, False), (2, 3, False)]
)
def test_fit_stops_and_retains_strict_best_even_below_min_delta(
    tmp_path,
    monkeypatch,
    patience,
    epochs,
    stopped_early,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    scores = iter([0.6, 0.6005, 0.6004])
    epoch = []

    def fake_train(*args):
        epoch.append(len(epoch) + 1)
        with torch.no_grad():
            model.weight.fill_(len(epoch))
        return 0.5

    def fake_score(_model, indices, *args, **kwargs):
        return (next(scores), 0.3) if indices[0] == 1 else (0.9, 0.3)

    monkeypatch.setattr(qnet_soft, "train_epoch", fake_train)
    monkeypatch.setattr(qnet_soft, "_optset_acc", fake_score)
    args = qnet_soft.parse_args(
        [
            "rank_cnn",
            str(epochs),
            "1",
            "--device",
            "cpu",
            "--early-stopping-patience",
            str(patience),
        ]
    )
    checkpoint, metadata_path = tmp_path / "best.pt", tmp_path / "best.json"
    metadata = {}
    best = qnet_soft.fit(
        model,
        None,
        (None,) * 5,
        tr=np.array([0]),
        va=np.array([1]),
        args=args,
        ckpt=checkpoint,
        metadata_path=metadata_path,
        manifest=metadata,
    )
    assert best == 0.6005
    assert epoch == [1, 2, 3]
    assert metadata["completed_epochs"] == 3
    assert metadata["best_epoch"] == 2
    assert metadata["stopped_early"] is stopped_early
    state = torch.load(checkpoint, weights_only=True)
    assert state["weight"].item() == 2


def test_requested_cuda_does_not_silently_fall_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = qnet_soft.parse_args(["rank_cnn", "1", "1", "--device", "cuda"])
    with pytest.raises(RuntimeError, match="explicitly requested"):
        qnet_soft.configure_device(args)


def test_gpu_limit_only_changes_this_process_allocator(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=100 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (80 * 1024**3, 100 * 1024**3))
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: calls.append((fraction, device)),
    )
    args = qnet_soft.parse_args(
        [
            "rank_cnn",
            "1",
            "1",
            "--device",
            "cuda",
            "--gpu-memory-limit-mib",
            "2048",
        ]
    )
    assert qnet_soft.configure_device(args) == "cuda"
    assert calls == [(0.02, 0)]


def test_yield_waits_for_own_cuda_batch_before_sleep(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(qnet_soft.time, "sleep", calls.append)
    qnet_soft.yield_device("cuda", 0.1)
    assert calls == ["synchronize", 0.1]
    calls.clear()
    qnet_soft.yield_device("cpu", 0)
    assert calls == []


@pytest.mark.parametrize(
    "epochs,completed,stopped,schedule",
    [
        (1, 1, False, "constant"),
        (10, 2, True, "constant"),
        (3, 2, True, "warmup_cosine"),
    ],
)
def test_cpu_smoke_saves_manifest_best_and_verified_numpy_export(
    tmp_path,
    monkeypatch,
    *,
    epochs,
    completed,
    stopped,
    schedule,
) -> None:
    n, n_train = 12, 8
    planes = np.zeros((n, 14, 4, 13), dtype=np.float32)
    scalars = np.zeros((n, 10), dtype=np.float32)
    masks = np.zeros((n, 56), dtype=np.float32)
    masks[:, :2] = 1
    targets = np.zeros((n, 56), dtype=np.float32)
    targets[:, 1] = 0.3
    shard = tmp_path / "shard.pkl"
    shard.write_bytes(b"test dataset signature")
    checkpoint = tmp_path / "run/best.pt"
    export = tmp_path / "run/weights.npz"
    monkeypatch.setattr(
        qnet_soft, "load_features", lambda _: (planes, scalars, masks, targets, n_train)
    )
    monkeypatch.setattr(qnet_soft, "matching_files", lambda _: [str(shard)])
    monkeypatch.setattr(
        qnet_soft, "_build", lambda *args: qnet_soft.ARMS["rank_cnn"](ch=4, layers=1)
    )
    monkeypatch.setattr(qnet_soft, "_optset_acc", lambda *args, **kwargs: (0.5, 0.3))
    monkeypatch.setattr(
        qnet_soft.sys,
        "argv",
        [
            "qnet_soft.py",
            "rank_cnn",
            str(epochs),
            "1",
            "--device",
            "cpu",
            "--cpu-threads",
            "1",
            "--batch-size",
            "3",
            "--early-stopping-patience",
            "1",
            "--lr-schedule",
            schedule,
            "--warmup-epochs",
            "1",
            "--checkpoint",
            str(checkpoint),
            "--export-numpy",
            str(export),
        ],
    )
    qnet_soft.main()
    manifest = json.loads(checkpoint.with_suffix(".json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_epochs"] == completed
    assert manifest["stopped_early"] is stopped
    assert manifest["scheduler_optimizer_steps"] == (completed * 3 if schedule != "constant" else 0)
    assert manifest["training_examples"] == n_train
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    with np.load(export) as archive:
        assert set(archive.files) == set(state)
        for key, value in state.items():
            np.testing.assert_array_equal(archive[key], value.numpy())
    assert not list(checkpoint.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        qnet_soft.main()


def test_existing_manifest_is_not_overwritten_even_if_weights_are_missing(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    metadata = checkpoint.with_suffix(".json")
    metadata.write_text("original run metadata")
    args = qnet_soft.parse_args(["rank_cnn", "1", "1", "--checkpoint", str(checkpoint)])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        qnet_soft.prepare_outputs(args, (1, 1, 1, 1))
    assert metadata.read_text() == "original run metadata"
