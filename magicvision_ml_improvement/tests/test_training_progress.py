import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from training_progress import TrainingProgress, duration


class FakeRun:
    def __init__(self, tmp_path, clock, interrupt=False):
        self.callbacks = {}
        self.clock = clock
        self.interrupt = interrupt
        self.trainer = SimpleNamespace(epochs=10, start_epoch=3, epoch=3, stop=False,
                                      args=SimpleNamespace(patience=20, imgsz=640, optimizer='AdamW'),
                                      batch_size=4, save_dir=tmp_path, best=tmp_path/'best.pt',
                                      last=tmp_path/'last.pt')

    def add_callback(self, name, callback):
        self.callbacks[name] = callback

    def train(self, **kwargs):
        trainer = self.trainer
        self.callbacks['on_train_start'](trainer)
        for epoch in (3, 4):
            trainer.epoch = epoch
            self.callbacks['on_train_epoch_start'](trainer)
            self.clock[0] += 60
            if self.interrupt and epoch == 4:
                raise KeyboardInterrupt()
            trainer.stop = epoch == 4
            self.callbacks['on_fit_epoch_end'](trainer)
        # Real Ultralytics final validation logs at epoch + 1 without training it.
        trainer.epoch += 1
        self.clock[0] += 15
        self.callbacks['on_fit_epoch_end'](trainer)
        trainer.epoch -= 1
        trainer.best.write_bytes(b'test fixture')


def test_resumed_epoch_counts_timing_and_final_validation_are_distinct(tmp_path, capsys):
    clock = [0]
    progress = TrainingProgress(clock=lambda: clock[0])
    result = progress.run(FakeRun(tmp_path, clock), resume=True)
    assert result['epochs_completed_this_run'] == 2
    assert result['last_completed_epoch'] == 5
    assert result['elapsed_seconds'] == 135
    assert result['epoch_seconds_including_validation'] == [60, 60]
    assert result['stopped_before_max_epochs'] is True
    assert result['best_weights'] == str(tmp_path/'best.pt')
    assert json.loads((tmp_path/'training_summary.json').read_text()) == result
    output = capsys.readouterr().out
    assert 'Epoca 4/10' in output and 'Epoca 5/10' in output
    assert 'Epoca 6/10' not in output
    assert 'Restante estimado hasta 10 epocas: 00:06:00' in output
    assert '00:02:15' in output


def test_interruption_preserves_only_finished_epochs_and_reraises(tmp_path):
    clock = [0]
    with pytest.raises(KeyboardInterrupt):
        TrainingProgress(clock=lambda: clock[0]).run(FakeRun(tmp_path, clock, interrupt=True))
    result = json.loads((tmp_path/'training_summary.json').read_text())
    assert result['status'] == 'interrupted'
    assert result['epochs_completed_this_run'] == 1
    assert result['last_completed_epoch'] == 4
    assert result['elapsed_seconds'] == 120
    assert result['best_weights'] is None


def test_duration_includes_hours():
    assert duration(7384.9) == '02:03:04'
