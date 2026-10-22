"""Console timing and durable progress for a single Ultralytics training run."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class TrainingProgress:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.started = clock()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.epoch_started = None
        self.epoch_seconds = []
        self.last_epoch = None
        self.trainer = None

    def attach(self, model):
        model.add_callback('on_train_start', self.on_start)
        model.add_callback('on_train_epoch_start', self.on_epoch_start)
        model.add_callback('on_fit_epoch_end', self.on_epoch_end)

    def on_start(self, trainer):
        self.trainer = trainer
        print('\n=== ENTRENAMIENTO YOLO26m ===', flush=True)
        print(f'Epocas: hasta {trainer.epochs} | Inicio: {trainer.start_epoch + 1} | '
              f'Paciencia: {trainer.args.patience}', flush=True)
        print(f'Imagen: {trainer.args.imgsz} | Batch: {trainer.batch_size} | '
              f'Optimizador: {trainer.args.optimizer}', flush=True)
        print(f'Resultados: {trainer.save_dir}', flush=True)
        print('El tiempo restante se estimara al completar la primera epoca; '
              'la parada temprana puede acortarlo.\n', flush=True)

    def on_epoch_start(self, trainer):
        self.trainer = trainer
        self.epoch_started = self.clock()

    def on_epoch_end(self, trainer):
        # Ultralytics invokes this hook again during final evaluation of best.pt.
        if self.epoch_started is None or self.last_epoch == trainer.epoch:
            return
        self.last_epoch = trainer.epoch
        self.epoch_seconds.append(self.clock() - self.epoch_started)
        self.epoch_started = None
        completed = trainer.epoch + 1
        remaining = max(0, trainer.epochs - completed)
        eta = remaining * sum(self.epoch_seconds) / len(self.epoch_seconds)
        if trainer.stop:
            estimate = 'Finalizando y evaluando el mejor modelo'
        else:
            estimate = f'Restante estimado hasta {trainer.epochs} epocas: {duration(eta)}'
        print(f'\n[Campus22] Epoca {completed}/{trainer.epochs} | '
              f'Duracion (incluye validacion): {duration(self.epoch_seconds[-1])} | '
              f'Transcurrido: {duration(self.clock() - self.started)} | {estimate}', flush=True)
        self.save('running')

    def save(self, status, error=None):
        trainer = self.trainer
        result = {
            'status': status, 'started_at': self.started_at,
            'elapsed_seconds': self.clock() - self.started,
            'epochs_completed_this_run': len(self.epoch_seconds),
            'last_completed_epoch': self.last_epoch + 1 if self.last_epoch is not None else None,
            'max_epochs': trainer.epochs if trainer is not None else None,
            'epoch_seconds_including_validation': list(self.epoch_seconds),
            'timing_scope': 'model.train call, including trainer setup and final evaluation',
        }
        if error:
            result['error'] = error
        if trainer is not None:
            directory = Path(trainer.save_dir)
            result['run_directory'] = str(directory)
            result['best_weights'] = str(trainer.best) if Path(trainer.best).is_file() else None
            result['last_weights'] = str(trainer.last) if Path(trainer.last).is_file() else None
            result['stopped_before_max_epochs'] = (self.last_epoch is not None
                                                   and self.last_epoch + 1 < trainer.epochs)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'training_summary.json').write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        return result

    def run(self, model, **kwargs):
        self.attach(model)
        try:
            model.train(**kwargs)
        except BaseException as exc:
            result = self.save('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', str(exc))
            print(f'\nEntrenamiento detenido | Tiempo: {duration(result["elapsed_seconds"])} | '
                  f'Epocas completas en esta ejecucion: {result["epochs_completed_this_run"]}', flush=True)
            raise
        result = self.save('completed')
        print('\n=== RESUMEN DEL ENTRENAMIENTO ===', flush=True)
        print(f'Tiempo total (preparacion del entrenador + entrenamiento + evaluacion): '
              f'{duration(result["elapsed_seconds"])}', flush=True)
        print(f'Epocas completadas en esta ejecucion: {result["epochs_completed_this_run"]} | '
              f'Ultima epoca: {result["last_completed_epoch"]}/{result["max_epochs"]}', flush=True)
        if result.get('stopped_before_max_epochs'):
            print('Finalizo antes del maximo de epocas configurado.', flush=True)
        if result.get('best_weights'):
            print(f'Mejor modelo: {result["best_weights"]}', flush=True)
        if result.get('run_directory'):
            print(f'Resumen guardado: {Path(result["run_directory"]) / "training_summary.json"}', flush=True)
        return result
