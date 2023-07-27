
import mxnet as mx
from gluonts.mx.trainer.callback import Callback

class EarlyStopping(Callback):
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e4,
                 mode: str = 'min',
                 restore_best_network: bool = True,):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.restore_best_network = restore_best_network

        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = -float('inf')

    def on_validation_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: mx.gluon.nn.HybridBlock,
        trainer: mx.gluon.Trainer,
    ) -> bool:
        should_continue = True

        if self.mode == 'min':
            score_improved = epoch_loss < self.best_score
        else:
            score_improved = epoch_loss > self.best_score

        if score_improved:
            self.best_score = epoch_loss
            print("better network found at epoch %s with loss %s"% (epoch_no, self.best_score)) 
            

            if self.restore_best_network:
                training_network.save_parameters("best_network.params")

            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                should_continue = False
                print(
                    f"EarlyStopping callback initiated stop of training at epoch {epoch_no}."
                )

                if self.restore_best_network:
                    print(
                        f"Restoring best network from epoch {epoch_no - self.patience}."
                    )
                    training_network.load_parameters("best_network.params")

        return should_continue