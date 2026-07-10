"""Brain — the per-vehicle model, optimizer and training step.

A faithful port of the consumer's ``brain.py`` (sereBench): it dispatches on
``model_type`` to build one of the vendored architectures (MLP / CNN1D /
TabResNet, all sharing the 2-D manifold bottleneck), trains it with a single
CrossEntropy objective over the 3 classes (NORMAL / ANOMALY / ATTACK), and
supports the FedProx proximal term and global-weight replacement used by
federated learning. The only change from upstream is importing the vendored
model classes; the numerical behaviour is identical.
"""

import logging
from threading import Lock

import torch
import torch.nn as nn
import torch.optim as optim

from .models import MLP, CNN1D, TabResNet

logger = logging.getLogger(__name__)

_MODEL_TYPE_TO_CLASS = {'cnn': CNN1D, 'resnet': TabResNet, 'mlp': MLP}


class Brain:
    def __init__(self, **kwargs):
        self.seed = kwargs.get('seed', None)
        if self.seed is not None:
            torch.manual_seed(self.seed)

        model_type = str(kwargs.get('model_type', 'mlp')).lower()
        resolved_class = _MODEL_TYPE_TO_CLASS.get(model_type, MLP)
        if model_type not in _MODEL_TYPE_TO_CLASS:
            logger.warning(f"Unknown model_type='{model_type}'; falling back to MLP.")

        if model_type == 'cnn':
            self.model = CNN1D(**kwargs)
        elif model_type == 'resnet':
            self.model = TabResNet(**kwargs)
        else:
            self.model = MLP(**kwargs)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Architecture '{type(self.model).__name__}' instantiated "
            f"({total_params} parameters)."
        )

        # Xavier init exactly as the consumer does: applied only for the
        # 'xavier' strategy, otherwise the framework defaults are kept.
        init_strategy = kwargs.get('initialization_strategy', None)
        if init_strategy == 'xavier':
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        optim_class_name = kwargs.get('optimizer')
        self.main_stream_optimizer = getattr(optim, optim_class_name)(
            self.model.parameters(), lr=kwargs.get('learning_rate'))
        self.main_stream_loss_function = nn.CrossEntropyLoss()

        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.model.to(self.device)
        self.model_lock = Lock()
        self.model_saving_path = kwargs.get('model_saving_path', 'default_model.pth')

        # FedProx: proximal coefficient (0 disables the term, recovering FedAvg).
        self.fedprox_mu = kwargs.get('fedprox_mu', 0.0)
        self.global_weights = None

    def train_step(self, feats, main_labels):
        with self.model_lock:
            self.model.train()
            self.main_stream_optimizer.zero_grad()

            main_pred, _ = self.model(feats)
            loss = self.main_stream_loss_function(main_pred, main_labels.squeeze())

            if self.fedprox_mu > 0.0 and self.global_weights is not None:
                prox_term = sum(
                    ((param - self.global_weights[name].to(self.device)) ** 2).sum()
                    for name, param in self.model.named_parameters()
                    if name in self.global_weights
                )
                loss = loss + (self.fedprox_mu / 2.0) * prox_term

            loss.backward()
            self.main_stream_optimizer.step()
            return main_pred.detach(), loss.item()

    def get_brain_state_copy(self):
        with self.model_lock:
            return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def save_model(self):
        with self.model_lock:
            torch.save(self.model.state_dict(), self.model_saving_path)

    def set_global_reference(self, weights):
        with self.model_lock:
            self.global_weights = {k: v.detach().clone().to(self.device) for k, v in weights.items()}

    def update_weights(self, new_weights):
        with self.model_lock:
            main_stream_optimizer_state = self.main_stream_optimizer.state_dict()
            self.model.load_state_dict(new_weights)
            for param in self.model.parameters():
                param.data = param.data.to(self.device)
            main_stream_optim_class = self.main_stream_optimizer.__class__
            self.main_stream_optimizer = main_stream_optim_class(
                self.model.parameters(),
                **{key: value for key, value in main_stream_optimizer_state['param_groups'][0].items()
                   if key != 'params'}
            )
