"""ConsumerNode — the in-process equivalent of Train_IoT_local_anomaly_detection.

A faithful port of ``consume.py`` (sereBench): per vehicle it runs the same set
of cooperating threads —

* **data consuming** — subscribes to ``{v}_anomalies`` / ``{v}_eval_anomalies``
  / ``{v}_normal_data`` on the bus, tensorises each record into the right
  buffer (clean anomaly / clean attack / clean diagnostics / adversarial-eval
  anomaly / adversarial-eval attack) and does online classification;
* **training** — samples a balanced batch from the clean buffers (plus the
  adversarial-eval buffers when ``adversarial_training`` is on), takes one
  gradient step, and every ``epoch_size`` batches reports epoch + online
  metrics; every ``run_benchmarks_freq_epochs`` epochs it runs the Gaussian
  visual eval, the fixed sigma-grid robustness eval, and (in a background
  thread) the HopSkipJump decision-based attack;
* **weight push / pull** — publishes its ``state_dict`` to ``{v}_weights`` and
  applies aggregated ``global_weights`` for federated learning.

Kafka producers/consumers become :class:`MessageBus` calls; the HTTP mitigation
control plane becomes two in-process callables supplied by the orchestrator.
Everything else — buffer routing, metric formulas, benchmark cadence, the HSJA
threat model — matches the platform line for line.
"""

import time
import logging
import threading
from threading import Lock

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from .brain import Brain
from .buffers import Buffer
from .simulator import EventType
from .hopskipjump import hopskipjump_attack_batch

# usBpPres / usMpPres indices in the 40-dim vector; opt-in HSJA subset matching
# the Gaussian Mp_std/Bp_std threat model (see consume.py).
HSJA_PRESSURE_FEATURE_INDICES = [32, 33]
PRESSURE_FEATURE_NAMES = ['usBpPres', 'usMpPres']

# Bookkeeping columns the producer adds and the consumer strips before
# tensorising (verbatim from consume.py).
columns_to_delete = ['Flotta', 'Veicolo', 'Codice', 'Nome', 'Descrizione', 'Timestamp',
                     'Timestamp chiusura', 'Durata', 'Posizione', 'Sistema', 'Componente',
                     'Timestamp segnale', 'Test']

HEALTHY = "HEALTHY"
INFECTED = "INFECTED"


def encode_array(arr):
    """Same hex-encoding the platform uses to ship numpy arrays over Kafka.

    Kept so the wandb logger's decode path is identical to the Dockerised one.
    """
    return {"data": arr.tobytes().hex(), "shape": arr.shape, "dtype": str(arr.dtype)}


class ConsumerNode:
    def __init__(self, vehicle_name, config, bus, get_status, stop_attack):
        self.vehicle_name = vehicle_name
        self.config = dict(config)
        self.bus = bus
        self._get_status = get_status          # callable(v) -> HEALTHY|INFECTED
        self._stop_attack = stop_attack        # callable(v) -> mitigation_time|None

        self.logger = logging.getLogger(f'[{vehicle_name}_CONS]')

        cfg = self.config
        self.batch_size = int(cfg.get('batch_size', 32))
        self.epoch_size = int(cfg.get('epoch_size', 20))
        self.save_model_freq_epochs = int(cfg.get('save_model_freq_epochs', 20))
        self.run_benchmarks_freq_epochs = int(cfg.get('run_benchmarks_freq_epochs',
                                                       self.save_model_freq_epochs))
        self.plot_creation_freq_benchmarks = int(cfg.get('plot_creation_freq_benchmarks', 3))
        self.hsja_enabled = bool(cfg.get('hsja_enabled', True))
        self.eval_sigmas = cfg.get('eval_sigmas', [0.5, 1.0, 1.5, 2.0])
        self.training_freq_seconds = float(cfg.get('training_freq_seconds', 0.01))
        self.weights_push_freq_seconds = float(cfg.get('weights_push_freq_seconds', 40))
        self.weights_pull_freq_seconds = float(cfg.get('weights_pull_freq_seconds', 60))
        self.adversarial_training = bool(cfg.get('adversarial_training', False))
        self.mitigation = bool(cfg.get('mitigation', False))

        self.true_positive_reward = float(cfg.get('true_positive_reward', 2.0))
        self.true_negative_reward = float(cfg.get('true_negative_reward', 0.0))
        self.false_positive_reward = float(cfg.get('false_positive_reward', -4.0))
        self.false_negative_reward = float(cfg.get('false_negative_reward', -10.0))

        # Brain (3-class classifier over NORMAL/ANOMALY/ATTACK).
        cfg['output_dim'] = 3
        self.brain = Brain(**cfg)

        buffer_size = int(cfg.get('buffer_size', 5000))
        self.attacks_buffer = Buffer(buffer_size)
        self.eval_attacks_buffer = Buffer(buffer_size)
        self.anomalies_buffer = Buffer(buffer_size)
        self.eval_anomalies_buffer = Buffer(buffer_size)
        self.diagnostics_buffer = Buffer(buffer_size)

        # Counters / accumulators (globals in consume.py).
        self.batch_counter = 0
        self.epoch_counter = 0
        self.records_processed = 0
        self.attacks_processed = 0
        self.anoms_processed = 0
        self.diagnostics_processed = 0
        self.eval_anomalies_processed = 0
        self.eval_attacks_processed = 0
        self.benchmark_eval_counter = 0

        self._reset_epoch_accumulators()

        self.online_batch_labels = []
        self.online_main_batch_preds = []
        self.mitigation_times = []
        self.mitigation_reward = 0
        self.lists_lock = Lock()

        self.FEATURE_COLUMNS = None
        self.PRESSURE_INDICES = None

        self._hsja_eval_running = False
        self._stop = False
        self._threads = []
        self._weights_consumer = None

    def _reset_epoch_accumulators(self):
        self.epoch_loss = 0
        self.epoch_accuracy = 0
        self.epoch_precision = 0
        self.epoch_recall = 0
        self.epoch_f1 = 0
        self.epoch_macro_f1 = 0

    # -- reporting shims (Kafka -> bus) ------------------------------------
    def _report_metrics(self, metrics):
        stats = {'vehicle_name': self.vehicle_name}
        stats.update(metrics)
        self.bus.produce(f"{self.vehicle_name}_statistics", stats)

    def _push_weights(self, weights):
        # Clone to mimic Kafka's pickle round-trip (no shared tensors across nodes).
        payload = {k: v.detach().clone() for k, v in weights.items()}
        self.bus.produce(f"{self.vehicle_name}_weights", payload)

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._stop = False
        self.bus.create_topic(f"{self.vehicle_name}_statistics")
        self.bus.create_topic(f"{self.vehicle_name}_weights")
        self._weights_consumer = self.bus.consumer(group_id=f"{self.vehicle_name}-cons")
        self._weights_consumer.subscribe(["global_weights"])

        self._threads = [
            threading.Thread(target=self._consume_vehicle_data, daemon=True,
                             name=f"{self.vehicle_name}_consume"),
            threading.Thread(target=self._train_model, daemon=True,
                             name=f"{self.vehicle_name}_train"),
            threading.Thread(target=self._push_weights_loop, daemon=True,
                             name=f"{self.vehicle_name}_push"),
            threading.Thread(target=self._pull_weights_loop, daemon=True,
                             name=f"{self.vehicle_name}_pull"),
        ]
        for t in self._threads:
            t.start()
        self.logger.info(f"Consumer runtime started for vehicle {self.vehicle_name}")

    def stop(self):
        self._stop = True
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []
        if self._weights_consumer is not None:
            self._weights_consumer.close()
        self.logger.info(f"Consumer stopped for vehicle {self.vehicle_name}")

    # -- data consuming (mirror consume.py) --------------------------------
    def _consume_vehicle_data(self):
        # Bound the high-volume telemetry queue so a firehose producer
        # (time_emulation=false) can never grow it without limit; the buffers
        # downstream are recent-data reservoirs, so drop-oldest is harmless.
        consumer = self.bus.consumer(group_id=f"{self.vehicle_name}-data", maxsize=20000)
        consumer.subscribe([
            f"{self.vehicle_name}_anomalies",
            f"{self.vehicle_name}_eval_anomalies",
            f"{self.vehicle_name}_normal_data",
        ])
        try:
            while not self._stop:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                topic, value = msg
                # Copy so the per-node columns_to_delete mutation never touches
                # another subscriber's shared dict (Kafka delivered bytes).
                self._process_message(topic, dict(value))
        finally:
            consumer.close()

    def _process_message(self, topic, msg):
        counting_message = False

        for col in columns_to_delete:
            if col in msg:
                del msg[col]

        if self.FEATURE_COLUMNS is None and 'event_type' in msg:
            self.FEATURE_COLUMNS = [k for k in msg.keys() if k != 'event_type']
            self.PRESSURE_INDICES = [self.FEATURE_COLUMNS.index(n)
                                     for n in PRESSURE_FEATURE_NAMES
                                     if n in self.FEATURE_COLUMNS]
            if not self.PRESSURE_INDICES:
                self.PRESSURE_INDICES = HSJA_PRESSURE_FEATURE_INDICES
            self.logger.info(
                f"[sigma-grid] captured {len(self.FEATURE_COLUMNS)} feature columns; "
                f"pressure indices -> {self.PRESSURE_INDICES}")

        if topic.endswith("_eval_anomalies"):
            if msg['event_type'] == EventType.ANOMALY.value:
                feat, label = self.eval_anomalies_buffer.format(msg)
                self.eval_anomalies_buffer.add(feat, label)
                self.eval_anomalies_processed += 1
            elif msg['event_type'] == EventType.ATTACK.value:
                feat, label = self.eval_attacks_buffer.format(msg)
                self.eval_attacks_buffer.add(feat, label)
                self.eval_attacks_processed += 1

        elif topic.endswith("_anomalies"):
            counting_message = True
            if msg['event_type'] == EventType.ANOMALY.value:
                feat, label = self.anomalies_buffer.format(msg)
                self.anomalies_buffer.add(feat, label)
                self.anoms_processed += 1
            elif msg['event_type'] == EventType.ATTACK.value:
                feat, label = self.attacks_buffer.format(msg)
                self.attacks_buffer.add(feat, label)
                self.attacks_processed += 1

        elif topic.endswith("_normal_data"):
            counting_message = True
            feat, label = self.diagnostics_buffer.format(msg)
            self.diagnostics_buffer.add(feat, label)
            self.diagnostics_processed += 1

        if counting_message:
            self.records_processed += 1
            self._online_classification(feat, label)

        if self.records_processed % 500 == 0 and self.records_processed > 0:
            self.logger.info(
                f"Received {self.records_processed} messages: {self.attacks_processed} attacks, "
                f"{self.anoms_processed} anomalies, {self.diagnostics_processed} diagnostics.")

    # -- online monitoring + in-process mitigation -------------------------
    def _online_classification(self, feat_tensor, main_label_tensor):
        self.brain.model.eval()
        with self.brain.model_lock, torch.no_grad():
            main_pred, _ = self.brain.model(feat_tensor.unsqueeze(0))
            main_pred = main_pred.argmax(dim=1)

        with self.lists_lock:
            self.online_batch_labels.append(main_label_tensor)
            self.online_main_batch_preds.append(main_pred.squeeze())

        self._mitigation_and_rewarding(main_pred, main_label_tensor)

    def _mitigation_and_rewarding(self, prediction, current_label):
        if prediction == 2:
            if current_label == prediction:
                if self.mitigation and self._get_status(self.vehicle_name) == INFECTED:
                    mitigation_time = self._stop_attack(self.vehicle_name)
                    if mitigation_time is not None:
                        with self.lists_lock:
                            self.mitigation_times.append(mitigation_time)
                self.mitigation_reward += self.true_positive_reward
            else:
                self.mitigation_reward += self.false_positive_reward
        else:
            if current_label == prediction:
                self.mitigation_reward += self.true_negative_reward
            else:
                self.mitigation_reward += self.false_negative_reward

    # -- weight push / pull ------------------------------------------------
    def _push_weights_loop(self):
        while not self._stop:
            self._sleep(self.weights_push_freq_seconds)
            if self._stop:
                break
            self._push_weights(self.brain.get_brain_state_copy())

    def _pull_weights_loop(self):
        while not self._stop:
            self._sleep(self.weights_pull_freq_seconds)
            if self._stop:
                break
            msg = self._weights_consumer.poll(timeout=1.0)
            if msg is not None:
                _, new_weights = msg
                self.brain.update_weights(new_weights)
                self.brain.set_global_reference(new_weights)
                self.logger.info("Local weights updated using global model.")

    def _sleep(self, seconds):
        """Interruptible sleep so stop() is responsive during long FL intervals."""
        end = time.time() + seconds
        while not self._stop and time.time() < end:
            time.sleep(min(0.5, max(0.0, end - time.time())))

    # -- benchmarks (mirror consume.py) ------------------------------------
    def _visual_evaluation(self, n=1000, include_plots=True):
        diag_feats, diag_labels = self.diagnostics_buffer.sample(n // 3)
        anom_feats, anom_labels = self.eval_anomalies_buffer.sample(n // 3)
        atk_feats, atk_labels = self.eval_attacks_buffer.sample(n // 3)

        if len(diag_feats) < 10 or len(anom_feats) < 10 or len(atk_feats) < 10:
            return None

        feats = torch.vstack((diag_feats, anom_feats, atk_feats))
        y = torch.vstack((diag_labels, anom_labels, atk_labels))
        self.brain.model.eval()
        with self.brain.model_lock, torch.no_grad():
            preds, manifold = self.brain.model(feats)
            preds = preds.argmax(dim=1)

        y = y.numpy()
        preds = preds.numpy()

        result = {
            'adv_eval_accuracy': accuracy_score(y, preds),
            'adv_eval_precision': precision_score(y, preds, zero_division=0, average='weighted'),
            'adv_eval_recall': recall_score(y, preds, zero_division=0, average='weighted'),
            'adv_eval_f1': f1_score(y, preds, zero_division=0, average='weighted'),
            'adv_eval_macro_f1': f1_score(y, preds, zero_division=0, average='macro'),
        }

        if include_plots:
            adv_eval_cm = confusion_matrix(y, preds, labels=[0, 1, 2])
            X = feats - feats.mean(0, keepdim=True)
            _, _, V = torch.pca_lowrank(X, q=2)
            X2 = X @ V[:, :2]
            result.update({
                'visual_eval_X': encode_array(X2.numpy()),
                'visual_eval_y': encode_array(y),
                'visual_eval_preds': encode_array(preds),
                'visual_eval_manifold': encode_array(manifold.numpy()),
                'adv_eval_confusion_matrix': encode_array(adv_eval_cm),
            })
        return result

    def _sigma_grid_evaluation(self, sigmas, n=300):
        per_class = max(n // 3, 1)
        diag_feats, diag_labels = self.diagnostics_buffer.sample(per_class)
        anom_feats, anom_labels = self.anomalies_buffer.sample(per_class)
        atk_feats, atk_labels = self.attacks_buffer.sample(per_class)

        if len(diag_feats) < 10 or len(anom_feats) < 10 or len(atk_feats) < 10:
            return None

        pressure_idx = self.PRESSURE_INDICES if self.PRESSURE_INDICES else HSJA_PRESSURE_FEATURE_INDICES
        pert_feats = torch.vstack((anom_feats, atk_feats))
        y = torch.vstack((diag_labels, anom_labels, atk_labels)).numpy().ravel()

        result = {}
        for sigma in sigmas:
            sigma = float(sigma)
            noisy = pert_feats.clone()
            noise = torch.randn(noisy.shape[0], len(pressure_idx)) * sigma * 100
            noisy[:, pressure_idx] += noise
            feats = torch.vstack((diag_feats, noisy))
            self.brain.model.eval()
            with self.brain.model_lock, torch.no_grad():
                preds, _ = self.brain.model(feats)
                preds = preds.argmax(dim=1).numpy()

            tag = f"{sigma:g}".replace('.', '_')
            result[f'adv_eval/sigma_{tag}/accuracy'] = accuracy_score(y, preds)
            result[f'adv_eval/sigma_{tag}/precision'] = precision_score(y, preds, zero_division=0, average='weighted')
            result[f'adv_eval/sigma_{tag}/recall'] = recall_score(y, preds, zero_division=0, average='weighted')
            result[f'adv_eval/sigma_{tag}/f1'] = f1_score(y, preds, zero_division=0, average='weighted')
            result[f'adv_eval/sigma_{tag}/macro_f1'] = f1_score(y, preds, zero_division=0, average='macro')
        return result

    def _hsja_evaluation(self, n_per_class, n_steps, n_grad_samples, include_plots,
                         feature_indices, clean_anchors, init_noise_scale):
        anom_buf = self.anomalies_buffer if clean_anchors else self.eval_anomalies_buffer
        atk_buf = self.attacks_buffer if clean_anchors else self.eval_attacks_buffer

        diag_feats, diag_labels = self.diagnostics_buffer.sample(n_per_class)
        anom_feats, anom_labels = anom_buf.sample(n_per_class)
        atk_feats, atk_labels = atk_buf.sample(n_per_class)

        if len(diag_feats) < 5 or len(anom_feats) < 5 or len(atk_feats) < 5:
            self.logger.warning("HSJA eval OMITTED — buffers not warm enough.")
            self._hsja_eval_running = False
            return

        all_feats = torch.vstack((diag_feats, anom_feats, atk_feats))
        all_labels_arr = torch.vstack((diag_labels, anom_labels, atk_labels)).squeeze(1).numpy()

        def predict_batch_fn(X):
            self.brain.model.eval()
            with self.brain.model_lock, torch.inference_mode():
                logits, _ = self.brain.model(X)
                return logits.argmax(dim=-1).tolist()

        if self._stop:
            self._hsja_eval_running = False
            return

        y_orig = torch.as_tensor(all_labels_arr, dtype=torch.long)
        X_adv, n_q = hopskipjump_attack_batch(
            predict_batch_fn, all_feats, y_orig,
            n_steps=n_steps, n_grad_samples=n_grad_samples,
            init_noise_scale=init_noise_scale, feature_indices=feature_indices,
            should_stop=lambda: self._stop,
        )

        total_queries = int(n_q.sum().item())
        pert_norms = torch.norm(X_adv - all_feats, dim=1).tolist()

        with self.brain.model_lock, torch.inference_mode():
            self.brain.model.eval()
            adv_logits, _ = self.brain.model(X_adv)
            adv_preds_arr = adv_logits.argmax(dim=-1).cpu().numpy()

        result = {
            'hsja_adv_eval/accuracy': accuracy_score(all_labels_arr, adv_preds_arr),
            'hsja_adv_eval/precision': precision_score(all_labels_arr, adv_preds_arr, zero_division=0, average='weighted'),
            'hsja_adv_eval/recall': recall_score(all_labels_arr, adv_preds_arr, zero_division=0, average='weighted'),
            'hsja_adv_eval/f1': f1_score(all_labels_arr, adv_preds_arr, zero_division=0, average='weighted'),
            'hsja_adv_eval/macro_f1': f1_score(all_labels_arr, adv_preds_arr, zero_division=0, average='macro'),
            'hsja_adv_eval/avg_perturbation': float(np.mean(pert_norms)),
            'hsja_adv_eval/avg_queries': total_queries / max(len(all_feats), 1),
        }

        if include_plots:
            adv_cm = confusion_matrix(all_labels_arr, adv_preds_arr, labels=[0, 1, 2])
            X = all_feats - all_feats.mean(0, keepdim=True)
            _, _, V = torch.pca_lowrank(X, q=2)
            X2 = (X @ V[:, :2]).numpy()
            with self.brain.model_lock, torch.inference_mode():
                self.brain.model.eval()
                _, adv_manifold = self.brain.model(X_adv)
            adv_manifold = adv_manifold.numpy()
            result.update({
                'hsja_visual_eval_X': encode_array(X2),
                'hsja_visual_eval_y': encode_array(all_labels_arr),
                'hsja_visual_eval_preds': encode_array(adv_preds_arr),
                'hsja_visual_eval_manifold': encode_array(adv_manifold),
                'hsja_adv_eval_confusion_matrix': encode_array(adv_cm),
            })

        self.logger.info(
            f"HSJA eval done — accuracy={result['hsja_adv_eval/accuracy']:.3f}, "
            f"avg_perturbation={result['hsja_adv_eval/avg_perturbation']:.4f}, "
            f"total_queries={total_queries}")
        self._report_metrics(result)
        self._hsja_eval_running = False

    def _run_hsja_bg(self, include_plots):
        try:
            self._hsja_evaluation(
                n_per_class=int(self.config.get('hsja_n_per_class', 10)),
                n_steps=int(self.config.get('hsja_n_steps', 30)),
                n_grad_samples=int(self.config.get('hsja_n_grad_samples', 30)),
                include_plots=include_plots,
                feature_indices=self.config.get('hsja_feature_indices', None),
                clean_anchors=bool(self.config.get('hsja_clean_anchors', True)),
                init_noise_scale=float(self.config.get('hsja_init_noise_scale', 3.0)),
            )
        except Exception as e:
            self.logger.error(f"HSJA evaluation raised: {e}", exc_info=True)
            self._hsja_eval_running = False

    # -- training loop (mirror consume.py) ---------------------------------
    def _train_model(self):
        while not self._stop:
            diag_feats, diag_labels = self.diagnostics_buffer.sample(self.batch_size)
            anom_feats, anom_labels = self.anomalies_buffer.sample(self.batch_size)
            atk_feats, atk_labels = self.attacks_buffer.sample(self.batch_size)

            if self.adversarial_training:
                adv_anom_feats, adv_anom_labels = self.eval_anomalies_buffer.sample(self.batch_size)
                adv_atk_feats, adv_atk_labels = self.eval_attacks_buffer.sample(self.batch_size)

            if (len(diag_feats) >= self.batch_size and len(anom_feats) >= self.batch_size
                    and len(atk_feats) >= self.batch_size):

                if (self.adversarial_training and len(adv_anom_feats) >= self.batch_size
                        and len(adv_atk_feats) >= self.batch_size):
                    batch_feats = torch.vstack((diag_feats, anom_feats, atk_feats,
                                                adv_anom_feats, adv_atk_feats))
                    batch_labels = torch.vstack((diag_labels, anom_labels, atk_labels,
                                                 adv_anom_labels, adv_atk_labels))
                else:
                    batch_feats = torch.vstack((diag_feats, anom_feats, atk_feats))
                    batch_labels = torch.vstack((diag_labels, anom_labels, atk_labels))

                self.batch_counter += 1
                batch_logits, batch_loss = self.brain.train_step(batch_feats, batch_labels)
                batch_preds = batch_logits.argmax(dim=1)

                self.epoch_loss += batch_loss
                self.epoch_accuracy += accuracy_score(batch_labels, batch_preds)
                self.epoch_precision += precision_score(batch_labels, batch_preds, zero_division=0, average='weighted')
                self.epoch_recall += recall_score(batch_labels, batch_preds, zero_division=0, average='weighted')
                self.epoch_f1 += f1_score(batch_labels, batch_preds, zero_division=0, average='weighted')
                self.epoch_macro_f1 += f1_score(batch_labels, batch_preds, zero_division=0, average='macro')

                if self.batch_counter % self.epoch_size == 0:
                    self._end_epoch()

            time.sleep(self.training_freq_seconds)

    def _end_epoch(self):
        self.epoch_counter += 1
        es = self.epoch_size
        metrics_dict = {
            'total_loss': self.epoch_loss / es,
            'class_accuracy': self.epoch_accuracy / es,
            'class_precision': self.epoch_precision / es,
            'class_recall': self.epoch_recall / es,
            'class_f1': self.epoch_f1 / es,
            'class_macro_f1': self.epoch_macro_f1 / es,
            'diagnostics_processed': self.diagnostics_processed,
            'anoms_processed': self.anoms_processed,
            'attacks_processed': self.attacks_processed,
            'records_processed': self.records_processed,
            'eval_anoms_processed': self.eval_anomalies_processed,
            'eval_attacks_processed': self.eval_attacks_processed,
        }

        if len(self.online_batch_labels) > 20:
            with self.lists_lock:
                ol, op = self.online_batch_labels, self.online_main_batch_preds
                metrics_dict.update({
                    'online_class_accuracy': accuracy_score(ol, op),
                    'online_class_precision': precision_score(ol, op, zero_division=0, average='weighted'),
                    'online_class_recall': recall_score(ol, op, zero_division=0, average='weighted'),
                    'online_class_f1': f1_score(ol, op, zero_division=0, average='weighted'),
                    'online_class_macro_f1': f1_score(ol, op, zero_division=0, average='macro'),
                    'online_confusion_matrix': encode_array(confusion_matrix(ol, op, labels=[0, 1, 2])),
                    'mitigation_time': np.array(self.mitigation_times).mean() if self.mitigation_times else 0.0,
                    'mitigation_reward': self.mitigation_reward,
                })
                self.online_batch_labels = []
                self.online_main_batch_preds = []
                self.mitigation_times = []
                self.mitigation_reward = 0

        self._report_metrics(metrics_dict)
        self._reset_epoch_accumulators()

        if self.epoch_counter % self.save_model_freq_epochs == 0:
            self.logger.info(f"(save_model skipped-to-disk parity) epoch {self.epoch_counter}")

        if self.epoch_counter % self.run_benchmarks_freq_epochs == 0:
            self._run_benchmarks()

    def _run_benchmarks(self):
        self.benchmark_eval_counter += 1
        include_plots = (self.benchmark_eval_counter % self.plot_creation_freq_benchmarks == 0)

        visual = self._visual_evaluation(include_plots=include_plots)
        if visual is not None:
            self._report_metrics(visual)

        if self.eval_sigmas:
            sigma_eval = self._sigma_grid_evaluation(self.eval_sigmas)
            if sigma_eval is not None:
                self._report_metrics(sigma_eval)

        if self.hsja_enabled and not self._hsja_eval_running:
            self._hsja_eval_running = True
            threading.Thread(target=self._run_hsja_bg, kwargs={'include_plots': include_plots},
                             daemon=True, name=f"{self.vehicle_name}_hsja").start()
            self.logger.info(
                f"HSJA evaluation thread started (benchmark round {self.benchmark_eval_counter}, "
                f"epoch {self.epoch_counter}).")

    # -- status ------------------------------------------------------------
    def status(self):
        return {
            'vehicle': self.vehicle_name,
            'epoch': self.epoch_counter,
            'records_processed': self.records_processed,
            'attacks_processed': self.attacks_processed,
            'anoms_processed': self.anoms_processed,
            'diagnostics_processed': self.diagnostics_processed,
        }
