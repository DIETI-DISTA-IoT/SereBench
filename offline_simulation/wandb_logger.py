"""WandbNode — the in-process equivalent of the Wandber observability node.

A faithful port of Wandber's ``manager_server.py`` (the ``Wandber`` class) plus
``kafka_consumer.py``'s statistics routing: it opens a single W&B run and
consumes every ``{vehicle}_statistics`` message off the bus, forwarding scalars
to ``wandb.log`` and rebuilding the interactive Plotly figures (2-D PCA / 2-D
manifold scatter and confusion-matrix heatmaps) for the Gaussian adversarial
eval, the HopSkipJump eval and the online-monitoring confusion matrix — exactly
as the Dockerised Wandber does. The only substitutions are Kafka -> bus and an
optional ``mode`` override so the offline run can log offline/disabled without
a W&B account.
"""

import logging
import threading

import numpy as np
import wandb
import plotly.express as px

from .simulator import EventType

CLASS_NAMES = ['NORMAL', 'ANOMALY', 'ATTACK']


def decode_array(obj):
    return np.frombuffer(bytes.fromhex(obj["data"]), dtype=obj["dtype"]).reshape(obj["shape"])


def plot_confusion_matrix(cm, title):
    fig = px.imshow(
        cm.astype(float), x=CLASS_NAMES, y=CLASS_NAMES,
        labels=dict(x='Predicted', y='True', color='Count'),
        color_continuous_scale='Blues', text_auto='.0f', title=title)
    fig.update_xaxes(side='bottom')
    return fig


def _scatter_2d(x, y, color_labels, title):
    return px.scatter(
        {'x': x[:, 0], 'y': x[:, 1], 'label': color_labels},
        x='x', y='y', color='label', opacity=0.4, title=title)


def plot_results(Y, all_preds, pca_embed, manifold, task_name):
    y_sq = Y.squeeze()
    label_names = [EventType(int(v)).name for v in y_sq]
    pred_names = [EventType(int(v)).name for v in all_preds]
    fig_pca = _scatter_2d(pca_embed, y_sq, label_names, f'Input-Space (2D-PCA) {task_name}')
    fig_labels = _scatter_2d(manifold, y_sq, label_names, f'2D-Representation-Space (labels) {task_name}')
    fig_preds = _scatter_2d(manifold, y_sq, pred_names, f'Predictions {task_name}')
    return fig_pca, fig_labels, fig_preds


class WandbNode:
    def __init__(self, args, bus, mode_override=None):
        self.args = args
        self.bus = bus
        self.logger = logging.getLogger("WANDBER")

        wandb_cfg = args['wandb']
        # online:true -> "online"; online:false -> "disabled" (as Wandber does).
        self.wandb_mode = mode_override or ("online" if wandb_cfg['online'] else "disabled")

        wandb.init(
            project=wandb_cfg['project_name'],
            mode=self.wandb_mode,
            name=wandb_cfg['run_name'],
            group=wandb_cfg['group'],
            config=args,
        )
        self.logger.info(f"Wandb initialized in {self.wandb_mode} mode")
        self.step = 0
        self._stop = False
        self._thread = None
        self._consumer = None

    def push_to_wandb(self, key, value, step=None, commit=True):
        wandb.log({key: value}, step=(step if step is not None else self.step), commit=commit)
        if step is None:
            self.step += 1

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._stop = False
        self._consumer = self.bus.consumer(group_id="WANDBER")
        # Wandber subscribes to several patterns but only *_statistics is
        # actually handled downstream; we mirror that.
        self._consumer.subscribe(["^.*_statistics$"])
        self._thread = threading.Thread(target=self._consume, daemon=True, name="wandber")
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5)
        if self._consumer is not None:
            self._consumer.close()
        try:
            wandb.finish()
        except Exception as e:
            self.logger.error(f"wandb.finish() raised: {e}")
        self.logger.info("Wandber stopped.")

    def _consume(self):
        while not self._stop:
            msg = self._consumer.poll(1.0)
            if msg is None:
                continue
            topic, value = msg
            try:
                if 'statistics' in topic:
                    self._handle_statistics_message(topic, value)
            except Exception as e:
                self.logger.error(f"Error handling stats message: {e}")

    def _handle_statistics_message(self, topic, data):
        vehicle_name = topic.split('_')[0]

        if 'adv_eval_accuracy' in data:
            if 'visual_eval_X' in data:
                fig_pca, fig_labels, fig_preds = plot_results(
                    decode_array(data['visual_eval_y']),
                    decode_array(data['visual_eval_preds']),
                    decode_array(data['visual_eval_X']),
                    decode_array(data['visual_eval_manifold']),
                    vehicle_name + ' manifold')
                self.push_to_wandb(key=f"{vehicle_name}_manifold_pca", value=fig_pca)
                self.push_to_wandb(key=f"{vehicle_name}_manifold_labels", value=fig_labels)
                self.push_to_wandb(key=f"{vehicle_name}_manifold_preds", value=fig_preds)
                self.push_to_wandb(
                    key=f"{vehicle_name}_adv_eval_confusion_matrix",
                    value=plot_confusion_matrix(
                        decode_array(data['adv_eval_confusion_matrix']).astype(int),
                        f"{vehicle_name} Gaussian adv eval"))
            self.push_to_wandb(key=topic, value={
                'adv_eval_accuracy': data['adv_eval_accuracy'],
                'adv_eval_precision': data['adv_eval_precision'],
                'adv_eval_recall': data['adv_eval_recall'],
                'adv_eval_f1': data['adv_eval_f1'],
                'adv_eval_macro_f1': data['adv_eval_macro_f1'],
            })

        elif any(k.startswith('hsja_adv_eval/') for k in data):
            if 'hsja_visual_eval_X' in data:
                fig_pca, fig_labels, fig_preds = plot_results(
                    decode_array(data['hsja_visual_eval_y']),
                    decode_array(data['hsja_visual_eval_preds']),
                    decode_array(data['hsja_visual_eval_X']),
                    decode_array(data['hsja_visual_eval_manifold']),
                    vehicle_name + ' HSJA')
                self.push_to_wandb(key=f"{vehicle_name}_hsja_manifold_pca", value=fig_pca)
                self.push_to_wandb(key=f"{vehicle_name}_hsja_manifold_labels", value=fig_labels)
                self.push_to_wandb(key=f"{vehicle_name}_hsja_manifold_preds", value=fig_preds)
                if 'hsja_adv_eval_confusion_matrix' in data:
                    self.push_to_wandb(
                        key=f"{vehicle_name}_hsja_adv_eval_confusion_matrix",
                        value=plot_confusion_matrix(
                            decode_array(data['hsja_adv_eval_confusion_matrix']).astype(int),
                            f"{vehicle_name} HSJA adv eval"))
            hsja_scalars = {k: v for k, v in data.items() if k.startswith('hsja_adv_eval/')}
            if hsja_scalars:
                self.push_to_wandb(key=topic, value=hsja_scalars)

        else:
            if 'online_confusion_matrix' in data:
                self.push_to_wandb(
                    key=f"{vehicle_name}_online_confusion_matrix",
                    value=plot_confusion_matrix(
                        decode_array(data.pop('online_confusion_matrix')).astype(int),
                        f"{vehicle_name} online monitoring"))
            self.push_to_wandb(key=topic, value=data)
