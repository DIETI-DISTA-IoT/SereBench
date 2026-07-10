"""Optional local dashboard — the SereBench control panel, minus Docker.

Serves a small web UI whose buttons map one-to-one onto the real dashboard's
controls (create vehicles, produce/consume all, start/stop federated learning,
start/stop W&B, automatic + preconfigured attacks, one-click experiment,
shutdown). The crucial difference from ``of-dashboard``: **no button ever
touches a container runtime**. Each action simply calls an
:class:`~offline_simulation.orchestrator.Orchestrator` method, which starts or
stops Python threads inside this very process — push a button, a thread spins
up; push another, it's gone.

Every action runs in a background thread so the request returns immediately, and
a status panel polls :meth:`Orchestrator.status` so you can watch each vehicle
flip HEALTHY/INFECTED, the epoch counters climb and the FL rounds tick over.
"""

import os
import threading

from flask import Flask, jsonify, render_template

from .orchestrator import Orchestrator


def run_dashboard(cfg, wandb_mode=None, host="127.0.0.1", port=8000):
    orch = Orchestrator(cfg, wandb_mode=wandb_mode)
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

    def _bg(fn):
        threading.Thread(target=fn, daemon=True).start()

    # Map each route to an orchestrator action. Slow ones (experiment/shutdown/
    # stops that join threads) run in the background so the UI stays responsive.
    actions = {
        "create-vehicles": orch.create_vehicles,
        "produce-all": orch.produce_all,
        "stop-producing-all": orch.stop_producing_all,
        "consume-all": orch.consume_all,
        "stop-consuming-all": orch.stop_consuming_all,
        "start-federated-learning": orch.start_federated_learning,
        "stop-federated-learning": orch.stop_federated_learning,
        "start-wandb": orch.start_wandb,
        "stop-wandb": orch.stop_wandb,
        "start-automatic-attacks": orch.start_automatic_attacks,
        "stop-automatic-attacks": orch.stop_automatic_attacks,
        "start-preconf-attack": orch.start_preconf_attack,
        "stop-preconf-attack": orch.stop_preconf_attack,
        "start-experiment": orch.start_experiment,
        "shutdown": orch.shutdown,
    }

    @app.route("/")
    def home():
        return render_template("index.html", vehicles=orch.vehicle_names)

    @app.route("/action/<name>", methods=["POST"])
    def action(name):
        fn = actions.get(name)
        if fn is None:
            return jsonify({"ok": False, "error": f"unknown action {name}"}), 404
        _bg(fn)
        return jsonify({"ok": True, "action": name})

    @app.route("/attack/<vehicle>", methods=["POST"])
    def attack(vehicle):
        if vehicle not in orch.vehicle_status_dict:
            return jsonify({"ok": False, "error": "unknown vehicle"}), 404
        orch.start_attack_from_vehicle(vehicle, origin="MANUAL")
        return jsonify({"ok": True, "vehicle": vehicle, "status": "INFECTED"})

    @app.route("/heal/<vehicle>", methods=["POST"])
    def heal(vehicle):
        if vehicle not in orch.vehicle_status_dict:
            return jsonify({"ok": False, "error": "unknown vehicle"}), 404
        orch.stop_attack_from_vehicle(vehicle, origin="MANUAL")
        return jsonify({"ok": True, "vehicle": vehicle, "status": "HEALTHY"})

    @app.route("/status")
    def status():
        return jsonify(orch.status())

    # threaded=True so status polls don't block while an action thread runs.
    app.run(host=host, port=port, threaded=True)
