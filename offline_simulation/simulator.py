"""
Vendored verbatim from of-core/train_simulator.py (sereBench branch).

This is an unmodified copy of the SereBench algorithm core, brought into the
monolithic offline_simulation package so the offline run reproduces the exact
same numerical behaviour as the Dockerised platform. Do not edit here; edit the
upstream module and re-vendor if the source changes.
"""

import numpy as np
from enum import Enum

# ============================================================
# UTILS: value bounding
# ============================================================
def bounded(value, min_val, max_val):
    return float(np.clip(value, min_val, max_val))


class EventType(Enum):
    NORMAL = 0
    ANOMALY = 1
    ATTACK = 2


class DrivingMode(Enum):
    STANDSTILL = 1
    TRACTION = 2
    COASTING = 4
    BRAKING = 5


class ERTMSStatus(Enum):
    NORMAL = 0
    DEGRADED = 1
    FAULTY = 2

class LineVoltageType(Enum):
    DC = 2
    AC = 4


# =============================
# GPS — realistic bounded motion
# =============================

# LAT parameters (from dataset)
LAT_MU  = 45.5165
LAT_STD = 0.0934

# LON parameters (from dataset)
LON_MU  = 11.1989
LON_STD = 1.465

# mean-reversion strength
theta = 0.05     # how strongly it returns to center
dt = 1.0

# process noise
normal_noise = 0.0005
anomaly_noise = 0.002
attack_noise = 0.01


# ====================================
# HYDRAULIC CONSTANTS (from dataset)
# ====================================
# Note:
# means here correspond to high-quantiles in data (e.g. 65-75%)... (practical stats)
# deltas to low-quantiles.


# Main reservoir Pressure ( pressure down on the brake actuator)
mean_nominal_Mp = 8.8
Mp_braking_delta = 1.8

Mp_emergency_delta = 2.5
Mp_attack_delta = 4.5

# Brake pipe Pressure ( pressure down on the brake actuator)
mean_nominal_Bp=4.8
Bp_braking_delta = 1.5

Bp_emergency_delta = 3.0
Bp_attack_delta = 4.0

# Brake Cylinder Pressure ( pressure up on the brake actuator)

# BCP main motor cars
mean_nominal_main_cyl=0.0
anom_mean_main_cyl=-0.5
attack_mean_main_cyl=-1.5

main_cyl_std = 1.1
main_cyl_braking_delta = -2.5
main_cyl_Emergency_delta = -3.5

# BCP trailer cars ( pressure up on the brake actuator)
mean_nominal_trailer_cyl=0.0
anom_mean_trailer_cyl=-1.0
attack_mean_trailer_cyl=-2.0

trailer_cyl_std = 0.9
trailer_cyl_braking_delta = -2.2
trailer_cyl_Emergency_delta = -3.2

# ========== EO HYDRAULIC CONSTANTS =========== #

class SimState:
    def __init__(self):

        # Train context
        self.speed = 0.0
        self.speed_limit = 120.0
        self.lat = 45.50
        self.lon = 11.20

        # Cab & mode
        self.active_cab = "M1"
        self.driving_mode = DrivingMode.STANDSTILL

        # Traction & electrical
        self.line_volt_type = LineVoltageType.DC
        self.line_voltage = 3000.0
        self.line_current = 0.

        self.batt_T = {
            "T2": 29.0,
            "T4": 29.0,
            "T5": 29.0,
            "T7": 29.0,
        }

        # Hydraulic subsystem
        self.brake_press_cylinder_main_BC1 = {m: 0.0 for m in ["M1","M3","M6","M8"]}
        self.brake_press_cylinder_main_BC2 = {m: 0.0 for m in ["M1","M3","M6","M8"]}
        self.brake_press_cylinder_trailer_BC1 = {t: 0.0 for t in ["T2","T4","T5","T7"]}
        self.brake_press_cylinder_trailer_BC2 = {t: 0.0 for t in ["T2","T4","T5","T7"]}
        self.bp = 5.0
        self.mp = 9.0

        # ERTMS / HMI
        self.ertms_status = ERTMSStatus.NORMAL
        self.ac_status = {"T2": 0, "T7": 0}
        self.dc_status = {"T2": 1, "T7": 1}
        self.sil_impact = 2000.0


class Train:

    def __init__(self, kwargs):
        self.state = SimState()
        print(f"Creating train with Mp_std={kwargs.Mp_std}, Bp_std={kwargs.Bp_std}")
        self.Mp_std = kwargs.Mp_std
        self.Bp_std = kwargs.Bp_std
        seed = getattr(kwargs, 'seed', None)
        self.rng = np.random.default_rng(seed)

    def _smooth_noise(self, scale=1.0):
        return self.rng.normal(0, scale)

    def generate_train_context(self, event_type):

        # Speed limit evolves slowly
        if self.rng.random() < 0.01:
            self.state.speed_limit = self.rng.choice([30, 60, 90, 120, 140, 160, 180])

        # Speed dynamics depend on driving mode (use enums for keys)
        accel_map = {
            DrivingMode.STANDSTILL: 0.0,   # standstill
            DrivingMode.TRACTION: +1.5,    # traction
            DrivingMode.COASTING: -0.2,    # coasting
            DrivingMode.BRAKING: -2.5,     # braking
        }

        # Default process noise
        base_noise_scale = 0.5

        # get accel according to current driving mode
        accel = accel_map.get(self.state.driving_mode, 0.0)


        # apply acceleration + noise (bounded)
        self.state.speed = bounded(
            self.state.speed + accel + self._smooth_noise(base_noise_scale),
            0, 250
        )

        """
        # If attack: create a clear, separable overspeed pattern (speed >> speed_limit)
        # This makes ATTACKs easy to detect by models that learn relationships between speed and speed_limit.
        if event_type == EventType.ATTACK:
            overshoot = float(np.abs(np.random.normal(30, 5)))  # typical overshoot 20-40 km/h
            self.state.speed = bounded(self.state.speed_limit + overshoot, self.state.speed_limit, 250)
        """

        # choose noise scale
        noise_scale = normal_noise
        """
        if event_type == EventType.ANOMALY:
            noise_scale = anomaly_noise
        elif event_type == EventType.ATTACK:
            noise_scale = attack_noise
        """

        # Ornstein-Uhlenbeck update
        self.state.lat += theta * (LAT_MU - self.state.lat) * dt + self.rng.normal(0, noise_scale)
        self.state.lon += theta * (LON_MU - self.state.lon) * dt + self.rng.normal(0, noise_scale)

        """
        # ATTACK: GPS spoof offsets
        if event_type == EventType.ATTACK and self.rng.random() < 0.3:
            self.state.lat += self.rng.normal(0.05, 0.02)
            self.state.lon += self.rng.normal(0.05, 0.02)
        """

        return {
            "ldvveltreno": self.state.speed,
            "ldvvelimps": self.state.speed_limit,
            "_GPS_LAT": self.state.lat,
            "_GPS_LON": self.state.lon
        }


    def generate_cab_control(self, event_type):

        # Occasional cab switch (rare event, unchanged)
        if self.rng.random() < 0.001:
            self.state.active_cab = "M8" if self.state.active_cab == "M1" else "M1"

        # Occasionally change driving mode:
        if self.rng.random() < 0.2:
            if self.state.speed < 1:
                self.state.driving_mode = self.rng.choice(
                    [DrivingMode.STANDSTILL, DrivingMode.TRACTION],
                    p=[0.7, 0.3]
                )
            elif self.state.speed < 20:
                self.state.driving_mode = self.rng.choice(
                    [DrivingMode.BRAKING, DrivingMode.TRACTION],
                    p=[0.3, 0.7]
                )
            else:
                self.state.driving_mode = self.rng.choice(
                    [DrivingMode.TRACTION, DrivingMode.COASTING, DrivingMode.BRAKING],
                    p=[0.3, 0.6, 0.1]
                )

        return {
            "CabEnabled_M1": 1 if self.state.active_cab == "M1" else 0,
            "CabEnabled_M8": 1 if self.state.active_cab == "M8" else 0,
            "MDS_StatoMarcia": self.state.driving_mode.value
        }


    def generate_traction(self, event_type):

        # Line voltage type changes occasionally
        if self.rng.random() < 0.01:
            if self.state.line_volt_type == LineVoltageType.DC:
                self.state.line_volt_type = LineVoltageType.AC
            else:
                self.state.line_volt_type = LineVoltageType.DC

        # Line current follows speed + mode
        desired_curr = 0
        if self.state.driving_mode == DrivingMode.TRACTION:
            desired_curr = 150 + 2 * self.state.speed
        elif self.state.driving_mode == DrivingMode.BRAKING:
            desired_curr = -50 - self.state.speed
        else:
            desired_curr = 0


        line_current_noise = 10
        DC_voltage_noise = 200
        AC_voltage_noise = 1000

        self.state.line_current = desired_curr + self._smooth_noise(line_current_noise)

        # Line voltage depends on AC/DC
        if self.state.line_volt_type == LineVoltageType.DC:
            self.state.line_voltage = 3000 + self._smooth_noise(DC_voltage_noise)
        else:
            self.state.line_voltage = 25000 + self._smooth_noise(AC_voltage_noise)


        reporting_batt_T = {}
        battery_drift_noise = 0.02
        battery_min = 21
        battery_max = 29.5

        # Battery voltage
        for k in self.state.batt_T.keys():
            self.state.batt_T[k] = bounded(self.state.batt_T[k] + self._smooth_noise(battery_drift_noise), battery_min, battery_max)
            reporting_batt_T[k] = self.state.batt_T[k]

        """
        if event_type == EventType.ANOMALY:
            for k in self.state.batt_T.keys():
                reporting_batt_T[k] = -10
        """

        return {
            "HMI_Vline": self.state.line_voltage,
            "LineVoltType": self.state.line_volt_type.value,
            "HMI_Iline": self.state.line_current,
            **{f"HMI_VBatt_{k}": reporting_batt_T[k] for k in self.state.batt_T.keys()},
            "HMI_Irsts_T2": 0,
            "HMI_Irsts_T7": 0,
        }


    def generate_hydraulics(self, event_type, adversarial=False):

        if self.state.driving_mode != DrivingMode.BRAKING:
            base_bp = mean_nominal_Bp
            base_mp = mean_nominal_Mp
            base_main_cyl = mean_nominal_main_cyl
            base_trailer_cyl = mean_nominal_trailer_cyl

        else:
            base_bp = mean_nominal_Bp - Bp_braking_delta
            base_mp = mean_nominal_Mp - Mp_braking_delta
            base_main_cyl = mean_nominal_main_cyl - main_cyl_braking_delta
            base_trailer_cyl = mean_nominal_trailer_cyl - trailer_cyl_braking_delta


        if event_type == EventType.ANOMALY:
            base_bp -=  Bp_emergency_delta
            base_mp -= Mp_emergency_delta
        elif event_type == EventType.ATTACK:
            base_bp -= Bp_attack_delta
            base_mp -= Mp_attack_delta

        self.state.bp = base_bp + self._smooth_noise(0.01)
        self.state.mp = base_mp + self._smooth_noise(0.01)

        for k in self.state.brake_press_cylinder_main_BC1.keys():
            self.state.brake_press_cylinder_main_BC1[k] = base_main_cyl + self._smooth_noise(main_cyl_std)
            self.state.brake_press_cylinder_main_BC2[k] = base_main_cyl + self._smooth_noise(main_cyl_std)

        for k in self.state.brake_press_cylinder_trailer_BC1.keys():
            self.state.brake_press_cylinder_trailer_BC1[k] = base_trailer_cyl + self._smooth_noise(trailer_cyl_std)
            self.state.brake_press_cylinder_trailer_BC2[k] = base_trailer_cyl + self._smooth_noise(trailer_cyl_std)

        # A change wrt dataset:
        # Suppose now that the main reservoir and the breake pipe's pressure are in
        # bars/100
        self.state.bp *= 100
        self.state.mp *= 100

        # Adversarial perturbation is applied AFTER the *100 rescale, so that
        # Mp_std/Bp_std (and the consumer's sigma-grid eval, which perturbs the
        # same usBpPres/usMpPres features as emitted on the wire) operate on the
        # same scale. Since std scales linearly under a constant multiplier
        # (std(c*X) = c*std(X), unlike variance which scales as c^2), this is
        # the *100-scaled equivalent of adding noise with std=Bp_std/Mp_std
        # before the rescale: (X + N) * 100 == X*100 + N*100.
        if adversarial:
            self.state.bp += self._smooth_noise(100 * self.Bp_std)
            self.state.mp += self._smooth_noise(100 * self.Mp_std)


        return {
            **{f"usB1BCilPres_{k}": self.state.brake_press_cylinder_main_BC1[k] for k in self.state.brake_press_cylinder_main_BC1.keys()},
            **{f"usB2BCilPres_{k}": self.state.brake_press_cylinder_main_BC2[k] for k in self.state.brake_press_cylinder_main_BC2.keys()},
            **{f"usB1BCilPres_{k}": self.state.brake_press_cylinder_trailer_BC1[k] for k in self.state.brake_press_cylinder_trailer_BC1.keys()},
            **{f"usB2BCilPres_{k}": self.state.brake_press_cylinder_trailer_BC2[k] for k in self.state.brake_press_cylinder_trailer_BC2.keys()},
            "usBpPres": self.state.bp,
            "usMpPres": self.state.mp
        }


    def generate_ertms(self, event_type):
        # ============================================================
        # SUBSYSTEM 5 - ERTMS / STATUS / HMI
        # ============================================================

        # baseline state
        if self.rng.random() < 0.001:
            self.state.ertms_status = self.rng.choice(
                [ERTMSStatus.NORMAL, ERTMSStatus.DEGRADED, ERTMSStatus.FAULTY],
                p=[0.85,0.1,0.05])

        # SIL counter increases
        self.state.sil_impact += self._smooth_noise(1)

        """
        # anomaly: ERTMS degraded unexpectedly
        if event_type == EventType.ANOMALY:
            self.state.ertms_status = 2

        # attack: force inconsistent state
        if event_type == EventType.ATTACK:
            self.state.ertms_status = self.rng.choice([0,1], p=[0.2,0.8])
            self.state.line_volt_type = 4 if self.state.line_volt_type == 2 else 2
        """

        return {
            "ERTMS_PiastraSts": self.state.ertms_status.value,
            "HMI_ACPntSts_T2": self.state.ac_status["T2"],
            "HMI_ACPntSts_T7": self.state.ac_status["T7"],
            "HMI_DCPntSts_T2": self.state.dc_status["T2"],
            "HMI_DCPntSts_T7": self.state.dc_status["T7"],
            "HMI_impSIL": self.state.sil_impact
        }


    def step(self, event_type=EventType.NORMAL, adversarial=False):

        state_dict = {}
        train_context = self.generate_train_context(event_type)
        state_dict.update(train_context)
        train_cab_control = self.generate_cab_control(event_type)
        state_dict.update(train_cab_control)
        train_traction = self.generate_traction(event_type)
        state_dict.update(train_traction)
        train_hydraulics = self.generate_hydraulics(event_type, adversarial)
        state_dict.update(train_hydraulics)
        train_ertms = self.generate_ertms(event_type)
        state_dict.update(train_ertms)

        state_dict["event_type"] = event_type.value # label

        return state_dict
