# Hydraulic Subsystem Simulation (`generate_hydraulics()`)

The `generate_hydraulics()` function simulates the **hydraulic subsystem** of a train, generating pressure values for the braking circuits. It is one of the 5 data generation functions in the simulator (together with `train_context`, `cab_control`, `traction`, and `ertms`).

```python
def generate_hydraulics(self, event_type, adversarial=False):
    # Main logic of the hydraulic generator
```

**Parameters:**

* `event_type`: Enum specifying the event type (NORMAL, ANOMALY, ATTACK)
* `adversarial`: Boolean flag that activates adversarial perturbations (model-specific noise)

---

## Hydraulic System Logic

### Base configuration depending on driving mode

The function implements a **state-based model** where base pressure values depend on the current driving mode:

```python
if self.state.driving_mode != DrivingMode.BRAKING:
    # Normal mode (driving, coasting, standstill)
    base_bp = mean_nominal_Bp      # ~4.8 bar
    base_mp = mean_nominal_Mp      # ~8.8 bar
    base_main_cyl = mean_nominal_main_cyl      # 0.0 bar
    base_trailer_cyl = mean_nominal_trailer_cyl # 0.0 bar
else:
    # Braking mode
    base_bp = mean_nominal_Bp - Bp_braking_delta      # 4.8 - 1.5 = 3.3 bar
    base_mp = mean_nominal_Mp - Mp_braking_delta      # 8.8 - 1.8 = 7.0 bar
    base_main_cyl = mean_nominal_main_cyl - main_cyl_braking_delta      # 0.0 - (-2.5) = 2.5 bar
    base_trailer_cyl = mean_nominal_trailer_cyl - trailer_cyl_braking_delta # 0.0 - (-2.2) = 2.2 bar
```

**Physical interpretation:**

* **Mp (Main reservoir pressure)**: Pressure in the main reservoir; decreases during braking
* **Bp (Brake pipe pressure)**: Pressure in the brake pipe; decreases during braking
* **Brake cylinders**: Pressure increases during braking (actuators are activated)

---

### Event-based perturbations

```python
if event_type == EventType.ANOMALY:
    base_bp -=  Bp_emergency_delta      # Additional reduction (-3.0 bar)
    base_mp -= Mp_emergency_delta       # Additional reduction (-2.5 bar)
elif event_type == EventType.ATTACK:
    base_bp -= Bp_attack_delta          # Additional reduction (-4.0 bar)
    base_mp -= Mp_attack_delta          # Additional reduction (-4.5 bar)
```

**Interpretation:**

* **NORMAL**: Nominal system behavior
* **ANOMALY**: Malfunction (e.g., abnormal pressure loss, often due to emergency braking)
* **ATTACK**: Intentional attack (e.g., sabotage or sensor manipulation)

---

### Noise injection

```python
self.state.bp = base_bp + smooth_noise(0.01)
self.state.mp = base_mp + smooth_noise(0.01)

if adversarial:
    self.state.bp += smooth_noise(self.Bp_std)
    self.state.mp += smooth_noise(self.Mp_std)
```

**Two noise levels:**

1. **Base noise (0.01)**: Realistic sensor noise (always present)
2. **Adversarial noise**: Additional configurable noise to make attack detection harder (used during adversarial training)

**Where do `self.Bp_std` and `self.Mp_std` come from?**
From the configuration file in the `config` folder, where the noise level for each vehicle is defined:

```yaml
# /config/default.yaml (or overrides)...

vehicles:
  - angela:
      Mp_std: 0
      Bp_std: 0
  
  - bob:
      Mp_std: 0.7
      Bp_std: 0.7
  
  - claude:
      Mp_std: 1.4
      Bp_std: 1.4

  - daniel:
      Mp_std: 2.1
      Bp_std: 2.1
```

---

## Output Data Structure

### Brake cylinders per car

There are **4 motor cars (M1, M3, M6, M8)** and **4 trailers (T2, T4, T5, T7)**, each with 2 brake cylinders:

```python
for k in self.state.brake_press_cylinder_main_BC1.keys():
    self.state.brake_press_cylinder_main_BC1[k] = base_main_cyl + smooth_noise(main_cyl_std)
    self.state.brake_press_cylinder_main_BC2[k] = base_main_cyl + smooth_noise(main_cyl_std)

for k in self.state.brake_press_cylinder_trailer_BC1.keys():
    self.state.brake_press_cylinder_trailer_BC1[k] = base_trailer_cyl + smooth_noise(trailer_cyl_std)
    self.state.brake_press_cylinder_trailer_BC2[k] = base_trailer_cyl + smooth_noise(trailer_cyl_std)
```

**Notation:**

* `usB1BCilPres_M1`: Cylinder 1 of motor car M1
* `usB2BCilPres_M1`: Cylinder 2 of motor car M1
* Same logic for trailers (T2, T4, T5, T7)

---

### Final scaling

```python
self.state.bp *= 100
self.state.mp *= 100
```

Values are multiplied by 100 to convert from bar to **hundreds of pascals** (or to match the dataset format).

---

### Return structure

```python
return {
    **{f"usB1BCilPres_{k}": self.state.brake_press_cylinder_main_BC1[k] for k in ...},
    **{f"usB2BCilPres_{k}": self.state.brake_press_cylinder_main_BC2[k] for k in ...},
    **{f"usB1BCilPres_{k}": self.state.brake_press_cylinder_trailer_BC1[k] for k in ...},
    **{f"usB2BCilPres_{k}": self.state.brake_press_cylinder_trailer_BC2[k] for k in ...},
    "usBpPres": self.state.bp,
    "usMpPres": self.state.mp
}
```

---

## Parameter and Delta Table

| Parameter               | NORMAL     | ANOMALY    | ATTACK     |
| ----------------------- | ---------- | ---------- | ---------- |
| **Mp (Main Pressure)**  | 8.8 bar    | -2.5 bar   | -4.5 bar   |
| **Bp (Brake Pressure)** | 4.8 bar    | -3.0 bar   | -4.0 bar   |
| **Main Cyl**            | 0.0 bar    | -3.5 bar   | -3.5 bar   |
| **Trailer Cyl**         | 0.0 bar    | -3.2 bar   | -3.2 bar   |
| **Brake Delta**         | -1.8, -1.5 | +2.5, +3.0 | +4.5, +4.0 |

---

## Data “Corruption” Flow Diagram

*(Diagram preserved as-is, translated labels)*

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMULATION START                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Driving Mode?   │
                    └────────┬────────┘
                             │
               ┌─────────────┼─────────────┐
               │             │             │
        ┌──────▼────────┐ ┌─▼──────────┐ ┌▼───────────────┐
        │   NORMAL      │ │ BRAKING    │ │ COASTING/etc.  │
        │ base_mp=8.8   │ │ base_mp=7.0│ │ base_mp=8.8    │
        │ base_bp=4.8   │ │ base_bp=3.3│ │ base_bp=4.8    │
        └──────┬────────┘ └─┬──────────┘ └┬───────────────┘
               │            │             │
               └────────────┼─────────────┘
                            │
                   ┌────────▼──────────┐
                   │ Event Type?       │
                   └────────┬──────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    ┌───▼────┐         ┌────▼────┐         ┌───▼──────┐
    │ NORMAL │         │ ANOMALY │         │  ATTACK  │
    │ Δmp=0  │         │ Δmp=-2.5│         │ Δmp=-4.5 │
    │ Δbp=0  │         │ Δbp=-3.0│         │ Δbp=-4.0 │
    └───┬────┘         └────┬────┘         └───┬──────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
               ┌────────────▼────────────┐
               │  Add Base Noise         │
               │  noise = N(0, 0.01)     │
               └────────────┬────────────┘
                            │
                   ┌────────▼──────────┐
                   │ adversarial=True? │
                   └────────┬──────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
    ┌───▼────────┐                      ┌──────▼──────┐
    │ Yes: Add   │                      │ No: Keep    │
    │ adv noise  │                      │ nominal     │
    └───┬────────┘                      └──────┬──────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            │
               ┌────────────▼────────────┐
               │ Scaling x100             │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │ Return hydraulic data   │
               └────────────────────────┘
```

---

## Concrete Numerical Example

Assume:

* **Driving mode**: BRAKING
* **Event**: ATTACK
* **Adversarial**: True

**Step-by-step for Mp:**

```
1. Nominal base: base_mp = 8.8 bar
2. Braking delta: base_mp -= 1.8 → 7.0 bar
3. Attack delta:  base_mp -= 4.5 → 2.5 bar
4. Base noise:    +0.003 → 2.503 bar
5. Adv noise:     +0.15  → 2.653 bar
6. Scaling:       2.653 * 100 = 265.3
```

Final result: **usMpPres = 265.3**

---

# Summary: Hydraulic Subsystem (`generate_hydraulics`)

The hydraulic subsystem simulates the train braking circuit, including the main reservoir (Mp), brake pipe (Bp), and brake cylinders for each car (8 cars total: 4 motor + 4 trailer, 2 cylinders per car).

Data generation follows a **layered perturbation model**:

1. **Level 1 – Base state:** Nominal pressures depend on driving mode. Normal mode uses nominal values (Mp=8.8 bar, Bp=4.8 bar). Braking reduces pressures (Mp=7.0 bar, Bp=3.3 bar).
2. **Level 2 – Event perturbations:** Additional deltas are applied. ANOMALY events apply moderate drops, while ATTACK events apply stronger drops to separate natural anomalies from deliberate attacks.
3. **Level 3 – Sensor noise:** Small Gaussian noise (σ=0.01) simulates realistic sensor uncertainty.
4. **Level 4 – Adversarial noise (optional):** When `adversarial=True`, extra configurable noise (Mp_std, Bp_std) is added to harden detection models.

Brake cylinders follow the same logic with specific standard deviations (main_cyl_std=1.1, trailer_cyl_std=0.9). Finally, all values are scaled by 100 to match the dataset format.
