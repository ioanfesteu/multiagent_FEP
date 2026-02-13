# Theoretical Insights: Generative Model & Markov Blanket

This document outlines the theoretical architecture of the agents within the `multiagent_FEP` simulation, analyzing their structure through the lens of the Free Energy Principle (Friston) and Neuropsychoanalysis (Solms).

---

## 1. The Generative Model (The Agent's "Beliefs")

In the context of Active Inference, the "Generative Model" is not a neural network trained on datasets, but a set of **priors (expectations)** about the agent's viable states. The agent acts to fulfill these expectations.

### A. Priors (Fundamental Preferences)
The agent is built on the core belief that it must exist within specific physiological bounds. Any deviation from these bounds is registered as "Surprise" (Free Energy).
* **Thermal Preference ($T_{pref}$):** The agent expects to maintain an optimal body temperature (e.g., 25°C).
* **Metabolic Expectation ($E_{crit}$):** The agent expects to be satiated (Energy > Critical Threshold).

**Physiological Error ($H$):**
The discrepancy between the model and reality is calculated as:
$$H = w_T |T_{int} - T_{pref}| + w_E \max(0, E_{crit} - E_{int})$$

### B. Action Selection Policy (Minimizing Expected Free Energy - $G$)
The agent does not just react; it **predicts** which future action will best align reality with its model. The Expected Free Energy ($G$) for a potential move is composed of:

1.  **$G_{pragmatic}$ (Exploitation / Survival):**
    * *Inference:* "Moving here will reduce my hunger/thermal stress."
    * Directly minimizes the physiological error $H$.
2.  **$G_{epistemic}$ (Exploration / Curiosity):**
    * *Inference:* "Moving here will reduce uncertainty about the environment."
    * Drives the agent to investigate unknown areas or food trails.
3.  **$G_{social}$ (Stigmergy / Swarm Intelligence):**
    * *Inference:* "Following this trail left by others leads to resources."
    * Connects the individual generative model to the collective behavior.

### C. Precision Modulation (Affect as Control)
The model includes a metacognitive layer: **Precision ($\beta$)**.
* **Definition:** The confidence the agent has in its current policy, based on the rate of error change ($-\Delta H$).
* **Dynamics:**
    * **High Precision (Calm/Flow):** Error is decreasing. The agent exploits the best known path deterministically.
    * **Low Precision (Panic/Distress):** Error is increasing. The agent loses confidence in its model and behaves stochastically (exploring randomly) to find a new solution.

---

## 2. The Markov Blanket (The System Boundary)

The Markov Blanket is the mathematical boundary that separates the agent's internal states from the external world, mediating their interaction. This structure validates the agent as an autopoietic (self-organizing) system.

### A. Sensory States ($s$) - *Inputs*
The "shadows" of the world that fall upon the agent's boundary.
1.  **Exteroception (Environmental):**
    * Local Grid Data: Temperature and Food concentration in the Moore Neighborhood (8 neighbors).
    * Social Signals: Pheromone trails intensity and marks leaved by other agents.
2.  **Interoception (Bodily - *Crucial for Solms*):**
    * The "feeling" of the internal error state ($H$). In this architecture, **Affect is a sensory modality** that informs the agent about the state of its own body relative to its priors.

### B. Active States ($a$) - *Outputs*
The mechanisms by which the agent changes the world or its position within it.
1.  **Locomotion:** Changing coordinates $(x, y)$. This is an *epistemic action* (changing what the agent will sense next).
2.  **Consumption:** Reducing `food_amount` in the grid.
3.  **Signaling:** Depositing pheromones (modifying the external memory/stigmergic field).

### C. Internal States ($\mu$) - *Protected Variables*
The hidden variables inside the blanket that the agent tries to keep within homeostatic bounds.
* **Physiological:** Internal Energy ($E_{int}$) and Body Temperature ($T_{int}$).
* **Computational/Affective:** Integrated Valence and Precision ($\beta$).

### D. External States ($\eta$) - *Hidden Reality*
The objective universe the agent cannot touch directly.
* The global `model.food` and `model.temperature` matrices.
* The actual identities and trajectories of other agents.

### Diagram: The Agent's Architecture

```text
       EXTERNAL WORLD (External States - η)
      (Global Grid, Food Sources, Other Agents)
                  |                 ^
                  v                 |
      -----------------------------------------
      | MARKOV BLANKET (The Boundary)         |
      |                                       |
      |  SENSORY STATES (s)   ACTIVE STATES (a)|
      |  (Local Smell, Temp,  (Movement,      |
      |   Interoception H)     Eating, Trails)|
      |          |                 ^          |
      -----------|-----------------|-----------
                 v                 |
       INTERNAL STATES (Internal States - μ)
      (Energy, Temp, Precision β, Valence)
```
---

## 3. Conclusion
The agents in this simulation are not simple input-output machines. They possess a generative model of their own survival and utilize a Markov Blanket to actively maintain their structural integrity.

They act to minimize the divergence between their expectations (priors) and their sensations, effectively "self-evidencing" their own existence through homeostatic regulation and social coordination.