# HOW TO TWEAK THE AGENTS PHYSIOLOGY & EMOTIONS

Based on the code in agents.py and model.py, these three parameters control the physics of the body, the dynamics of emotion, and the impact of mood on behavior.

Here is the breakdown of what each parameter does in the context of the Solms-Friston Active Inference model:

### 1. *self.eta = 0.15* (Thermal Coupling / Permeability)
This represents the physical vulnerability of the agent to the environment. It defines how quickly the outside temperature changes the agent's internal body temperature.

In Physics terms: It is a heat transfer coefficient.
In Code (agents.py):
```python
# The agent's body temp moves 15% of the way towards the environment temp per step
self.T_int += self.model.eta * (T_env - self.T_int)
```
Effect:

* **High Eta:** The agent freezes or overheats instantly. It must react very fast to survive.

* **Low Eta:** The agent is well-insulated. It can travel through dangerous zones for longer without dying.

### 2. *self.mu_affect = 0.1* (Emotional Inertia)
This controls how fast the agent's "mood" (Valence) changes. It acts as a filter for the agent's subjective experience of error.

In Cognitive terms: It is the "memory" of recent feelings. It prevents the agent from being bipolar (switching from happy to sad every single step).
In Code (agents.py):
```python
# Exponential Moving Average (EMA) of valence
# Only 10% of the new feeling is added; 90% is the old mood.
self.valence_integrated += self.model.mu_affect * (inst_valence - self.valence_integrated)
```
Effect:

* **High Mu:** The agent is emotionally volatile. One bad step makes it panic immediately.

* **Low Mu (0.1):** The agent has "grit." It takes a sustained period of bad news (increasing error) to ruin its mood.

### 3. *self.sigma = 1.0* (Precision Sensitivity)
This is the gain factor connecting Mood to Action. It determines how much the agent's confidence ($\beta$) fluctuates based on its mood.

In FEP terms: It modulates the "Precision" of beliefs based on affective state.
In Code (agents.py):
```python
# Calculate a multiplier based on mood
factor = np.exp(self.model.sigma * self.valence_integrated)

# Apply to Beta (Inverse Temperature of Softmax)
self.current_beta = np.clip(BETA_BASE * factor, 0.5, 30.0)
```
Effect:

* **High Sigma:** The agent is dramatic. If it feels slightly good, it becomes hyper-confident (deterministic). If it feels slightly bad, it enters chaotic random walk mode immediately.

* **Low Sigma:** The agent is stoic. Its decision-making style (random vs. precise) doesn't change much even if it is suffering.

### Summary Table
| Parameter | Name | Function | High Value Effect |
|-----------|------|----------|-------------------|
| eta | Thermal Coupling | How fast body temp changes. | Agent dies quickly in cold/hot zones. |
| mu_affect | Mood Rate | How fast mood updates. | Emotionally unstable/volatile agent. |
| sigma | Sensitivity | How much mood affects choices. | Manic-depressive behavior (switches between robot-like precision and total randomness). |
