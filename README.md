# MINDWORM // Artificial Phenomenology

> "What I cannot create, I do not understand." — **Richard Feynman**
<br/>

## // EXPERIMENT 02: THE ALLOSTATIC AGENTS

This repository contains the simulation kernel for **Experiment 02** from website https://mindworm.icu, an investigation into the emergence of primitive consciousness based on the theories of neuropsychologist **Mark Solms** (*The Hidden Spring*) and **Karl Friston's** *Free Energy Principle*.

---

### The Concept

This is not a game. The agents in this simulation are **self-evidencing systems** resisting entropy. They do not follow arbitrary if/then rules; they are driven by a **biological imperative to maintain homeostasis**.

In this model, what we call "feelings" (hunger, cold, comfort, distress) are modeled as the subjective experience of the agent's internal state deviations. Consciousness, according to Solms, arises from the need to manage these affective states to prioritize attention.

---

### Demo

Here is a visualization of the agents minimizing Free Energy in real-time. Notice how they form trails and change color based on their internal homeostatic error ($H$):

![Swarm Simulation Demo](swarm_demo.gif)

*(White = Optimal, Blue = Cold, Red = Hot, Brown = Hungry)*

---

### Dynamics

The complexity of the swarm arises from simple, local interactions rather than global coordination.

* **Myopic Sensing:** Agents can only sense the 8 adjacent cells (Moore neighborhood). They have no global knowledge of the map or the location of food patches.
* **Two-Channel Trace System:** To navigate this uncertainty, agents interact with two types of decaying markers:
    * **Navigation Trace:** Agents mark their path as they move. This acts as a "repellent" memory, discouraging backtracking and forcing the exploration of new territory.
    * **Food Pheromone:** Upon discovering energy, agents release a specific, high-valence scent. This volatile signal acts as a rudimentary form of stigmergic communication.

---

### 🧠 The Math of "Feeling"

The simulation logic is governed by **Active Inference**. Agents do not merely react to stimuli; they generate predictions about the future to minimize their expected Free Energy ($G$).

#### 1. Physiological Error ($H$)

The agent strives to maintain its internal variables (Temperature $T$ and Energy $E$) within viable bounds:

$$
H = w_T |T_{int} - T_{pref}| + w_E \max(0, E_{crit} - E_{int})
$$

#### 2. Active Inference ($G$)

The agent's decision-making is modeled as a **nested hierarchy** of inference, activated by affective states. This ensures computational efficiency, as higher-level cognition is only engaged when necessary.

1.  **Level 1 (Autonomous Drive):** The default state, focused on immediate survival and exploration.
    $$ G_{base}(a) = \underbrace{G_{pragmatic}(a)}_{\text{Survival}} + \underbrace{G_{epistemic}(a)}_{\text{Curiosity}} $$

2.  **Level 2 (Socio-Cognitive Drive):** This higher level is engaged only when **Arousal** (stress/need) is high. It represents focused, goal-directed thinking (e.g., "I am hungry, I must follow scents and memories of food").
    $$ G_{mod}(a) = \text{Arousal} \cdot \alpha \cdot \left( \underbrace{G_{social}(a)}_{\text{Pheromones}} + \underbrace{G_{memory}(a)}_{\text{Experience}} \right) $$

3.  **Total Expected Free Energy:** The final policy is chosen by minimizing the total G.
    $$ G_{total}(a) = G_{base}(a) + G_{mod}(a) $$

This nested structure ensures that agents don't waste cognitive resources on complex social and memory-based navigation when their basic needs are met. High arousal acts as a switch, bringing these more sophisticated strategies online.

#### 3. Associative Thermal Memory (Linear Receptive Fields)

The agent builds a map of its homeostatic successes. Instead of a computationally expensive Sum-KDE (Gaussian kernels), we use a neuromorphic approximation with **linear receptive fields** (triangular kernels). This is significantly faster.

The expected reward $\hat{V}$ for a given temperature $T$ is the sum of past rewards ($r_k$), weighted by linear proximity.

$$ \hat{V}(T) = \sum_{k} r_k \cdot \max\left(0, 1 - \frac{|T - T_k|}{\sigma_T}\right) $$

Here, $\sigma_T$ acts as the "radius" of the receptive field. This value contributes to the Expected Free Energy, attracting the agent towards familiar thermal contexts.

#### 4. Affect & Precision ($\beta$)

Affect (Valence) is the rate of change of the homeostatic error. This value modulates the agent's **Precision ($\beta$)**, which represents its confidence in its predictions.

$$ \text{valence}_t \approx -(H_t - H_{t-1}) $$

The precision is then updated using a fast, linear approximation instead of an exponential function:

$$ \beta_t \propto \beta_0 \cdot (1 + \sigma \cdot \text{valence}_t) $$

* **Positive Valence:** Error is decreasing, leading to a **High $\beta$** (exploitation, decisive behavior).
* **Negative Valence:** Error is increasing, leading to a **Low $\beta$** (exploration, volatile behavior).

#### 5. Action Selection (Winner-Takes-All)

Instead of a classic Softmax, action selection is modeled as a **neuromorphic Winner-Takes-All (WTA) circuit with noise**. Precision ($\beta$) acts as a noise inhibitor.

$$ \text{Action} = \arg\max_a \left( G(a) + \mathcal{U}\left(-\frac{1}{\beta}, \frac{1}{\beta}\right) \right) $$

* **High $\beta$** (high confidence) leads to low noise, and the agent chooses the action with the best expected outcome ($G(a)$).
* **Low $\beta$** (low confidence) leads to high noise, making the agent's choice more random and exploratory.

---

### 💻 Tech Stack

* **Python 3.11**
* **MESA:** Agent-Based Modeling framework.
* **NumPy:** Vectorized field calculations.
* **Solara:** Reactive web UI for visualization.
* **Matplotlib:** Backend rendering for video generation.

---

### 🚀 Running the Simulation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ioanfesteu/multiagent_FEP.git](https://github.com/ioanfesteu/multiagent_FEP.git)
   cd multiagent_FEP
   ```

2. **Install dependencies:**
   ```bash
   pip install mesa numpy matplotlib plotly solara
   ```

3. **Run the basic dashboard.** Uses the matplotib library, good for starting out.
   ```bash
   solara run multiagent_FEP.py
   ```

4. **Run the interactive dashboard.** Uses Plotly library. You can select individual agents and watch
their parameters change. Select individual agents from dropdown menu or by pausing simulation, selecting agent by clicking on, and then again clicking on the play button.
   ```bash
   solara run multiagent_FEP_i.py
   ```

5. **Make a video of the whole simulation:**
by uncommenting the last line of code in *multiagent_FEP.py*. For every simulation step a snapshot will be saved on disk.
After simulation is done make a video of the whole simulation with ffmpeg.
   ```bash
   python multiagent_FEP.py

   ffmpeg -framerate 10 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p swarm_simulation.mp4
   ```

---

### ⚙️ Tweaking the Simulation
Tweaks can be made in *agents.py*.

At the begining of *agents.py* you will find all the values you can play with and are pretty explanatory I hope. 

Special atention should be payed for *eta*, *mu_affect* and *sigma*. You can find all the explanations in *HOWTO.md*.

---

### 🌡️ Thermal Memory Field

Agents are equipped with an **Associative Thermal Memory**. When they find food, they memorize the environmental temperature of that location. Over time, they build a probabilistic map of "good temperatures".

In the interactive dashboard (`multiagent_FEP_i.py`), you can see this belief system visualized as **Green Isobars** when you select an agent. These contours show where the agent *expects* to find food based on its past experience, guiding its navigation through "thermal surfing".

---

### 📖 Want to learn more?

The main reason I started this project is because I wanted to learn about FEP and active inference. At first it seemed very intimidating to me especially when I was presented with the mathematical framework used by Friston. Then I said that there must be easier ways to understand this paradigm. Reading Mark Solms' book, "The Hidden Spring", helped me enormously to understand concepts that were unfamiliar to me. Therefore, I created some documents to help the curious reader better understand the philosophical and technical foundations of this project. 

Start with INSIGHTS.md in /docs folder. Good luck!

---

### 📜 License
CODE IS LAW. This project is open for research and educational purposes.
