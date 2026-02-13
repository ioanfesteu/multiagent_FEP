# MINDWORM // Artificial Phenomenology

> "What I cannot create, I do not understand." — **Richard Feynman**
<<<<<<< HEAD
=======

>>>>>>> 2984937e86c06da619c70a44dc2bb56974b06459
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

Before moving, an agent simulates all possible adjacent steps and calculates the Expected Free Energy ($G$):

$$
G(action) = \underbrace{G_{pragmatic}}_{\text{Survival}} + \underbrace{G_{epistemic}}_{\text{Curiosity}} + \underbrace{G_{social}}_{\text{Swarm}}
$$

#### 3. Affect & Precision ($\beta$)

Affect (Mood) is the rate of change of the error. This integrated valence determines the agent's **Precision ($\beta$)**.

$$\beta_t = -\frac{H_t - H_{t-1}}{\Delta t}$$

* **Positive Affect:** Error is decreasing ($H_t < H_{t-1}$), leading to a **High $\beta$** (decisive behavior).
* **Negative Affect:** Error is increasing ($H_t > H_{t-1}$), leading to a **Low $\beta$** (volatile or exploratory behavior).



#### 4. Action Selection (Softmax)

The agent selects its next move stochastically using a Softmax function modulated by Precision ($\beta$):

$$
P(a) = \frac{e^{\beta \cdot G(a)}}{\sum_{i} e^{\beta \cdot G(a_i)}}
$$

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
   pip install mesa numpy matplotlib solara
   ```

3. **Run the interactive dashboard:**
   ```bash
   solara run multiagent_FEP.py
   ```

4. **Make a video of the whole simulation:**
by uncommenting the last line of code. For every simulation step a snapshot will be saved on disk.
After simulation is done make a video of the whole simulation with ffmpeg.
   ```bash
   python multiagent_FEP.py

   ffmpeg -framerate 10 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p swarm_simulation.mp4
   ```

---

### ⚙️ Tweaking the Simulation
Tweaks can be made in *agents.py* and *model.py*.

At the begining of *agents.py* you will find all the values you can play with and are pretty explanatory I hope. 

Special atention should be payed for *eta*, *mu_affect* and *sigma* (lines 62, 63, 64) in *model.py*. You can find all the explanations in *HOWTO.md*.

---

### 📖 Want to learn more?

The main reason I started this project is because I wanted to learn about FEP and active inference. At first it seemed very intimidating to me especially when I was presented with the mathematical framework used by Friston. Then I said that there must be easier ways to understand this paradigm. Reading Mark Solms' book, "The Hidden Spring", helped me enormously to understand concepts that were unfamiliar to me. Therefore, I created some documents to help the curious reader better understand the philosophical and technical foundations of this project. 

Start with INSIGHTS.md in /docs folder. Good luck!

---

### 📜 License
CODE IS LAW. This project is open for research and educational purposes.
