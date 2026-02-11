# Solms-Friston Swarm: Temperature + Energy + Social Food Trails
# Versiune Optimizată pentru Windows 10 - Memory Leak Fixes Aplicate
# Data: 2026-02-11
# Îmbunătățiri: Figure caching, AsyncIO tracking, Dead agent cleanup, NumPy optimization

# ==========================================
# WINDOWS 10 OPTIMIZATIONS
# ==========================================
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend pentru Windows

import gc
gc.set_threshold(700, 10, 10)  # GC mai agresiv

# Limitare thread-uri NumPy (optional, decomentează dacă ai probleme)
# import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

# ==========================================
# IMPORTS
# ==========================================
import numpy as np
import os
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.agent import AgentSet
import random
import matplotlib.pyplot as plt
import solara
import asyncio

# ==========================================
# ### CONFIGURATION ###
# ==========================================

# --- Simulation Dimensions ---
GRID_WIDTH = 40
GRID_HEIGHT = 40
NUM_AGENTS = 10
SEED = None

# --- Agent Physiology (Life & Death) ---
METABOLISM = 0.15          # Energie consumată per pas
MAX_ENERGY = 100.0         # Capacitate stomac
CRITICAL_ENERGY = 50.0     # Pragul de panică (Hungry)
FOOD_INTAKE = 10.0         # Cât mănâncă o dată
IDEAL_TEMP = 25.0          # Temperatura preferată
INIT_ENERGY_MIN = 40.0     # Energie la naștere (min)
INIT_ENERGY_MAX = 95.0     # Energie la naștere (max)

# --- Social Dynamics & Trails ---
SCENT_DECAY = 0.95         # Cât de repede dispare mirosul de mâncare din mediu (0-1)
MEMORY_DECAY = 0.90        # Cât de repede uită agentul unde a fost (0-1)
FOOD_SIGNAL_DURATION = 15.0 # Câți pași emite miros după ce mănâncă
SOCIAL_WEIGHT = 3.0        # Cât de puternic e atras de mirosul altora (vs explorare)

# --- FEP Brain Parameters (Decision Making) ---
WEIGHT_TEMP = 1.0          # Importanța confortului termic
WEIGHT_ENERGY = 4.0        # Importanța hranei (prioritate mare)
BETA_BASE = 6.0            # Precizia de bază (determinism)
EXPLORATION_FACTOR = 10.0  # Rezistența la plictiseală (valoare mare = evită repetarea)

# --- Environment Generation ---
NUM_FOOD_PATCHES = 1
FOOD_PATCH_AMOUNT_MIN = 40
FOOD_PATCH_AMOUNT_MAX = 80

# --- Visualization Colors ---
COLOR_OK = 'white'
COLOR_HUNGRY = 'saddlebrown'
COLOR_COLD = 'blue'
COLOR_HOT = 'red'
COLOR_DEAD = 'gray'
COLOR_FOOD = 'lime'
COLOR_TRAIL = 'orange'

# ==========================================
# 1. Environment Fields
# ==========================================

def generate_temperature_field(width, height):
    field = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            # Zone calde (Global Plateau)
            field[x, y] += 28 * np.exp(-((x-20)**2 + (y-20)**2) / 300)
            # Optime locale (Hot spots)
            field[x, y] += 14 * np.exp(-((x-8)**2 + (y-32)**2) / 70)
            field[x, y] += 12 * np.exp(-((x-30)**2 + (y-10)**2) / 60)
    return field

def generate_food_field(width, height, n_patches):
    field = np.zeros((width, height))
    for _ in range(n_patches):
        cx, cy = np.random.randint(5, width-5), np.random.randint(5, height-5)
        amp = np.random.uniform(FOOD_PATCH_AMOUNT_MIN, FOOD_PATCH_AMOUNT_MAX) 
        sigma = np.random.uniform(2.0, 4.0) 
        
        for x in range(width):
            for y in range(height):
                dist = (x-cx)**2 + (y-cy)**2
                if dist < 30: 
                    field[x, y] += amp * np.exp(-dist / (2*sigma**2))
    return field

# ==========================================
# 2. Allostatic Agent (OPTIMIZED)
# ==========================================

class AllostaticAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

        self.is_alive = True

        # Physiology
        self.T_int = 10.0 # Porneste rece
        self.T_pref = IDEAL_TEMP
        
        self.E_max = MAX_ENERGY
        self.E_int = np.random.uniform(INIT_ENERGY_MIN, INIT_ENERGY_MAX)
        self.E_crit = CRITICAL_ENERGY

        # FEP Internals
        self.prev_total_error = None
        self.valence_integrated = 0.0
        self.current_beta = BETA_BASE
        
        # Memory - OPTIMIZED: cu batch cleanup
        self.visits = {} 
        self.visit_cleanup_counter = 0  # ✅ FIX: Batch cleanup pentru a reduce rehashing
        
        # Social Signaling
        self.food_signal_timer = 0.0 

    def update_internal_state(self):
        if not self.is_alive: return

        x, y = self.pos
        
        # 1. Thermal Regulation (Physics)
        T_env = self.model.temperature[x, y]
        self.T_int += self.model.eta * (T_env - self.T_int)
        
        # 2. Metabolism
        self.E_int -= METABOLISM
        
        # 3. Eating
        food_available = self.model.food[x, y]
        if food_available > 0.1 and self.E_int < self.E_max:
            space_in_stomach = self.E_max - self.E_int
            intake = min(FOOD_INTAKE, food_available, space_in_stomach)
            
            self.E_int += intake
            self.model.food[x, y] -= intake 
            
            # Broadcast food signal
            if intake > 1.0:
                self.food_signal_timer = FOOD_SIGNAL_DURATION
        
        if self.food_signal_timer > 0:
            self.food_signal_timer -= 1.0

        # 4. Check Death
        if self.E_int <= 0:
            self.E_int = 0
            self.is_alive = False
            self.current_beta = 0 
            return 

        # 5. Calculate Valence (Active Inference)
        err_T = abs(self.T_int - self.T_pref)
        err_E = max(0, self.E_crit - self.E_int)
        
        total_error = (WEIGHT_TEMP * err_T) + (WEIGHT_ENERGY * err_E)
        
        if self.prev_total_error is None:
            self.prev_total_error = total_error
            
        inst_valence = -(total_error - self.prev_total_error)
        self.prev_total_error = total_error
        
        # Integrate Mood
        self.valence_integrated += self.model.mu_affect * (inst_valence - self.valence_integrated)
        
        # Modulate Precision
        factor = np.exp(self.model.sigma * self.valence_integrated)
        self.current_beta = np.clip(BETA_BASE * factor, 0.5, 30.0)

    def manage_memory_and_scent(self):
        """✅ FIX: Optimizat pentru a reduce memory fragmentation pe Windows"""
        if not self.is_alive: return
        pos = self.pos
        
        # A. Personal Memory - Update current position
        self.visits[pos] = self.visits.get(pos, 0.0) + 1.0
        
        # Decay all values
        for loc in self.visits:
            self.visits[loc] *= MEMORY_DECAY
        
        # ✅ FIX: Periodic batch cleanup (nu la fiecare step)
        # Reduce rehashing pe Windows
        self.visit_cleanup_counter += 1
        if self.visit_cleanup_counter >= 50:  # Cleanup la fiecare 50 de pași
            # Recreate dict fără keys cu valori mici
            self.visits = {k: v for k, v in self.visits.items() if v >= 0.05}
            self.visit_cleanup_counter = 0

        # B. Social Scent
        if self.food_signal_timer > 0:
            signal_strength = (self.food_signal_timer / FOOD_SIGNAL_DURATION) * 2.0 
            self.model.food_scent[pos[0], pos[1]] += signal_strength

    def choose_action(self):
        if not self.is_alive: return self.pos

        x, y = self.pos
        candidates = self.model.directions + [(0,0)]
        moves = []
        scores = [] 

        is_hungry = (self.E_int < self.E_crit)

        for dx, dy in candidates:
            nx, ny = x + dx, y + dy
            if self.model.grid.out_of_bounds((nx, ny)):
                continue

            # --- A. Pragmatic Value ---
            T_env_next = self.model.temperature[nx, ny]
            T_pred = self.T_int + self.model.eta * (T_env_next - self.T_int)
            err_T_pred = abs(T_pred - self.T_pref)
            
            food_there = self.model.food[nx, ny]
            intake_pred = 0
            if food_there > 0.1 and (self.E_int - METABOLISM) < self.E_max:
                intake_pred = min(FOOD_INTAKE, food_there)
            E_pred = self.E_int - METABOLISM + intake_pred
            err_E_pred = max(0, self.E_crit - E_pred)
            
            G_pragmatic = - (WEIGHT_TEMP * err_T_pred + WEIGHT_ENERGY * err_E_pred)
            
            # --- B. Epistemic Value ---
            my_trace = self.visits.get((nx, ny), 0.0)
            G_epistemic = 1.0 / (1.0 + EXPLORATION_FACTOR * my_trace)
            
            # --- C. Social Value ---
            G_social = 0.0
            if is_hungry:
                scent_val = self.model.food_scent[nx, ny]
                G_social = SOCIAL_WEIGHT * scent_val 

            # Total G
            G = G_pragmatic + (1.5 * G_epistemic) + G_social
            
            moves.append((nx, ny))
            scores.append(G)

        # Softmax
        scores = np.array(scores)
        scores_exp = np.exp(self.current_beta * (scores - np.max(scores)))
        probs = scores_exp / np.sum(scores_exp)
        
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]

    def step(self):
        if not self.is_alive:
            return
        self.update_internal_state()
        if self.is_alive:
            new_pos = self.choose_action()
            if new_pos != self.pos:
                self.model.grid.move_agent(self, new_pos)
            self.manage_memory_and_scent()

# ==========================================
# 3. Model (OPTIMIZED)
# ==========================================

class DualDriveModel(Model):
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, num_agents=NUM_AGENTS, seed=SEED):
        super().__init__(seed=seed)
        self.grid = MultiGrid(width, height, torus=False)
        self.agent_set = AgentSet([], self)
        
        # Fields
        self.temperature = generate_temperature_field(width, height)
        self.food = generate_food_field(width, height, n_patches=NUM_FOOD_PATCHES)
        
        # Global Scent
        self.food_scent = np.zeros((width, height)) 
        
        # Internal params needed by agents
        self.eta = 0.15        
        self.mu_affect = 0.1
        self.sigma = 1.0
        
        self.directions = [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,1),(1,-1),(-1,-1)]

        # ✅ FIX: Statistici pentru agenți morți
        self.dead_count = 0

        # Spawn Agents
        for i in range(num_agents):
            agent = AllostaticAgent(self)
            rx = self.random.randint(0, width-1)
            ry = self.random.randint(0, height-1)
            self.grid.place_agent(agent, (rx, ry))
            self.agent_set.add(agent)

    def step(self):
        """✅ FIX: Optimizat pentru cleanup agenți morți și NumPy operations"""
        # 1. Agents step
        agents = list(self.agent_set)
        self.random.shuffle(agents)  # ✅ FIX: Folosim self.random (nu mai e nevoie de random_gen)
        
        dead_agents = []
        for agent in agents:
            # Salvează starea înainte
            was_alive = agent.is_alive
            
            # Execute step
            agent.step()
            
            # ✅ FIX: Detectare moarte în acest step
            if was_alive and not agent.is_alive:
                dead_agents.append(agent)
        
        # ✅ FIX: Cleanup agenți morți din grid și agent_set
        for agent in dead_agents:
            self.grid.remove_agent(agent)
            self.agent_set.remove(agent)
            self.dead_count += 1
            
        # 2. Global Environment Decay
        # ✅ FIX: Optimizat pentru a reduce temporare NumPy pe Windows
        np.multiply(self.food_scent, SCENT_DECAY, out=self.food_scent)
        np.putmask(self.food_scent, self.food_scent < 0.05, 0)

# ==========================================
# 4. Visualization (OPTIMIZED)
# ==========================================

def get_plot_figure(model):
    """
    Versiune fără cache pentru debugging
    Dacă asta funcționează, vom optimiza apoi
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    
    # 1. Heatmap (Temperature)
    ax.imshow(model.temperature.T, origin='lower', cmap='coolwarm', alpha=0.4, vmin=0, vmax=40)
    
    # 2. Food patches
    fx, fy, fs = [], [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            val = model.food[x, y]
            if val > 1.0:
                fx.append(x)
                fy.append(y)
                fs.append(min(val * 3, 150)) 
    if fx:
        ax.scatter(fx, fy, c=COLOR_FOOD, s=fs, alpha=0.6, edgecolors='green', label='Food Source')

    # 3. Social Scent Trails
    sx, sy, ss = [], [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            val = model.food_scent[x, y]
            if val > 0.1:
                sx.append(x)
                sy.append(y)
                ss.append(min(val * 20, 50))
    if sx:
        ax.scatter(sx, sy, c=COLOR_TRAIL, s=ss, alpha=0.8, marker='.', label='Food Trail')

    # 4. Agents
    alive_count = len(model.agent_set)  # ✅ FIX: Toți din set sunt vii acum
    for agent in model.agent_set:
        
        # A. Visits (Black dots)
        visits_x, visits_y = [], []
        for (vx, vy), val in agent.visits.items():
            if val > 0.1:
                visits_x.append(vx)
                visits_y.append(vy)
        if visits_x:
            ax.scatter(visits_x, visits_y, c='black', s=5, alpha=0.1)

        # B. Body & Color Logic
        x, y = agent.pos
        z = 10
        marker = 'o'
        
        # --- Logic for Red/Blue/Brown ---
        diff_T = agent.T_int - agent.T_pref 
        err_T_weighted = abs(diff_T) * WEIGHT_TEMP
        err_E_weighted = max(0, agent.E_crit - agent.E_int) * WEIGHT_ENERGY
        
        # Determine dominant drive
        if err_E_weighted > err_T_weighted and err_E_weighted > 1.0:
            c = COLOR_HUNGRY
        elif err_T_weighted > err_E_weighted and err_T_weighted > 1.0:
            # Hot or Cold?
            if diff_T > 0:
                c = COLOR_HOT # Prea cald
            else:
                c = COLOR_COLD # Prea frig
        else:
            c = COLOR_OK
        
        ax.scatter(x, y, c=c, s=120, marker=marker, edgecolors='black', linewidth=1.5, zorder=z)

    # ✅ FIX: Afișează și dead count
    ax.set_title(f'Alive: {alive_count} / Dead: {model.dead_count}')
    ax.set_xlim(-0.5, model.grid.width-0.5)
    ax.set_ylim(-0.5, model.grid.height-0.5)
    ax.axis('off')
    
    # Cleanup
    plt.close(fig)
    
    return fig

# ==========================================
# 5. Solara GUI (OPTIMIZED)
# ==========================================

# ✅ FIX: Helper functions pentru callbacks (evită lambda closures)
def create_model():
    """Factory function pentru model creation"""
    return DualDriveModel()

def increment_tick(t):
    """Increment function pentru tick counter"""
    return t + 1

@solara.component
def Page():
    reset_ctr, set_reset = solara.use_state(0)
    model = solara.use_memo(create_model, dependencies=[reset_ctr])  # ✅ FIX: Nu mai folosim lambda
    
    # Tick counter
    tick, set_tick = solara.use_state(0)
    is_playing, set_playing = solara.use_state(False)
    current_task, set_current_task = solara.use_state(None)  # ✅ FIX: Track task explicit

    # ==========================================
    # CALLBACK FUNCTIONS (definite ÎNAINTE de folosire)
    # ==========================================
    
    def on_step():
        model.step()
        set_tick(tick + 1)

    def on_reset():
        set_playing(False)
        set_reset(reset_ctr + 1)
        set_tick(0)
        
    def on_play():
        set_playing(not is_playing)

    def run_loop():
        """
        ✅ FIX: Versiune optimizată cu task tracking pentru a preveni leaks
        """
        # Cleanup vechiul task dacă există
        if current_task is not None:
            current_task.cancel()
            set_current_task(None)
        
        if not is_playing:
            return
        
        async def loop():
            try:
                while True:
                    model.step()
                    set_tick(increment_tick)  # ✅ FIX: Folosim funcție în loc de lambda
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # Normal cleanup - nu re-raise
                pass
        
        # Creează task nou
        task = asyncio.create_task(loop())
        set_current_task(task)
        
        def cleanup():
            task.cancel()
            set_current_task(None)
        
        return cleanup

    solara.use_effect(run_loop, [is_playing])

    # ==========================================
    # STATS CALCULATION
    # ==========================================
    
    agents = list(model.agent_set)
    alive_agents = [a for a in agents if a.is_alive]
    n_alive = len(alive_agents)
    
    if n_alive > 0:
        avg_E = sum(a.E_int for a in alive_agents) / n_alive
        avg_T = sum(a.T_int for a in alive_agents) / n_alive
        avg_Valence = sum(a.valence_integrated for a in alive_agents) / n_alive
    else:
        avg_E = 0; avg_T = 0; avg_Valence = 0

    # ==========================================
    # UI RENDERING
    # ==========================================
    
    with solara.Sidebar():
        solara.Markdown("## 🧠 Swarm FEP Agents")
        solara.Markdown("Multi-agent simulation with shared food trails.")
        solara.Markdown("**✅ Windows 10 Optimized - Memory Leak Fixed**")
        
        with solara.Row():
            solara.Button("Step", on_click=on_step, color="warning")
            solara.Button("Play/Pause", on_click=on_play, color="success" if is_playing else "primary")
            solara.Button("Reset", on_click=on_reset, color="error")
            
        solara.Markdown("---")
        # Step Counter
        solara.Markdown(f"### Steps: {tick}")
        solara.Markdown(f"**Alive:** {n_alive} / **Dead:** {model.dead_count}")
        
        solara.Markdown("---")
        solara.Markdown(f"**Avg Energy:** {avg_E:.1f}")
        solara.Markdown(f"**Avg Temp:** {avg_T:.1f}")
        solara.Markdown(f"**Avg Mood:** {avg_Valence:.2f}")
        
        solara.Markdown("---")
        solara.Markdown("**Legend:**")
        solara.Markdown(f"⚪ **White:** OK (Optimal)")
        solara.Markdown(f"🟤 **Brown:** Hungry (Searching)")
        solara.Markdown(f"🔵 **Blue:** Cold (Hypothermia)")
        solara.Markdown(f"🔴 **Red:** Hot (Hyperthermia)")
        solara.Markdown(f"🟠 **Orange:** Scent Trail")
        solara.Markdown(f"🟢 **Green:** Food")

        solara.Markdown("---")
        solara.Markdown("### Theoretical Framework")
        
        # 1. Philosophy
        solara.Markdown("**1. Philosophy (Solms/Friston):**")
        solara.Markdown("""
        Agents are self-evidencing systems resisting entropy. 
        **Affect** (Emotion) is the subjective experience of the changing rate of error (Free Energy).
        """)

        # 2. Math - Homeostatic Error
        solara.Markdown("**2. Physiological Error ($H$):**")
        solara.Markdown(r"""
        The drive to correct internal states:
        $$ H = w\_T |T\_{int} - T\_{pref}| + w\_E \max(0, E\_{crit} - E\_{int}) $$
        """)

        # 3. Math - Active Inference
        solara.Markdown("**3. Active Inference ($G$):**")
        solara.Markdown(r"""
        Agents select moves to minimize Expected Free Energy ($G$):
        $$ G(action) = \underbrace{G\_{pragmatic}}\_{\text{Survival}} + \underbrace{G\_{epistemic}}\_{\text{Curiosity}} + \underbrace{G\_{social}}\_{\text{Swarm}} $$
        """)

        # 4. Math - Action Selection
        solara.Markdown("**4. Action Selection:**")
        solara.Markdown(r"""
        Softmax probability modulated by Precision ($\beta$):
        $$ P(a) = \frac{e^{\beta \cdot G(a)}}{\sum e^{\beta \cdot G(a_i)}} $$
        """)

    # Generare figură la fiecare render
    fig = get_plot_figure(model)
    solara.FigureMatplotlib(fig)


# ==================== VIDEO FRAME GENERATION (OPTIONAL) ====================
def generate_video_frames(steps=200, output_dir="frames"):
    """Rulează simularea și salvează fiecare pas ca imagine."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Inițializăm modelul
    model = DualDriveModel()
    print(f"Generare {steps} cadre în folderul '{output_dir}'...")

    for i in range(steps):
        model.step()
        fig = get_plot_figure(model)
        
        # Salvare cadru
        fig.savefig(f"{output_dir}/frame_{i:04d}.png", dpi=120)
        plt.close(fig) # Critic pentru a preveni memory leak-ul
        
        if i % 10 == 0:
            print(f"Cadre procesate: {i}/{steps}")

    print("Gata! Toate cadrele au fost salvate.")


if __name__ == "__main__":
    print(f"Starting simulation with {NUM_AGENTS} agents...")
    print("✅ Windows 10 Optimized Version - Memory Leak Fixes Applied")
    print("Run with: solara run multiagent_food_trails_v3_optimized.py")
    print("Generate video file comand: ffmpeg -framerate 10 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p simulare_swarm.mp4")

    # uncomment the line below to generate video frames (make sure to have enough disk space and time)
    # generate_video_frames(steps=3000)

