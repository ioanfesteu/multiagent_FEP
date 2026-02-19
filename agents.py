# agents.py
import numpy as np
from mesa import Agent

# ==========================================
# ### CONFIGURATION ###
# ==========================================

# --- Simulation Dimensions ---
GRID_WIDTH = 40
GRID_HEIGHT = 40
NUM_AGENTS = 5
SEED = 3 # Set to an integer for reproducibility

# --- Agent Physiology (Life & Death) ---
METABOLISM = 0.15          # Energy consumed per step
MAX_ENERGY = 100.0         # Stomach capacity :P
CRITICAL_ENERGY = 50.0     # Panic threshold (Hungry)
FOOD_INTAKE = 10.0         # Amount eaten at once
IDEAL_TEMP = 25.0          # Preferred temperature
INIT_ENERGY_MIN = 40.0     # Birth energy (min)
INIT_ENERGY_MAX = 95.0     # Birth energy (max)

# --- Social Dynamics & Trails ---
SCENT_DECAY = 0.98         # How fast food scent disappears from environment (0-1)
MEMORY_DECAY = 0.98        # How fast the agent forgets where it has been (0-1)
FOOD_SIGNAL_DURATION = 25.0 # How many steps it emits scent after eating
SOCIAL_WEIGHT = 3.0        # How strongly it is attracted to others' scent (vs exploration)

# --- FEP Brain Parameters (Decision Making) ---
WEIGHT_TEMP = 1.0          # Importance of thermal comfort
WEIGHT_ENERGY = 4.0        # Importance of food (high priority)
BETA_BASE = 6.0            # Base precision (determinism)
BETA_MAX = 30.0            # Maximum precision (clipping)
WEIGHT_EPISTEMIC = 1.5     # Importance of curiosity (Agency/Exploration). Curiosity vs. Survival (G_pragmatic) vs. Socializing (G_social)
EXPLORATION_FACTOR = 10.0  # Boredom resistance (high value = avoids repetition)

# --- Psycho-behavioral Parameters ---
ETA = 0.1                  # Thermal conductivity / Physical inertia
MU_AFFECT = 0.4            # Affect integration rate / Emotional stability
SIGMA = 0.8                # Precision sensitivity to affect / Psychosomatic coupling

# ==========================================
# ### ASSOCIATIVE THERMAL MEMORY ###
# ==========================================
#
# Agents remember T_int at each feeding event. Because T_int evolves through
# thermodynamic exchange with the environment (ETA), it already encodes
# a physically grounded context — no explicit temporal decay is needed.
# Old memories become irrelevant naturally as T_int drifts away from them.
#
# G_memory(a) = -alpha * V_hat(T_pred)
# V_hat(T)    = mean intake over traces weighted by Gaussian similarity
#
MEMORY_MAX_TRACES = 5     # FIFO capacity (N_max)
MEMORY_SIGMA_T = 6.0     # Gaussian kernel width in °C — wider range to create gradients from afar
MEMORY_ALPHA = 1.0       # Weight of G_memory (reduced because Sum-KDE produces larger values than Mean-KDE)
# Future: MEMORY_GAMMA = 0.5  # For affect-modulated σ_T: sigma = MEMORY_SIGMA_T * exp(-MEMORY_GAMMA * valence_integrated)

# --- Environment Generation ---
NUM_FOOD_PATCHES = 1
FOOD_PATCH_AMOUNT_MIN = 30
FOOD_PATCH_AMOUNT_MAX = 80
TEMP_BASE_MAX = 28.0       # Temperatura maxima a zonei centrale
TEMP_SPOT_1 = 14.0         # Temperatura sursei locale 1
TEMP_SPOT_2 = 12.0         # Temperatura sursei locale 2

# --- Visualization Colors ---
COLOR_OK = 'white'
COLOR_HUNGRY = 'saddlebrown'
COLOR_COLD = 'blue'
COLOR_HOT = 'red'
COLOR_DEAD = 'gray'
COLOR_FOOD = 'lime'
COLOR_TRAIL = 'orange'

# ==========================================
# Allostatic Agent (OPTIMIZED)
# ==========================================

class AllostaticAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

        self.is_alive = True

        # Physiology
        self.T_int = 10.0 # Starts cold
        self.T_pref = IDEAL_TEMP
        
        self.E_max = MAX_ENERGY
        self.E_int = np.random.uniform(INIT_ENERGY_MIN, INIT_ENERGY_MAX)
        self.E_crit = CRITICAL_ENERGY

        # FEP Internals
        self.prev_total_error = None
        self.valence_integrated = 0.0
        self.valence_bound = 2.0  # For dynamic progress bar scaling
        self.current_beta = BETA_BASE
        
        # Memory - OPTIMIZED: with batch cleanup
        self.visits = {}
        self.visit_cleanup_counter = 0  # ✅ FIX: Batch cleanup to reduce rehashing

        # Social Signaling
        self.food_signal_timer = 0.0

        # Associative Thermal Memory: list of (T_env, intake) pairs.
        # We store T_env (environment temp) to avoid thermal inertia confusion.
        self.thermal_memory = []

    def update_internal_state(self):
        if not self.is_alive: return

        x, y = self.pos
        
        # 1. Thermal Regulation (Physics)
        T_env = self.model.temperature[x, y]
        self.T_int += ETA * (T_env - self.T_int)
        
        # 2. Metabolism
        self.E_int -= METABOLISM
        
        # 3. Eating
        food_available = self.model.food[x, y]
        if food_available > 0.1 and self.E_int < self.E_max:
            space_in_stomach = self.E_max - self.E_int
            intake = min(FOOD_INTAKE, food_available, space_in_stomach)
            
            self.E_int += intake
            self.model.food[x, y] -= intake 
            
            # Broadcast food signal and record thermal memory trace
            if intake > 1.0:
                self.food_signal_timer = FOOD_SIGNAL_DURATION
                self._record_feeding(intake=intake, T_context=T_env)
        
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
        self.valence_integrated += MU_AFFECT * (inst_valence - self.valence_integrated)
        
        # Modulate Precision
        factor = np.exp(SIGMA * self.valence_integrated)
        self.current_beta = np.clip(BETA_BASE * factor, 0.5, BETA_MAX)

        # Update valence bound for visualization
        current_abs_valence = abs(self.valence_integrated)
        if current_abs_valence > self.valence_bound:
            self.valence_bound = current_abs_valence

    # ------------------------------------------
    # Associative Thermal Memory
    # ------------------------------------------

    def _memory_value(self, T_query):
        """
        Sum of Gaussian kernels (Density Estimation).
        We switched from Nadaraya-Watson (weighted average) to Sum-KDE.
        
        Why? Weighted average creates a flat plateau if only one memory cluster exists
        (e.g., value is 10.0 everywhere). Sum-KDE creates a peak at the memory 
        location and decays to zero far away, providing a navigation gradient.

        V_hat(T) = sum_k [ r_k * exp(-(T - T_k)^2 / (2 * sigma_T^2)) ]
        """
        if not self.thermal_memory:
            return 0.0
        kernels = np.array([
            np.exp(-((T_query - T_k) ** 2) / (2.0 * MEMORY_SIGMA_T ** 2))
            for T_k, _ in self.thermal_memory
        ])
        intakes = np.array([r_k for _, r_k in self.thermal_memory])
        return float((intakes * kernels).sum())

    def _record_feeding(self, intake, T_context):
        """
        Store (T_context, intake) at the moment of eating.
        T_context should be T_env (real temperature) to avoid inertia bias.
        FIFO eviction when capacity N_max is reached.
        """
        if len(self.thermal_memory) >= MEMORY_MAX_TRACES:
            self.thermal_memory.pop(0)
        self.thermal_memory.append((T_context, intake))

    # ------------------------------------------
    # Core Agent Methods
    # ------------------------------------------

    def manage_memory_and_scent(self):
        """✅ FIX: Optimized to reduce memory fragmentation on Windows"""
        if not self.is_alive: return
        pos = self.pos
        
        # A. Personal Memory - Update current position
        self.visits[pos] = self.visits.get(pos, 0.0) + 1.0
        
        # Shared Memory - Mark global field (Stigmergy)
        self.model.shared_memory[pos[0], pos[1]] += 1.0
        
        # Decay all values
        for loc in self.visits:
            self.visits[loc] *= MEMORY_DECAY
        
        # ✅ FIX: Periodic batch cleanup (not every step)
        # Reduce rehashing on Windows
        self.visit_cleanup_counter += 1
        if self.visit_cleanup_counter >= 50:  # Cleanup every 50 steps
            # Recreate dict without keys with small values
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

            # --- A. Pragmatic Value (SURVIVAL) ---
            T_env_next = self.model.temperature[nx, ny]
            T_pred = self.T_int + ETA * (T_env_next - self.T_int)
            err_T_pred = abs(T_pred - self.T_pref)
            
            food_there = self.model.food[nx, ny]
            intake_pred = 0
            if food_there > 0.1 and (self.E_int - METABOLISM) < self.E_max:
                intake_pred = min(FOOD_INTAKE, food_there)
            E_pred = self.E_int - METABOLISM + intake_pred
            err_E_pred = max(0, self.E_crit - E_pred)
            
            G_pragmatic = - (WEIGHT_TEMP * err_T_pred + WEIGHT_ENERGY * err_E_pred)
            
            # --- B. Epistemic Value (AGENCY) ---
            # Switch to Shared Memory: Agents now avoid/seek where *anyone* has been
            shared_trace = self.model.shared_memory[nx, ny]
            G_epistemic = 1.0 / (1.0 + EXPLORATION_FACTOR * shared_trace)
            
            # --- C. Social Value ---
            G_social = 0.0
            if is_hungry:
                scent_val = self.model.food_scent[nx, ny]
                G_social = SOCIAL_WEIGHT * scent_val 
            # if is_hungry:
            #     scent_val = self.model.food_scent[nx, ny]
            #     # ✅ FIX: Ignoram mirosul de pe pozitia curenta daca nu mai e mancare
            #     # Altfel agentul ramane blocat in propriul miros dupa ce mananca
            #     if dx == 0 and dy == 0 and food_there < 0.1:
            #         G_social = 0.0
            #     else:
            #         G_social = SOCIAL_WEIGHT * scent_val 

            # --- D. Associative Theenvironmental temperature of the target cell.
            # We switched to T_env to avoid thermal inertia confusion.
            # Active only when hungry and memory is non-empty.
            G_memory = 0.0
            if is_hungry and self.thermal_memory:
                G_memory = MEMORY_ALPHA * self._memory_value(T_env_next)

            # Total G
            G = G_pragmatic + (WEIGHT_EPISTEMIC * G_epistemic) + G_social + G_memory
            
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