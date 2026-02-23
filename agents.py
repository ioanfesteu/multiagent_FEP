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
SEED = None # Set to an integer for reproducibility

# --- Environment Generation ---
NUM_FOOD_PATCHES = 1
FOOD_PATCH_AMOUNT_MIN = 30
FOOD_PATCH_AMOUNT_MAX = 60
FOOD_REGROWTH_RATE = 0.01   # Cantitatea de hrana regenerata per step
TEMP_BASE_MAX = 28.0       # Temperatura maxima a zonei centrale
TEMP_SPOT_1 = 14.0         # Temperatura sursei locale 1
TEMP_SPOT_2 = 12.0         # Temperatura sursei locale 2

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
MEMORY_DECAY = 0.90        # How fast the agent forgets where it has been (0-1)
FOOD_SIGNAL_DURATION = 25.0 # How many steps it emits scent after eating

# --- FEP Brain Parameters (Decision Making) ---
# NOTE: These are now defaults. Actual values are read from SimConfig to allow UI tuning.

# --- NIVEL 1 (Autonom) ---
WEIGHT_PRAGMATIC = 1.0     
WEIGHT_EPISTEMIC = 2.0     
WEIGHT_TEMP = 1.0          # Sub-weight for Pragmatic
WEIGHT_ENERGY = 4.0        # Sub-weight for Pragmatic

# --- NIVEL 2 (Socio-Cognitiv) ---
SOCIAL_WEIGHT = 0.5        
MEMORY_WEIGHT = 0.8         

# --- PARAMETRI AFECT & PRECIZIE ---
AROUSAL_SCALING = 0.1      # Multiplicator pentru intervenția Nivelului 2
BASE_PRECISION = 2.0       # (\beta_0) Încrederea de bază
BETA_SENSITIVITY = 1.0     # Cât de mult modifică valența precizia

BETA_BASE = BASE_PRECISION # Alias for backward compatibility
BETA_MAX = 30.0            # Maximum precision (clipping)
EXPLORATION_FACTOR = 20.0  # Boredom resistance (high value = avoids repetition)

# --- Psycho-behavioral Parameters ---
ETA = 0.15                  # Thermal conductivity / Physical inertia
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
# Future: MEMORY_GAMMA = 0.5  # For affect-modulated σ_T: sigma = MEMORY_SIGMA_T * exp(-MEMORY_GAMMA * valence_integrated)

# --- Visualization Colors ---
COLOR_OK = 'white'
COLOR_HUNGRY = 'saddlebrown'
COLOR_COLD = 'blue'
COLOR_HOT = 'red'
COLOR_DEAD = 'gray'
COLOR_FOOD = 'lime'
COLOR_TRAIL = 'orange'

# ==========================================
# ### DYNAMIC CONFIGURATION ###
# ==========================================
class SimConfig:
    """Holds simulation parameters that can be tweaked in real-time from the UI."""
    WEIGHT_PRAGMATIC = WEIGHT_PRAGMATIC # G_pragmatic weight
    WEIGHT_EPISTEMIC = WEIGHT_EPISTEMIC # G_epistemic weight
    SOCIAL_WEIGHT = SOCIAL_WEIGHT      # G_social weight
    MEMORY_WEIGHT = MEMORY_WEIGHT      # G_memory weight
    MEMORY_SIGMA_T = 6.0     # Thermal memory width
    RANDOM_SEED = None       # For reproducibility

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
        self.current_homeostatic_error = 0.0
        self.prev_homeostatic_error = 0.0
        self.valence = 0.0             # Starea afectivă (+/-)
        self.affective_arousal = 0.0   # Intensitatea stării (Alertă/Panic)
        self.valence_bound = 2.0  # For dynamic progress bar scaling
        self.current_beta = BETA_BASE

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

        # 5. Calculate Internal State (Phase 3: Valence & Arousal)
        err_T = abs(self.T_int - self.T_pref)
        err_E = max(0, self.E_crit - self.E_int)
        
        # H_t: Eroarea homeostatică totală
        self.current_homeostatic_error = (WEIGHT_TEMP * err_T) + (WEIGHT_ENERGY * err_E)
        
        # Valence: Derivata negativă a erorii (lucrurile merg bine vs rău)
        # Smoothing: 0.7 * old + 0.3 * new
        delta_H = self.current_homeostatic_error - self.prev_homeostatic_error
        self.valence = (1.0 - MU_AFFECT) * self.valence + MU_AFFECT * (-delta_H)
        
        self.prev_homeostatic_error = self.current_homeostatic_error
        
        # Arousal: Crește când eroarea e mare (stres) sau când valența e negativă (panică)
        # Formula euristică: Eroarea curentă + Bonus de panică dacă valența scade
        self.affective_arousal = self.current_homeostatic_error * (1.0 + max(0, -self.valence * 2.0))
        
        # Modulate Precision (Beta) based on Valence
        # factor = np.exp(SIGMA * self.valence)
        # self.current_beta = np.clip(BETA_BASE * factor, 0.5, BETA_MAX)
        
        # ********************************************************************************
        # Codul nou (fără exponențiale, ultra-rapid):
        # Factorul devine pur și simplu o scalare algebrică. 
        # Când valence = 0, factor = 1.0. Când valence e pozitivă, crește liniar.
        factor = max(0.1, 1.0 + (SIGMA * self.valence)) 
        self.current_beta = max(0.5, min(BETA_MAX, BETA_BASE * factor))
        # ********************************************************************************
        
        # Update valence bound for visualization
        current_abs_valence = abs(self.valence)
        if current_abs_valence > self.valence_bound:
            self.valence_bound = current_abs_valence

    # ------------------------------------------
    # Associative Thermal Memory
    # ------------------------------------------

    # def _memory_value(self, T_query):
    #     """
    #     Sum of Gaussian kernels (Density Estimation).
    #     We switched from Nadaraya-Watson (weighted average) to Sum-KDE.
        
    #     Why? Weighted average creates a flat plateau if only one memory cluster exists
    #     (e.g., value is 10.0 everywhere). Sum-KDE creates a peak at the memory 
    #     location and decays to zero far away, providing a navigation gradient.

    #     V_hat(T) = sum_k [ r_k * exp(-(T - T_k)^2 / (2 * sigma_T^2)) ]
    #     """
    #     if not self.thermal_memory:
    #         return 0.0
    #     kernels = np.array([
    #         np.exp(-((T_query - T_k) ** 2) / (2.0 * SimConfig.MEMORY_SIGMA_T ** 2))
    #         for T_k, _ in self.thermal_memory
    #     ])
    #     intakes = np.array([r_k for _, r_k in self.thermal_memory])
    #     return float((intakes * kernels).sum())

    # *****************************************************************************
    def _memory_value(self, T_query):
        """
        Înlocuire neuromorfică a Gaussienei cu Câmpuri Receptive Liniare.
        Fără exponențiale, fără ridicări la pătrat. Doar distanță pură.
        """
        if not self.thermal_memory:
            return 0.0
            
        total_value = 0.0
        
        # Parametrul existent (MEMORY_SIGMA_T) devine "raza" memoriei.
        # Ex: Dacă e 6.0, agentul consideră relevantă o memorie doar dacă e la +/- 6 grade distanță.
        max_distance = SimConfig.MEMORY_SIGMA_T 
        
        for T_k, intake in self.thermal_memory:
            # 1. Distanța absolută simplă (câte grade sunt între memoria mea și temperatura testată)
            distance = abs(T_query - T_k)
            
            # 2. Dacă temperatura testată este suficient de aproape de memoria mea
            if distance < max_distance:
                # 3. Calculăm activarea liniară (regula de trei simplă)
                # Dacă distance = 0 (potrivire perfectă), activarea = 1.0
                # Dacă distance se apropie de max_distance, activarea scade spre 0.0
                activation = 1.0 - (distance / max_distance)
                
                # Adăugăm valoarea amintirii ponderată de cât de puternic a strigat neuronul
                total_value += intake * activation
                
        return total_value
    # *****************************************************************************

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
        """Manages shared memory and food scent emission."""
        if not self.is_alive: return
        pos = self.pos

        # A. Shared Memory - Mark global field (Stigmergy)
        self.model.shared_memory[pos[0], pos[1]] += 1.0

        # B. Social Scent
        if self.food_signal_timer > 0:
            signal_strength = (self.food_signal_timer / FOOD_SIGNAL_DURATION) * 2.0
            self.model.food_scent[pos[0], pos[1]] += signal_strength

    def choose_action(self):
        if not self.is_alive: return self.pos

        x, y = self.pos
        # candidates = self.model.directions + [(0,0)]
        candidates = self.model.directions # No "stay put" option to encourage movement
        moves = []
        scores = [] 

        for dx, dy in candidates:
            nx, ny = x + dx, y + dy
            if self.model.grid.out_of_bounds((nx, ny)):
                continue

            # =================================================
            # PHASE 4: NESTED ACTIVE INFERENCE
            # =================================================

            # --- NIVEL 1: AUTONOM (Base G) ---
            
            # 1. G_pragmatic (Supravietuire / Homeostazie)
            T_env_next = self.model.temperature[nx, ny]
            T_pred = self.T_int + ETA * (T_env_next - self.T_int)
            err_T_pred = abs(T_pred - self.T_pref)
            
            food_there = self.model.food[nx, ny]
            intake_pred = 0
            if food_there > 0.1 and (self.E_int - METABOLISM) < self.E_max:
                intake_pred = min(FOOD_INTAKE, food_there)
            E_pred = self.E_int - METABOLISM + intake_pred
            err_E_pred = max(0, self.E_crit - E_pred)
            
            # Notă: G e negativ (cost), deci folosim minus
            term_pragmatic = - (WEIGHT_TEMP * err_T_pred + WEIGHT_ENERGY * err_E_pred)
            G_pragmatic = SimConfig.WEIGHT_PRAGMATIC * term_pragmatic
            
            # 2. G_epistemic (Curiozitate / Explorare)
            # Evităm locurile deja vizitate de roi (Shared Memory)
            shared_trace = self.model.shared_memory[nx, ny]
            
            # EXPLORATION_FACTOR controlează panta de "plictiseală" (cât de repede scade interesul la vizite repetate)
            term_epistemic = 1.0 / (1.0 + EXPLORATION_FACTOR * shared_trace)
            
            # WEIGHT_EPISTEMIC controlează magnitudinea maximă a curiozității față de foame/frig
            G_epistemic = SimConfig.WEIGHT_EPISTEMIC * term_epistemic
            
            # Integrare Nivel 1
            G_base = G_pragmatic + G_epistemic

            # --- NIVEL 2: SOCIO-COGNITIV (Top-Down Modulation) ---
            # Se activează doar dacă Arousal-ul este ridicat (stres/nevoie)
            
            G_higher_top_down = 0.0
            
            # Calculăm motivația de hrană (Hunger Drive) progresiv
            # 0.0 = Sătul, 1.0 = Complet gol
            # Folosim pătratul pentru a reduce sensibilitatea când agentul este aproape plin
            saturation = np.clip(self.E_int / self.E_max, 0.0, 1.0)
            hunger_drive = (1.0 - saturation) ** 2

            if self.affective_arousal > 0.1:
                # G_social (Feromoni) - ponderat de foame
                scent_val = self.model.food_scent[nx, ny]
                term_social = SimConfig.SOCIAL_WEIGHT * scent_val * hunger_drive
                
                # G_memory (Memorie Termică Asociativă) - ponderat de foame
                term_memory = 0.0
                if self.thermal_memory:
                    term_memory = SimConfig.MEMORY_WEIGHT * self._memory_value(T_env_next) * hunger_drive
                
                # Modulare prin Arousal: Stresul amplifică semnalele, dar conținutul depinde de foame
                G_higher_top_down = (self.affective_arousal * AROUSAL_SCALING) * (term_social + term_memory)

            # --- INTEGRARE TOTALĂ ---
            G_total = G_base + G_higher_top_down
            
            moves.append((nx, ny))
            scores.append(G_total)

        # Softmax
        # scores = np.array(scores)
        # scores_exp = np.exp(self.current_beta * (scores - np.max(scores)))
        # probs = scores_exp / np.sum(scores_exp)
        # idx = np.argmax(probs)                             # Deterministic choice of the best action
        # # idx = np.random.choice(len(moves), p=probs)      # Stochastic choice (uncomment for more exploration)
        # return moves[idx]

        # ********************************************************************************
        # --- CODUL NOU NEUROMORFIC (Winner-Takes-All) ---
        # 1. Calculăm magnitudinea zgomotului de fond. 
        # Beta acționează ca un inhibitor al zgomotului. 
        noise_amplitude = 1.0 / (self.current_beta + 1e-5) 

        best_move_idx = 0
        highest_membrane_potential = -float('inf')

        # 2. Fiecare opțiune este un "neuron" care se excită
        for i in range(len(moves)):
            # Potențialul membranei = Semnalul util (G_total) + Fluctuații locale (Zgomot)
            # Folosim o simplă adunare și generare de număr aleatoriu
            synaptic_noise = np.random.uniform(-noise_amplitude, noise_amplitude)
            membrane_potential = scores[i] + synaptic_noise
            
            # Inhibiție laterală instantanee: Primul care atinge pragul maxim câștigă
            # și "stinge" complet toți ceilalți neuroni.
            if membrane_potential > highest_membrane_potential:
                highest_membrane_potential = membrane_potential
                best_move_idx = i
                
        return moves[best_move_idx]
        # ********************************************************************************

    def step(self):
        if not self.is_alive:
            return
        self.update_internal_state()
        if self.is_alive:
            new_pos = self.choose_action()
            if new_pos != self.pos:
                self.model.grid.move_agent(self, new_pos)
            self.manage_memory_and_scent()