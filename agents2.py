# agents.py
import numpy as np
from mesa import Agent

# ==========================================
# ### CONFIGURATION ###
# ==========================================

# --- Simulation Dimensions ---
GRID_WIDTH = 80
GRID_HEIGHT = 40
NUM_AGENTS = 10
SEED = 3 

# --- Agent Physiology (Life & Death) ---
METABOLISM = 0.15          
MAX_ENERGY = 100.0         
CRITICAL_ENERGY = 50.0     
FOOD_INTAKE = 10.0         
IDEAL_TEMP = 25.0          
INIT_ENERGY_MIN = 40.0     
INIT_ENERGY_MAX = 95.0     

# --- Social Dynamics & Trails ---
SCENT_DECAY = 0.98         
MEMORY_DECAY = 0.90        
FOOD_SIGNAL_DURATION = 25.0 
SOCIAL_WEIGHT = 3.0        

# --- FEP Brain Parameters (Decision Making) ---
WEIGHT_TEMP = 1.0          
WEIGHT_EPISTEMIC = 5.0     
EXPLORATION_FACTOR = 2.0   
BETA_MIN = 0.5             
BETA_MAX = 5.0             

# --- Associative Learning (NEW) ---
# Definește cum explorarea capătă utilitate prin corelații senzoriale
LEARNING_RATE_MEMORY = 0.2  # Viteza de actualizare a amintirii termice
WEIGHT_ASSOCIATIVE = 3.0    # Importanța memoriei când agentului îi este foame
SIGMA_ASSOC = 2.0           # Precizia corelației (funcție Gauss)

# --- Affective Modulation ---
ETA = 0.1                  
MU_AFFECT = 0.05           

# --- Environment Constants ---
NUM_FOOD_PATCHES = 6
FOOD_PATCH_AMOUNT_MIN = 20.0
FOOD_PATCH_AMOUNT_MAX = 50.0
TEMP_BASE_MAX = 35.0
TEMP_SPOT_1 = 15.0
TEMP_SPOT_2 = 10.0

# ==========================================
# ### AGENT CLASS ###
# ==========================================

class AllostaticAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        
        # Internal States (Homeostatic variables)
        self.E_int = self.random.uniform(INIT_ENERGY_MIN, INIT_ENERGY_MAX)
        self.T_int = IDEAL_TEMP
        self.E_max = MAX_ENERGY
        
        # Affective states
        self.current_h = 0.0
        self.current_beta = BETA_MAX
        self.is_alive = True
        
        # Memory & Signaling
        self.food_signal_timer = 0
        
        # --- NEW: Associative Memory (No coordinates) ---
        self.T_food_memory = IDEAL_TEMP  # "Unde am găsit hrană, cum era temperatura?"
        self.has_learned_food_temp = False

    def update_internal_state(self):
        if not self.is_alive: return

        x, y = self.pos
        
        # 1. Thermal Regulation
        T_env = self.model.temperature[x, y]
        self.T_int += ETA * (T_env - self.T_int)
        
        # 2. Metabolism
        self.E_int -= METABOLISM
        
        # 3. Eating & Associative Learning
        food_available = self.model.food[x, y]
        if food_available > 0.1 and self.E_int < self.E_max:
            space_in_stomach = self.E_max - self.E_int
            intake = min(FOOD_INTAKE, food_available, space_in_stomach)
            
            self.E_int += intake
            self.model.food[x, y] -= intake 
            
            # --- ÎNVĂȚARE: Dacă mănâncă, asociază temperatura internă cu succesul ---
            if intake > 1.0:
                self.food_signal_timer = FOOD_SIGNAL_DURATION
                
                # Actualizăm amintirea termică (Moving Average)
                self.T_food_memory = ( (1 - LEARNING_RATE_MEMORY) * self.T_food_memory + 
                                       LEARNING_RATE_MEMORY * self.T_int )
                self.has_learned_food_temp = True

        # 4. Check Survival
        if self.E_int <= 0 or self.T_int < 0 or self.T_int > 50:
            self.is_alive = False

    def choose_action(self):
        if not self.is_alive: return

        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
        moves = []
        scores = []
        
        # State detection
        is_hungry = self.E_int < CRITICAL_ENERGY
        
        for nx, ny in neighbors:
            # --- A. Pragmatic Value (Homeostatic Error) ---
            # Predict internal state if we move there
            T_env_next = self.model.temperature[nx, ny]
            T_int_next = self.T_int + ETA * (T_env_next - self.T_int)
            
            # Error H = |T_int - T_ideal| + |E_int - E_max|
            # (Note: Energy error is constant for all move options in this step)
            h_next = abs(T_int_next - IDEAL_TEMP) + abs(self.E_int - MAX_ENERGY)
            G_pragmatic = -h_next * WEIGHT_TEMP
            
            # --- B. Epistemic Value (Exploration/Novelty) ---
            shared_trace = self.model.shared_memory[nx, ny]
            G_epistemic = 1.0 / (1.0 + EXPLORATION_FACTOR * shared_trace)
            
            # --- C. Social Value (Scent) ---
            G_social = 0.0
            if is_hungry:
                scent_val = self.model.food_scent[nx, ny]
                G_social = SOCIAL_WEIGHT * scent_val 

            # --- D. Associative Value (NEW: Utilitatea Explorării) ---
            G_associative = 0.0
            if is_hungry and self.has_learned_food_temp:
                # Similitudine Gaussiana între temperatura vecinului și memoria termică a hranei
                temp_diff = T_env_next - self.T_food_memory
                G_associative = np.exp(-(temp_diff**2) / (2 * (SIGMA_ASSOC**2)))
                G_associative *= WEIGHT_ASSOCIATIVE

            # Total G (Expected Free Energy)
            G = G_pragmatic + (WEIGHT_EPISTEMIC * G_epistemic) + G_social + G_associative
            
            moves.append((nx, ny))
            scores.append(G)

        # Affective Modulation (Precision beta)
        h_current = abs(self.T_int - IDEAL_TEMP) + abs(self.E_int - MAX_ENERGY)
        dh = h_current - self.current_h
        self.current_h = h_current
        
        # If dh > 0 (error increasing), reduce beta -> more exploration/randomness
        target_beta = BETA_MIN if dh > 0 else BETA_MAX
        self.current_beta += MU_AFFECT * (target_beta - self.current_beta)

        # Softmax Action Selection
        scores = np.array(scores)
        scores_exp = np.exp(self.current_beta * (scores - np.max(scores)))
        probs = scores_exp / np.sum(scores_exp)
        
        idx = np.random.choice(len(moves), p=probs)
        new_pos = moves[idx]
        
        # Update shared memory trail
        self.model.shared_memory[new_pos] += 1.0
        self.model.grid.move_agent(self, new_pos)

    def step(self):
        self.update_internal_state()
        if self.is_alive:
            self.choose_action()