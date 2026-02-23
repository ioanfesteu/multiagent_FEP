# model.py
import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agents import (
    AllostaticAgent, 
    GRID_WIDTH, GRID_HEIGHT, NUM_AGENTS, SEED,
    NUM_FOOD_PATCHES, FOOD_PATCH_AMOUNT_MIN, FOOD_PATCH_AMOUNT_MAX, FOOD_INTAKE,
    FOOD_REGROWTH_RATE, FOOD_REGROWTH_THRESHOLD,
    SCENT_DECAY, MEMORY_DECAY,
    TEMP_BASE_MAX, TEMP_SPOT_1, TEMP_SPOT_2
)

# ==========================================
# Environment Fields
# ==========================================

def generate_temperature_field(width, height):
    field = np.zeros((width, height))
    
    for x in range(width):
        for y in range(height):
            # Warm zones (Global Plateau)
            field[x, y] += TEMP_BASE_MAX * np.exp(-((x - width/2)**2 + (y - height/2)**2) / (width*7.5))
            # Local optima (Hot spots)
            field[x, y] += TEMP_SPOT_1 * np.exp(-((x - width*0.2)**2 + (y - height*0.8)**2) / 70)
            field[x, y] += TEMP_SPOT_2 * np.exp(-((x - width*0.75)**2 + (y - height*0.25)**2) / 60)
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
# Model (OPTIMIZED)
# ==========================================

class DualDriveModel(Model):
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, num_agents=NUM_AGENTS, seed=SEED):
        super().__init__(seed=seed)
        self.grid = MultiGrid(width, height, torus=False)
        
        # Fields
        self.temperature = generate_temperature_field(width, height)
        self.food = generate_food_field(width, height, n_patches=NUM_FOOD_PATCHES)

        self.growing_patch_location = None
        # FIX: Pragul trebuie calculat pe baza cantității *reale* de hrană de la început,
        # nu pe baza unei valori teoretice minime.
        self.total_food_threshold = np.sum(self.food) * FOOD_REGROWTH_THRESHOLD
        # Global Scent
        self.food_scent = np.zeros((width, height)) 
        
        # Global Navigation Memory (Shared Stigmergy)
        self.shared_memory = np.zeros((width, height))
        
        self.directions = [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,1),(1,-1),(-1,-1)]

        # ✅ FIX: Statistics for dead agents
        self.dead_count = 0

        # Spawn Agents
        for i in range(num_agents):
            agent = AllostaticAgent(self)
            rx = self.random.randint(0, width-1)
            ry = self.random.randint(0, height-1)
            self.grid.place_agent(agent, (rx, ry))
            
        # ==========================================
        # DATA COLLECTOR (Scientific Evaluation)
        # ==========================================
        self.datacollector = DataCollector(
            agent_reporters={
                "Energy": "E_int",
                "Temp": "T_int",
                "Valence": "valence_integrated",
                "Beta": "current_beta",
                "Alive": "is_alive",
                "X": lambda a: a.pos[0],
                "Y": lambda a: a.pos[1]
            }
        )

    def _add_gaussian_food(self, cx, cy, amount, sigma=3.0):
        """Adds food in a Gaussian pattern centered at (cx, cy)."""
        # FIX: The 'amount' parameter should represent the TOTAL food added, not the peak height.
        # The integral of a 2D Gaussian A*exp(-(r^2)/(2s^2)) is 2*pi*s^2*A.
        # To make the total integral equal to 'amount', the peak A must be amount / (2*pi*s^2).
        # This was the bug causing the new patch to be deposited "all at once".
        peak_height = amount / (2 * np.pi * sigma**2)

        radius = int(sigma * 3)
        x_min = max(0, cx - radius)
        x_max = min(self.grid.width, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(self.grid.height, cy + radius + 1)

        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(x_range, y_range, indexing='ij')

        dist_sq = (xx - cx)**2 + (yy - cy)**2
        growth = peak_height * np.exp(-dist_sq / (2 * sigma**2))
        self.food[x_min:x_max, y_min:y_max] += growth

    def step(self):
        """✅ FIX: Optimized for dead agent cleanup and NumPy operations"""
        # 1. Agents step
        agents = list(self.agents)
        self.random.shuffle(agents)  # ✅ FIX: Using self.random (no longer need random_gen)
        
        dead_agents = []
        for agent in agents:
            # Save state before
            was_alive = agent.is_alive
            
            # Execute step
            agent.step()
            
            # ✅ FIX: Death detection in this step
            if was_alive and not agent.is_alive:
                dead_agents.append(agent)
        
        # ✅ FIX: Cleanup dead agents from grid and agent_set
        for agent in dead_agents:
            self.grid.remove_agent(agent)
            self.agents.remove(agent)
            self.dead_count += 1
            
        # 2. Global Environment Decay
        # ✅ FIX: Optimized to reduce NumPy temporaries on Windows
        np.multiply(self.food_scent, SCENT_DECAY, out=self.food_scent)
        np.putmask(self.food_scent, self.food_scent < 0.05, 0)
        
        # 3. Food Regrowth Logic
        # A new patch starts growing slowly at a new random location
        # once the total food on the map drops below a threshold.
        
        # 3.1. Designate a new growth spot if needed
        if self.growing_patch_location is None:
            if np.sum(self.food) < self.total_food_threshold:
                # Time to choose a new spot
                gx = self.random.randint(5, self.grid.width - 5)
                gy = self.random.randint(5, self.grid.height - 5)
                self.growing_patch_location = (gx, gy)
                self._add_gaussian_food(gx, gy, FOOD_INTAKE, sigma=3.0)

        # 3.2. If a growth spot is designated, make it grow
        if self.growing_patch_location is not None:
            gx, gy = self.growing_patch_location
            self._add_gaussian_food(gx, gy, FOOD_REGROWTH_RATE, sigma=3.0)
        
        # Decay Shared Memory
        np.multiply(self.shared_memory, MEMORY_DECAY, out=self.shared_memory)
        np.putmask(self.shared_memory, self.shared_memory < 0.05, 0)
        
        # Collect Data
        self.datacollector.collect(self)