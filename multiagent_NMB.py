# multiagent_NMB.py
# Nested Markov Blankets Visualization Server
# Solara + Matplotlib

import solara
import matplotlib.pyplot as plt
import numpy as np
import asyncio

# Setam backend-ul Matplotlib pentru a evita problemele de thread pe Windows
import matplotlib
matplotlib.use('Agg')

from model import DualDriveModel
from agents import (
    COLOR_OK, COLOR_HUNGRY, COLOR_COLD, COLOR_HOT, 
    COLOR_FOOD, COLOR_TRAIL, WEIGHT_TEMP, WEIGHT_ENERGY,
    BETA_MAX, SimConfig
)

# ==========================================
# COMPONENTE UI CUSTOM
# ==========================================

@solara.component
def ValenceProgressBar(value, min_val=-2.5, max_val=2.5):
    """
    Bara de progres bidirectionala pentru Valenta (Mood).
    Rosu (stanga) = Negativ, Verde (dreapta) = Pozitiv.
    """
    val = max(min_val, min(value, max_val))
    
    # Normalizam la procentaj (0-100)
    range_span = max_val - min_val
    if range_span == 0: range_span = 1
    
    zero_pos = 50.0
    
    if val >= 0:
        left = zero_pos
        width = (val / max_val) * 50.0 if max_val > 0 else 0
        color = "#4caf50" # Green
    else:
        width = (abs(val) / abs(min_val)) * 50.0 if min_val < 0 else 0
        left = zero_pos - width
        color = "#f44336" # Red
        
    solara.HTML(unsafe_innerHTML=f"""
    <div style="width: 100%; background-color: #e0e0e0; height: 10px; border-radius: 4px; position: relative; margin-top: 5px; margin-bottom: 5px;">
        <div style="position: absolute; left: 50%; width: 1px; height: 100%; background-color: #555; z-index: 1;"></div>
        <div style="position: absolute; left: {left}%; width: {width}%; background-color: {color}; height: 100%; border-radius: 2px;"></div>
    </div>
    """)

# ==========================================
# VIZUALIZARE (MATPLOTLIB)
# ==========================================

def get_plot_figure(model, selected_agent_id=None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    # 1. Temperatura (Background)
    ax.imshow(model.temperature.T, origin='lower', cmap='coolwarm', alpha=0.3, vmin=0, vmax=40)
    
    # 1.1 Shared Memory (Feromoni/Urme trecere - Negru)
    mx, my, ms = [], [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            val = model.shared_memory[x, y]
            if val > 0.5:
                mx.append(x)
                my.append(y)
                ms.append(min(val * 2, 30))
    if mx:
        ax.scatter(mx, my, c='black', s=ms, alpha=0.15, marker='o', label='Trace')

    # 1.2 Food Scent (Urme mancare - Auriu)
    sx, sy, ss = [], [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            val = model.food_scent[x, y]
            if val > 0.1:
                sx.append(x)
                sy.append(y)
                ss.append(min(val * 15, 60))
    if sx:
        ax.scatter(sx, sy, c='gold', s=ss, alpha=0.6, marker='.', label='Scent')

    # 2. Hrana
    fx, fy = [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            if model.food[x, y] > 1.0:
                fx.append(x)
                fy.append(y)
    if fx:
        ax.scatter(fx, fy, c=COLOR_FOOD, s=20, marker='s', alpha=0.7, label='Food')

    # 3. Agenti
    for agent in model.agents:
        if not agent.is_alive: continue
        
        x, y = agent.pos
        
        # Determinam culoarea in functie de starea interna
        diff_T = agent.T_int - agent.T_pref 
        err_T_weighted = abs(diff_T) * WEIGHT_TEMP
        err_E_weighted = max(0, agent.E_crit - agent.E_int) * WEIGHT_ENERGY
        
        if err_E_weighted > err_T_weighted and err_E_weighted > 1.0:
            c = COLOR_HUNGRY
        elif err_T_weighted > err_E_weighted and err_T_weighted > 1.0:
            c = COLOR_HOT if diff_T > 0 else COLOR_COLD
        else:
            c = COLOR_OK
            
        # Evidentiere agent selectat
        if selected_agent_id is not None and agent.unique_id == selected_agent_id:
            edge_color = 'cyan'
            line_width = 2.5
            size = 150
            z_order = 100
        else:
            edge_color = 'black'
            line_width = 1.0
            size = 80
            z_order = 10
            
        ax.scatter(x, y, c=c, s=size, edgecolors=edge_color, linewidth=line_width, zorder=z_order)

    ax.set_xlim(-0.5, model.grid.width-0.5)
    ax.set_ylim(-0.5, model.grid.height-0.5)
    ax.axis('off')
    plt.tight_layout()
    
    return fig

# ==========================================
# LOGICA PRINCIPALA (SOLARA)
# ==========================================

def create_model():
    return DualDriveModel()

@solara.component
def Page():
    # --- State ---
    reset_ctr, set_reset = solara.use_state(0)
    model = solara.use_memo(create_model, dependencies=[reset_ctr])
    
    tick, set_tick = solara.use_state(0)
    is_playing, set_playing = solara.use_state(False)
    
    # Selectie Agent
    selected_agent_id, set_selected_agent_id = solara.use_state(None)

    # --- Simulation Loop ---
    def run_loop():
        if not is_playing: return
        
        async def loop():
            while is_playing:
                if len(model.agents) == 0: # Safety check
                    break
                model.step()
                set_tick(lambda t: t + 1)
                await asyncio.sleep(0.1)
        
        task = asyncio.create_task(loop())
        def cleanup(): task.cancel()
        return cleanup

    solara.use_effect(run_loop, [is_playing])

    # --- Handlers ---
    def on_step():
        model.step()
        set_tick(tick + 1)

    def on_reset():
        set_playing(False)
        set_reset(reset_ctr + 1)
        set_tick(0)
        set_selected_agent_id(None)

    # --- Stats ---
    agents_list = list(model.agents)
    alive_agents = [a for a in agents_list if a.is_alive]
    dead_count = model.dead_count if hasattr(model, 'dead_count') else (len(agents_list) - len(alive_agents))
    
    # Lista ID-uri pentru dropdown
    agent_ids = sorted([a.unique_id for a in agents_list])

    # --- UI Layout ---
    with solara.Sidebar():
        solara.Markdown("## 🧠 Nested FEP Monitor")

        # --- Parametri Dinamici (Sliders) ---
        # Folosim variabile locale pentru a forta re-randarea UI-ului, apoi actualizam SimConfig
        w_pragmatic, set_w_pragmatic = solara.use_state(SimConfig.WEIGHT_PRAGMATIC)
        w_epistemic, set_w_epistemic = solara.use_state(SimConfig.WEIGHT_EPISTEMIC)
        w_social, set_w_social = solara.use_state(SimConfig.SOCIAL_WEIGHT)
        w_memory, set_w_memory = solara.use_state(SimConfig.MEMORY_WEIGHT)

        # Actualizare Config Global
        SimConfig.WEIGHT_PRAGMATIC = w_pragmatic
        SimConfig.WEIGHT_EPISTEMIC = w_epistemic
        SimConfig.SOCIAL_WEIGHT = w_social
        SimConfig.MEMORY_WEIGHT = w_memory
        
        # Controale Simulare
        with solara.Row():
            solara.Button("Step", on_click=on_step, color="warning")
            solara.Button("Play/Pause", on_click=lambda: set_playing(not is_playing), color="success" if is_playing else "primary")
            solara.Button("Reset", on_click=on_reset, color="error")

        solara.Markdown("### 🎛️ Active Inference Weights")
        
        solara.Markdown(f"**Pragmatic (Survival):** {w_pragmatic:.1f}")
        solara.SliderFloat(label="", value=w_pragmatic, min=0.0, max=5.0, step=0.1, on_value=set_w_pragmatic)

        solara.Markdown(f"**Epistemic (Curiosity):** {w_epistemic:.1f}")
        solara.SliderFloat(label="", value=w_epistemic, min=0.0, max=5.0, step=0.1, on_value=set_w_epistemic)

        solara.Markdown(f"**Social (Pheromones):** {w_social:.1f}")
        solara.SliderFloat(label="", value=w_social, min=0.0, max=10.0, step=0.1, on_value=set_w_social)

        solara.Markdown(f"**Cognitive (Memory):** {w_memory:.1f}")
        solara.SliderFloat(label="", value=w_memory, min=0.0, max=5.0, step=0.1, on_value=set_w_memory)

        solara.Markdown("---")
        solara.Markdown(f"**Steps:** {tick}")
        solara.Markdown(f"**Alive:** {len(alive_agents)} | **Dead:** {dead_count}")
        
        solara.Markdown("---")
        solara.Markdown("### 🕵️ Agent Inspector")
        
        # Dropdown Selectie
        solara.Select(
            label="Select Agent ID",
            values=[None] + agent_ids,
            value=selected_agent_id,
            on_value=set_selected_agent_id
        )
        
        # Detalii Agent Selectat
        if selected_agent_id is not None:
            # Cautam agentul in lista curenta (poate fi mort, dar obiectul exista)
            agent = next((a for a in agents_list if a.unique_id == selected_agent_id), None)
            
            if agent:
                status_icon = "🟢" if agent.is_alive else "💀"
                solara.Markdown(f"**Status:** {status_icon} (ID: {agent.unique_id})")
                
                # 1. Energy
                solara.Text(f"Energy: {agent.E_int:.1f} / {agent.E_max}")
                solara.ProgressLinear(value=(agent.E_int / agent.E_max) * 100, color="brown")
                
                # 2. Temperature
                solara.Text(f"Temperature: {agent.T_int:.1f} (Pref: {agent.T_pref})")
                # Normalizam vizual intre 0 si 40 grade
                solara.ProgressLinear(value=(agent.T_int / 40.0) * 100, color="blue")
                
                # 3. Valence (Mood)
                solara.Text(f"Valence (Mood): {agent.valence:.2f}")
                ValenceProgressBar(agent.valence, min_val=-2.0, max_val=2.0)
                
                # 4. Arousal (NOU)
                # Arousal e de obicei pozitiv. Scalam relativ la un maxim estimat (ex: 5.0)
                arousal_pct = min(100, (agent.affective_arousal / 5.0) * 100)
                solara.Text(f"Arousal (Alertness): {agent.affective_arousal:.2f}")
                solara.ProgressLinear(value=arousal_pct, color="orange")
                
            else:
                solara.Warning("Agent not found.")

    # --- Main View ---
    solara.FigureMatplotlib(get_plot_figure(model, selected_agent_id))