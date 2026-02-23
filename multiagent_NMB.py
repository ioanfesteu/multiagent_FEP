# multiagent_NMB.py
# Nested Markov Blankets Visualization Server
# Solara + Matplotlib

import solara
import matplotlib.pyplot as plt
import numpy as np
import asyncio
from matplotlib.figure import Figure

# Setam backend-ul Matplotlib pentru a evita problemele de thread pe Windows
import matplotlib
matplotlib.use('Agg')

from model import DualDriveModel
from agents import (
    COLOR_OK, COLOR_HUNGRY, COLOR_COLD, COLOR_HOT, 
    COLOR_FOOD, COLOR_TRAIL, WEIGHT_TEMP, WEIGHT_ENERGY,
    BETA_MAX, SimConfig, AROUSAL_SCALING
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

def get_plot_figure(model, selected_agent_id=None, show_contours=True):
    # Determinam daca afisam detaliile de memorie (Contururi + Grafic)
    show_details = (selected_agent_id is not None) and show_contours

    if show_details:
        # Marim figura pe verticala pentru a acomoda graficul de memorie
        fig = Figure(figsize=(8, 11))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
        ax = fig.add_subplot(gs[0, :])
    else:
        # Figura patrata standard doar pentru harta
        fig = Figure(figsize=(8, 8))
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

    # 1.3 Thermal Memory Overlay (Doar pentru agentul selectat)
    if show_details:
        agent = next((a for a in model.agents if a.unique_id == selected_agent_id), None)
        
        # Verificam daca agentul exista si are amintiri
        if agent and hasattr(agent, 'thermal_memory') and agent.thermal_memory:
            # Preluam sigma din configuratie
            sigma = getattr(SimConfig, 'MEMORY_SIGMA_T', 6.0)
            
            # Initializam campul de memorie (dimensiunea gridului)
            mem_field = np.zeros_like(model.temperature)
            
            # Calculam suma campurilor receptive liniare (Vectorizat)
            # V_hat(T) = sum(r_k * max(0, 1 - |T - T_k|/sigma))
            for t_k, r_k in agent.thermal_memory:
                diff = np.abs(model.temperature - t_k)
                kernel = np.maximum(0, 1.0 - (diff / sigma))
                mem_field += r_k * kernel
            
            # In loc de overlay, folosim linii de contur (izobare) pentru a nu obtura harta
            vmax = mem_field.max()
            if vmax > 0.1: # Doar daca exista un camp semnificativ
                # Desenam 5 linii de contur intre 50% si 100% din valoarea maxima
                levels = np.linspace(vmax * 0.5, vmax, 5)
                ax.contour(mem_field.T, levels=levels, origin='lower', cmap='Greens', linewidths=2, zorder=5)
 
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
    
    # ==========================================
    # SUBPLOT 2: Profilul Memoriei (Sum-KDE)
    # ==========================================
    if show_details:
        ax_mem = fig.add_subplot(gs[1, 0])
        
        agent = next((a for a in model.agents if a.unique_id == selected_agent_id), None)
        if agent and hasattr(agent, 'thermal_memory') and agent.thermal_memory:
            # Generam curba continua
            t_range = np.linspace(0, 40, 200)
            sigma = getattr(SimConfig, 'MEMORY_SIGMA_T', 6.0)
            v_values = []
            for t in t_range:
                val = 0
                for t_k, r_k in agent.thermal_memory:
                    # Linear Receptive Field logic
                    dist = abs(t - t_k)
                    if dist < sigma:
                        val += r_k * (1.0 - (dist / sigma))
                v_values.append(val)
            
            # Plotare curba
            ax_mem.plot(t_range, v_values, color='green', lw=2, label='Attraction Field')
            ax_mem.fill_between(t_range, v_values, color='green', alpha=0.2)
            
            # Plotare "batoane" pentru amintirile individuale (Raw Data)
            for t_k, r_k in agent.thermal_memory:
                ax_mem.vlines(t_k, 0, r_k, colors='black', linestyles='solid', lw=1.5, alpha=0.6)
                
            ax_mem.set_title(f"Agent {selected_agent_id}: Thermal Memory Profile (Linear Receptive Fields)", fontsize=9)
        else:
            ax_mem.text(0.5, 0.5, "No memory traces yet", ha='center', va='center', color='gray')

        ax_mem.set_xlim(0, 40)
        ax_mem.set_xlabel("Temperature (°C)", fontsize=8)
        ax_mem.set_ylabel("Expected Reward", fontsize=8)
        ax_mem.grid(True, linestyle='--', alpha=0.5)
        
        # ==========================================
        # SUBPLOT 3: Radar Chart (Weights)
        # ==========================================
        ax_radar = fig.add_subplot(gs[1, 1], polar=True)
        if agent:
            # Calculam ponderile efective (Level 1 + Level 2 Modulat)
            saturation = np.clip(agent.E_int / agent.E_max, 0.0, 1.0)
            hunger_drive = (1.0 - saturation) ** 2
            modulator = agent.affective_arousal * AROUSAL_SCALING if agent.affective_arousal > 0.1 else 0.0
            
            val_P = SimConfig.WEIGHT_PRAGMATIC
            val_E = SimConfig.WEIGHT_EPISTEMIC
            val_S = SimConfig.SOCIAL_WEIGHT * modulator * hunger_drive
            val_M = SimConfig.MEMORY_WEIGHT * modulator * hunger_drive
            
            values = [val_P, val_E, val_S, val_M]
            labels = ['Pragmatic', 'Epistemic', 'Social', 'Memory']
            
            # Configurare Radar
            N = len(labels)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            values += values[:1] # Inchidem bucla
            angles += angles[:1]
            
            ax_radar.plot(angles, values, linewidth=2, linestyle='solid', color='purple')
            ax_radar.fill(angles, values, 'purple', alpha=0.2)
            
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(labels, size=8)
            ax_radar.set_title("Current Drive Weights", size=9, pad=10)
            ax_radar.set_yticks([]) # Ascundem cercurile concentrice pentru claritate
        else:
            ax_radar.axis('off')
    
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
    show_contours, set_show_contours = solara.use_state(True)

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

        solara.SliderFloat(label=f"Pragmatic (Survival): {w_pragmatic:.1f}", value=w_pragmatic, min=0.0, max=5.0, step=0.1, on_value=set_w_pragmatic)
        solara.SliderFloat(label=f"Epistemic (Curiosity): {w_epistemic:.1f}", value=w_epistemic, min=0.0, max=5.0, step=0.1, on_value=set_w_epistemic)
        solara.SliderFloat(label=f"Social (Pheromones): {w_social:.1f}", value=w_social, min=0.0, max=10.0, step=0.1, on_value=set_w_social)
        solara.SliderFloat(label=f"Cognitive (Memory): {w_memory:.1f}", value=w_memory, min=0.0, max=5.0, step=0.1, on_value=set_w_memory)

        solara.Markdown("---")
        solara.Markdown(f"**Steps:** {tick}")
        solara.Markdown(f"**Alive:** {len(alive_agents)} | **Dead:** {dead_count}")
        
        solara.Markdown("---")
        # solara.Markdown("### 🕵️ Agent Inspector")
        
        # Dropdown Selectie
        solara.Select(
            label="Select Agent ID",
            values=[None] + agent_ids,
            value=selected_agent_id,
            on_value=set_selected_agent_id
        )
        
        if selected_agent_id is not None:
            solara.Checkbox(label="Show Memory", value=show_contours, on_value=set_show_contours)
        
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
    solara.FigureMatplotlib(get_plot_figure(model, selected_agent_id, show_contours))