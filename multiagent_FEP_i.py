# multiagent_FEP_i.py
# Versiune interactiva bazata pe Plotly pentru selectia agentilor cu mouse-ul

import solara
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import asyncio

# Importam modelul si constantele existente
from model import DualDriveModel
from agents import (
    NUM_AGENTS, COLOR_OK, COLOR_HUNGRY, COLOR_COLD, COLOR_HOT, 
    COLOR_FOOD, COLOR_TRAIL, WEIGHT_TEMP, WEIGHT_ENERGY, BETA_MAX
)

# ==========================================
# Visualization (PLOTLY)
# ==========================================

def get_plot_figure(model, selected_agent_id=None):
    """
    Genereaza graficul interactiv Plotly.
    """
    fig = go.Figure()

    # 1. Heatmap (Temperature) - Background
    # Folosim hoverinfo='skip' pentru a nu interfera cu click-ul pe agenti
    fig.add_trace(go.Heatmap(
        z=model.temperature.T,
        x=np.arange(model.grid.width),
        y=np.arange(model.grid.height),
        colorscale='RdBu', reversescale=True, zmin=0, zmax=40,
        showscale=False,
        hoverinfo='skip', 
        opacity=1,
        uid="heatmap"
    ))

    # 2. Food patches
    fx, fy, fs = [], [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            val = model.food[x, y]
            if val > 1.0:
                fx.append(x)
                fy.append(y)
                fs.append(min(val * 0.6, 12))
    
    fig.add_trace(go.Scatter(
        x=fx, y=fy, mode='markers',
        marker=dict(color=COLOR_FOOD, size=fs, line=dict(color='green', width=1), opacity=1),
        name='Food', hoverinfo='skip', uid="food"
    ))

    # 3. Social Scent Trails
    sx, sy, ss = [], [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            val = model.food_scent[x, y]
            if val > 0.1:
                sx.append(x)
                sy.append(y)
                ss.append(min(val * 5, 10))
    
    fig.add_trace(go.Scatter(
        x=sx, y=sy, mode='markers',
        marker=dict(color=COLOR_TRAIL, size=ss, opacity=0.4),
        name='Trail', hoverinfo='skip', uid="trail"
    ))

    # 4. Agents
    ax, ay, ac, asize, aedge, awidth, aids, atext = [], [], [], [], [], [], [], []
    
    for agent in model.agents:
        if agent.pos is None: continue # Safety check for dead agents
        x, y = agent.pos
        
        # --- Logic for Red/Blue/Brown (Copiat din agents.py) ---
        diff_T = agent.T_int - agent.T_pref 
        err_T_weighted = abs(diff_T) * WEIGHT_TEMP
        err_E_weighted = max(0, agent.E_crit - agent.E_int) * WEIGHT_ENERGY
        
        if err_E_weighted > err_T_weighted and err_E_weighted > 1.0:
            c = COLOR_HUNGRY
        elif err_T_weighted > err_E_weighted and err_T_weighted > 1.0:
            if diff_T > 0: c = COLOR_HOT
            else: c = COLOR_COLD
        else:
            c = COLOR_OK
            
        # Highlight selection
        if selected_agent_id is not None and agent.unique_id == selected_agent_id:
            edge_c = 'red' # Culoare evidentiere
            line_w = 3
            size = 16
        else:
            edge_c = 'black'
            line_w = 1
            size = 12

        ax.append(x)
        ay.append(y)
        ac.append(c)
        asize.append(size)
        aedge.append(edge_c)
        awidth.append(line_w)
        aids.append(agent.unique_id)
        atext.append(f"Agent {agent.unique_id}<br>E: {agent.E_int:.1f}<br>T: {agent.T_int:.1f}")

    # Adaugam agentii ca un singur trace Scatter
    fig.add_trace(go.Scatter(
        x=ax, y=ay, 
        mode='markers',
        marker=dict(color=ac, size=asize, line=dict(color=aedge, width=awidth)),
        customdata=aids, # AICI stocam ID-ul pentru click event
        hovertext=atext,
        hoverinfo='text',
        name='Agents',
        uid="agents"
    ))

    # 5. Thermal Memory Field — contour lines (isobars) of thermal attraction
    # Shown only when an agent is selected and has feeding memories.
    # Contours indicate zones where the agent expects to find food based on T_int context.
    if selected_agent_id is not None:
        sel_agent = next((a for a in model.agents if a.unique_id == selected_agent_id), None)
        if sel_agent and sel_agent.is_alive and sel_agent.thermal_memory:
            memory_field = np.zeros((model.grid.width, model.grid.height))
            for x in range(model.grid.width):
                for y in range(model.grid.height):
                    memory_field[x, y] = sel_agent._memory_value(model.temperature[x, y])
            vmax = memory_field.max()
            if vmax > 1e-6:
                # Show contours only above 80% of max value — tight focus on peak thermal attraction
                threshold = vmax * 0.80
                n_contours = 5
                fig.add_trace(go.Contour(
                    z=memory_field.T,
                    x=np.arange(model.grid.width),
                    y=np.arange(model.grid.height),
                    colorscale=[[0, 'rgba(0,200,0,0)'], [1, 'rgba(0,200,0,1)']],
                    contours=dict(
                        start=threshold,
                        end=vmax,
                        size=(vmax - threshold) / max(n_contours, 1),
                        showlabels=False,
                        coloring='lines',  # lines only, no fill
                    ),
                    line=dict(width=2),
                    showscale=False,
                    hoverinfo='skip',
                    uid="memory_contour"
                ))



    # Calculam dimensiunile figurii pentru a mentine aspect ratio corect
    # Inaltimea fixa de 600px, latimea se adapteaza
    plot_height = 600
    plot_width = int(plot_height * (model.grid.width / model.grid.height))

    # Layout settings
    fig.update_layout(
        xaxis=dict(range=[-0.5, model.grid.width-0.5], visible=False, fixedrange=True),
        yaxis=dict(range=[-0.5, model.grid.height-0.5], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        plot_bgcolor='white',
        clickmode='event', # Activeaza evenimentele de click
        hovermode='closest',
        hoverdistance=50, # Mareste raza de detectie a click-ului (snap)
        uirevision='constant', # Pastreaza starea de zoom/pan
        transition={'duration': 0}, # Dezactiveaza animatiile pentru raspuns rapid
        width=plot_width, height=plot_height
    )
    
    return fig

# ==========================================
# Solara GUI
# ==========================================

@solara.component
def ValenceProgressBar(value, min_val=-5.0, max_val=5.0):
    """
    Custom progress bar that starts from the middle (0).
    """
    val = max(min_val, min(value, max_val))
    
    # Normalize to percentage (0 to 100 relative to full width)
    # 0 is at 50%
    range_span = max_val - min_val
    if range_span == 0: range_span = 1
    
    zero_pos = 50.0
    
    if val >= 0:
        left = zero_pos
        width = (val / max_val) * 50.0 if max_val > 0 else 0
        color = "green"
    else:
        width = (abs(val) / abs(min_val)) * 50.0 if min_val < 0 else 0
        left = zero_pos - width
        color = "red"
        
    solara.HTML(unsafe_innerHTML=f"""
    <div style="width: 100%; background-color: #e0e0e0; height: 12px; border-radius: 4px; position: relative; margin-top: 5px;">
        <!-- Center Marker -->
        <div style="position: absolute; left: 50%; width: 2px; height: 100%; background-color: #555; z-index: 1;"></div>
        <!-- Bar -->
        <div style="position: absolute; left: {left}%; width: {width}%; background-color: {color}; height: 100%; border-radius: 2px;"></div>
    </div>
    """)

def create_model():
    return DualDriveModel()

@solara.component
def Page():
    reset_ctr, set_reset = solara.use_state(0)
    model = solara.use_memo(create_model, dependencies=[reset_ctr])
    
    tick, set_tick = solara.use_state(0)
    is_playing, set_playing = solara.use_state(False)
    
    # Starea pentru agentul selectat
    selected_agent_id, set_selected_agent_id = solara.use_state(None)

    # --- Simulation Loop ---
    def run_loop():
        if not is_playing: return
        
        async def loop():
            while True:
                if len(model.agents) == 0:
                    set_playing(False)
                    break
                model.step()
                set_tick(lambda t: t + 1)
                await asyncio.sleep(0.1)
        
        task = asyncio.create_task(loop())
        def cleanup(): task.cancel()
        return cleanup

    solara.use_effect(run_loop, [is_playing])

    # --- Event Handlers ---
    def on_step():
        model.step()
        set_tick(tick + 1)

    def on_reset():
        set_playing(False)
        set_reset(reset_ctr + 1)
        set_tick(0)
        set_selected_agent_id(None)

    def on_play():
        set_playing(not is_playing)

    def on_export():
        """Exporta datele colectate in CSV"""
        agent_df = model.datacollector.get_agent_vars_dataframe()
        
        agent_df.to_csv("fep_swarm_agent_data.csv")
        print("✅ Data exported to 'fep_swarm_agent_data.csv'")

    # Handler pentru click pe grafic
    def on_plot_click(data):
        # data contine informatii despre punctul click-uit
        # print(f"DEBUG: Click data received: {data}") 
        
        if data is None:
            return
            
        points = data.get('points')
        if points is None:
            return

        agent_id = None

        # Logica bazata pe log-ul primit: {'point_indexes': [9], ...}
        # Solara returneaza indecsii punctelor, nu obiectele complete cu customdata
        if isinstance(points, dict) and 'point_indexes' in points:
            indexes = points['point_indexes']
            if isinstance(indexes, list) and len(indexes) > 0:
                idx = indexes[0]
                # Reconstruim lista agentilor pentru a gasi ID-ul dupa index
                # Ordinea de iterare in model.agents este aceeasi ca la generarea graficului
                # Filtram agentii fara pozitie pentru a pastra sincronizarea cu graficul
                agents_list = [a for a in model.agents if a.pos is not None]
                if 0 <= idx < len(agents_list):
                    agent_id = agents_list[idx].unique_id
        
        if agent_id is not None:
            # print(f"DEBUG: Selecting Agent ID: {agent_id}")
            set_selected_agent_id(agent_id)
            

    # --- Stats Calculation ---
    agents = list(model.agents)
    alive_agents = [a for a in agents if a.is_alive]
    n_alive = len(alive_agents)
    
    if n_alive > 0:
        avg_E = sum(a.E_int for a in alive_agents) / n_alive
        avg_T = sum(a.T_int for a in alive_agents) / n_alive
        avg_Valence = sum(a.valence_integrated for a in alive_agents) / n_alive
    else:
        avg_E = 0; avg_T = 0; avg_Valence = 0

    # --- UI ---
    with solara.Sidebar():
        solara.Markdown("## 🧠 Swarm FEP Agents")
        solara.Markdown("Multi-agent simulation with shared food trails.")
        
        with solara.Row():
            solara.Button("Step", on_click=on_step, color="warning")
            solara.Button("Play/Pause", on_click=on_play, color="success" if is_playing else "primary")
            solara.Button("Reset", on_click=on_reset, color="error")
            solara.Button("Export CSV", on_click=on_export, color="primary", outlined=True)
            
            
        solara.Markdown("---")
        
        # Dropdown sincronizat cu selectia de pe grafic
        agent_ids = sorted([a.unique_id for a in model.agents])
        solara.Select(
            label="Select Agent (or click graph when paused)", 
            values=[None] + agent_ids, 
            value=selected_agent_id, 
            on_value=set_selected_agent_id
        )
        
        if selected_agent_id is not None:
            agent = next((a for a in model.agents if a.unique_id == selected_agent_id), None)
            
            if agent and agent.is_alive:
                solara.Markdown(f"### Agent {agent.unique_id} Details")
                
                # Energy
                solara.Text(f"Energy: {agent.E_int:.1f} / {agent.E_max}")
                solara.ProgressLinear(value=((agent.E_int / max(1, agent.E_max)) * 100), color="brown")
                solara.HTML(tag="div", style={"height": "15px"})
                
                # Temperature
                solara.Text(f"Temperature: {agent.T_int:.1f} (Pref: {agent.T_pref})")
                max_temp = np.max(model.temperature)
                solara.ProgressLinear(value=((agent.T_int / max(1, max_temp)) * 100), color="blue")
                solara.HTML(tag="div", style={"height": "15px"})
                
                # Precision
                solara.Text(f"Precision (Beta): {agent.current_beta:.2f}")
                solara.ProgressLinear(value=((agent.current_beta / BETA_MAX) * 100), color="purple")
                solara.HTML(tag="div", style={"height": "15px"})
                
                # Valence
                solara.Text(f"Valence (Mood): {agent.valence_integrated:.3f}")
                bound = max(1.0, agent.valence_bound)
                ValenceProgressBar(agent.valence_integrated, min_val=-bound, max_val=bound)

                # Thermal Memory
                solara.HTML(tag="div", style={"height": "15px"})
                n_traces = len(agent.thermal_memory)
                solara.Markdown(f"**🧠 Food Memory:** {n_traces} / 20 traces")
                if n_traces > 0:
                    # Calculate current dynamic sigma for display
                    from agents import MEMORY_SIGMA_T, MEMORY_GAMMA
                    dynamic_sigma = MEMORY_SIGMA_T * np.exp(-MEMORY_GAMMA * agent.valence_integrated)
                    
                    contexts = ", ".join(f"{t:.1f}°C" for t, _ in agent.thermal_memory[-5:])
                    solara.Text(f"Last {min(n_traces, 5)} contexts: {contexts}")
                    solara.Text(f"Memory Precision (Sigma): {dynamic_sigma:.2f}°C")
                    avg_reward = sum(r for _, r in agent.thermal_memory) / n_traces
                    solara.Text(f"Avg reward: {avg_reward:.1f} energy")
                    solara.Markdown("*🟢 Green overlay = thermal attraction field*")
                else:
                    solara.Text("No feeding events recorded yet.")

            elif agent and not agent.is_alive:
                solara.Error(f"Agent {selected_agent_id} is DEAD 💀")
            else:
                solara.Warning("Agent dead. RIP.")

        # Global Stats
        solara.Markdown("---")
        solara.Markdown(f"### Steps: {tick}")
        solara.Markdown(f"**Alive:** {n_alive} / **Dead:** {model.dead_count}")
        
        solara.Markdown("---")
        solara.Markdown(f"**Avg Energy:** {avg_E:.1f}")
        solara.Markdown(f"**Avg Temp:** {avg_T:.1f}")
        solara.Markdown(f"**Avg Mood:** {avg_Valence:.2f}")

        # Legend
        solara.Markdown("---")
        solara.Markdown("**Legend:**")
        solara.Markdown(f"⚪ **White:** OK (Optimal)")
        solara.Markdown(f"🟤 **Brown:** Hungry")
        solara.Markdown(f"🔵 **Blue:** Cold")
        solara.Markdown(f"🔴 **Red:** Hot")
        solara.Markdown(f"🟢 **Green:** Food")
        if selected_agent_id is not None:
             solara.Markdown(f"🔵 **Cyan Border:** Selected Agent")

        # Theoretical Framework
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

        # 3. Active Inference ($G$)
        solara.Markdown("**3. Active Inference ($G$):**")
        solara.Markdown(r"""
        Agents select moves to minimize Expected Free Energy ($G$):
        $$ G(a) = G_{pragmatic} + G_{epistemic} + G_{social} + G_{memory} $$
        """)
        
        # 3.1 Associative Memory
        solara.Markdown("**3.1 Associative Thermal Memory:**")
        solara.Markdown(r"""
        Agents remember the internal temperature ($T_{int}$) when finding food.
        The cumulative attraction $V_{hat}$ is a Sum-KDE:
        $$ V_{hat}(T) = \sum_{k} r_k \cdot e^{-\frac{(T - T_k)^2}{2\sigma^2}} $$
        $$ G_{memory}(a) = -\alpha \cdot V_{hat}(T_{pred}^{(a)}) $$
        """)
        
        # 4. Affect and Precision
        solara.Markdown("**4. Affect and Precision:**")
        solara.Markdown(r"""
        Affect (Mood) is the rate of change of the error. This integrated valence determines the agent's Precision β.
        $$\beta_t = -\frac{H_t - H_{t-1}}{\Delta t}$$
        """)

        # 5. Math - Action Selection
        solara.Markdown("**5. Action Selection:**")
        solara.Markdown(r"""
        Softmax probability modulated by Precision ($\beta$):
        $$ P(a) = \frac{e^{\beta \cdot G(a)}}{\sum e^{\beta \cdot G(a_i)}} $$
        """)
    
    # Main Area
    fig = get_plot_figure(model, selected_agent_id)
    solara.FigurePlotly(fig, on_click=on_plot_click)