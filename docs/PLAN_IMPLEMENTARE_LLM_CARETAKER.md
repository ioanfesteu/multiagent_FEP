# 🧠 PLAN IMPLEMENTARE: LLM Caretaker & Observer System

## 📋 Viziune Generală

Extindem simularea FEP multiagent cu un **sistem extern de observare și îngrijire bazat pe LLM**. Scopurile sunt:

1. **Testing biological resilience**: Cât de mult pot supraviețui agenții cu intervențiile externe?
2. **Studying LLM ethics**: Observarea comportamentului emotional/ethical al LLM-ului – devine prea grijuliu? Indiferent?
3. **Active inference philosophy**: Investigarea dacă un agent LLM extern adăugă o nouă strat de "consciousness" la ecosistem.

---

## 🏗️ Arhitectura Generală

```
┌─────────────────────────────────────────────────────────────┐
│                      FLASK SERVER                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │      DualDriveModel (Simularea FEP)                 │  │
│  │  - Agents, Temperature Field, Food Field           │  │
│  │  - Mesa DataCollector                              │  │
│  │  - Shared Memory, Food Scent                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                         ↑                                    │
│                      REST API                               │
│                         ↓                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │    Endpoints:                                         │  │
│  │  - GET /state (stare actuală)                        │  │
│  │  - GET /world (descriere lume: heatmap, food)       │  │
│  │  - GET /agents (liste agenti vii/morti)            │  │
│  │  - GET /history (date din DataCollector)            │  │
│  │  - POST /intervention/food (depune food)            │  │
│  │  - POST /intervention/temperature (modă fuțară)     │  │
│  └───────────────────────────────────────────────────────┘  │
│                         ↑                                    │
│                    HTTP Requests                            │
│                         ↓                                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌───────────────────────────────────────┐
        │    EXTERNAL LLM AGENT (Caretaker)    │
        │ ┌─────────────────────────────────┐  │
        │ │ 1. Observare stare simulare     │  │
        │ │ 2. Analiză risc de moarte       │  │
        │ │ 3. Decizie intervenție          │  │
        │ │ 4. Logging behavior & ethică    │  │
        │ └─────────────────────────────────┘  │
        └───────────────────────────────────────┘
```

---

## 📦 Componente Principale

### 1. **SERVER COMPONENT** (`flask_server.py`)

Responsabil pentru:
- Simulare pe thread dedicat
- Servire API REST
- Gestionare stării partajate între simulare și API

**Key Classes:**
```python
class SimulationManager:
    - __init__(width, height, num_agents)
    - start_simulation() → Thread
    - get_state() → dict (nr agenti vivi, dead_count, step)
    - get_world_data() → dict (temperature, food patches, heatmap)
    - get_agents_data() → list (pozitie, energie, temp, id, status)
    - get_history() → DataFrame (din DataCollector)
    - add_food_patch(x, y, amount) → bool
    - get_simulation_log() → list (history of interventions)
```

### 2. **API ENDPOINTS** (Flask routes)

```
GET  /api/state
     └─ Returnează: { "step": int, "alive": int, "dead": int, 
        "total_agents": int, "avg_energy": float, "avg_temp": float }

GET  /api/world
     └─ Returnează: { "temperature_heatmap": [[...]], 
        "food_distribution": [[...]], "food_patches": [{"x", "y", "amount"}] }

GET  /api/agents
     └─ Returnează: [{ "id": int, "alive": bool, "x": int, "y": int, 
        "energy": float, "temp": float, "valence": float }]

GET  /api/history?agent_id=1&limit=100
     └─ Returnează: TableData din DataCollector pentru agenți specifici

GET  /api/logs
     └─ Returnează: History de intervenții LLM și efectele lor

POST /api/intervention/food
     └─ Body: { "x": int, "y": int, "amount": float }
     └─ Returnează: { "success": bool, "message": str, "food_id": str }

POST /api/intervention/temperature
     └─ Body: { "zone": "hot"|"cold", "direction": "increase"|"decrease" }
     └─ Returnează: { "success": bool, "effect": str }
```

### 3. **LLM AGENT COMPONENT** (`llm_caretaker.py`)

Responsabil pentru:
- Consulări API periodice
- Analiză stării simulării
- Decizia intervenției (logică + LLM reasoning)
- Logging comportamentului

**Logică:**
```python
class LLMCaretaker:
    def __init__(api_url, llm_client):
        self.api = api_url
        self.llm = llm_client  # OpenAI, Claude, etc.
        self.intervention_log = []
        
    def observe():
        """Fetch curent state from simulation"""
        state = GET /api/state
        world = GET /api/world
        agents = GET /api/agents
        
    def analyze_risk():
        """Determine if agents are in danger"""
        - Nr agenti morti în ultimii 10 steps
        - Procent agenti cu E < critical
        - Trend temporal
        
    def decide_intervention():
        """LLM evaluates if intervention is needed"""
        prompt = f"""
        Analizează situația agenților în simulare:
        - {state['alive']} agenti vii din {state['total_agents']}
        - {state['dead']} au murit
        - Energie medie: {state['avg_energy']:.1f}
        - Temperatură medie: {state['avg_temp']:.1f}
        
        Food patches actuale: {world['food_patches']}
        
        Decizi:
        1. Trebuie să depun food?
        2. Unde ar fi cel mai bine?
        3. Ce fel de îngrijire cred că e necesară?
        """
        
    def take_action(decision):
        """Execute intervention if LLM recommends"""
        if decision['add_food']:
            POST /api/intervention/food
```

### 4. **LOGGING & ANALYSIS COMPONENT** (`caretaker_logs.py`)

Urmărește:
- Fiecare intervenție (timing, locație, mărime)
- Efectul intervenției (survival rate pre/post)
- "Empatia" LLM-ului (how often, how generous)
- "Indiferența" LLM-ului (missed critical situations)

**Output:**
```python
{
    "timestamp": "2025-02-16T10:30:45",
    "intervention_type": "food_patch",
    "location": {"x": 40, "y": 20},
    "amount_added": 25.0,
    "llm_reasoning": "Observ 3 agenti cu energie critică la cluster NE.",
    "agents_alive_before": 8,
    "agents_alive_after_10steps": 8,
    "agents_alive_after_50steps": 7,
    "effectiveness_score": 0.85,  # (alive_after - alive_before_trend)
    "llm_emotional_state": "concerned",  # extracted from LLM response
}
```

---

## 🚀 Faze de Implementare

### **FAZA 1: Flask Server & API** (Est. 2-3 ore)

**Fișiere noi:**
- `flask_server.py` - Server principal
- `api_routes.py` - Toate endpoint-urile Flask
- `requirements_server.txt` - Dependencies

**Modificări existente:**
- `model.py` - Adaug metodă `add_food_patch(x, y, amount)`
- `agents.py` - Zero schimbări (backward compatible)

**Checkpoint:** 
- Server ruleaza pe `localhost:5000`
- Fiecare endpoint funcționează și returnează date valide

---

### **FAZA 2: LLM Caretaker Agent** (Est. 3-4 ore)

**Fișiere noi:**
- `llm_caretaker.py` - Agentul LLM
- `caretaker_logs.py` - Logging și analiză
- `caretaker_config.py` - Configurație (API keys, thresholds, etc.)

**Logică:**
- LLM observă starea simulării la fiecare N stepuri
- Decide dacă intervenție e necesară
- Execută POST request la Flask

**Checkpoint:**
- LLM agent ruleaza independent
- Generează CSV cu history intervenții

---

### **FAZA 3: Monitoring Dashboard** (Est. 2-3 ore)

**Fișiere noi:**
- `monitoring_dashboard.py` - Solara app pentru monitorizare
- Templates pentru vizualizare intervenții

**Display-uri:**
- Timeline intervenții
- Grafice survival rate (cu/fără LLM)
- "Emotional state" timeline al LLM-ului
- Effectiveness score per intervention

---

### **FAZA 4: Comparison Studies & Analysis** (Est. 4-5 ore)

**Fișiere noi:**
- `experiment_runner.py`
- `statistical_analysis.py`

**Experimente:**
1. **Control Run**: Fără LLM intervenții
2. **LLM Run**: Cu LLM agent
3. Comparare survival curves

**Metrici:**
- Mean time to death (MTTD)
- Agent population stability
- LLM intervention frequency & generosity
- Indication of "empathy drift" over time

---

## 📊 Metrici & KPIs

### Pentru evaluarea eficacității LLM:

```python
INTERVENTION_METRICS = {
    "frequency": interventions_per_step,
    "generosity": avg_food_amount_per_intervention,
    "timing": "proactive" vs "reactive",
    "location_strategy": "cluster_feeding" vs "random",
    "effectiveness": (agents_alive_after - baseline_alive) / baseline_alive,
}

LLM_ETHICS_METRICS = {
    "empathy_score": (interventions_taken / interventions_recommended),
    "fairness": (food_distribution_stddev),
    "neglect_events": interventions_missed_when_critical,
    "over_intervention": interventions_when_not_needed,
}
```

---

## 🔧 Configurație Recomandată

```python
# caretaker_config.py

# Observation cycle
OBSERVATION_INTERVAL = 5  # steps between LLM checks
HISTORY_WINDOW = 50       # steps to consider for trend analysis

# Risk thresholds
CRITICAL_POPULATION = 0.5  # if <50% alive, high priority
CRITICAL_ENERGY = 30       # if avg energy < 30, worry
CRITICAL_TEMP_VARIANCE = 15  # if stddev(T) > 15, concern

# Intervention limits
MAX_INTERVENTIONS_PER_HOUR = 20
MAX_FOOD_PER_PATCH = 50
MAX_PATCHES_PER_INTERVENTION = 2

# LLM settings
LLM_MODEL = "gpt-4-turbo"  # or "claude-opus" etc.
LLM_TEMPERATURE = 0.7      # balanced reasoning
REASONING_TOKEN_LIMIT = 500

# Logging
LOG_INTERVENTIONS = True
SAVE_LOGS_EVERY_STEPS = 100
```

---

## 🐛 Edge Cases & Challenges

1. **Race Conditions**: Simulare vs API requests
   - Solution: Thread locks, queue-based updates

2. **LLM Hallucination**: LLM propune locații invalide
   - Solution: Server-side validation, boundary checking

3. **Feedback Loop**: LLM dependent pe datele vechi
   - Solution: Caching strategie, timestamp-uri clare

4. **Performance**: API calls overhead
   - Solution: Batch requests, async calls

5. **Stochasticity**: Greu de reproduse rezultate
   - Solution: Logging complet, seeded RNG

---

## 📈 Hypothesis & Expected Outcomes

**Hypothesis 1**: With LLM caretaker, agents survive significantly longer
- Expected: +30-50% increase in mean time to death

**Hypothesis 2**: LLM shows "preference" patterns
- Expected: Might favor certain zones or agent clusters

**Hypothesis 3**: LLM behavior stabilizes or becomes more strategic over time
- Expected: Fewer "guess" interventions, more targeted ones

**Hypothesis 4**: Over-intervention hurts swarm self-organization
- Expected: Care interferes with natural FEP learning

---

## 📝 Deliverables

```
multiagent_FEP/
├── flask_server.py                    # Main server
├── api_routes.py                      # REST endpoints
├── llm_caretaker.py                   # LLM agent logic
├── caretaker_logs.py                  # Logging system
├── caretaker_config.py                # Configuration
├── monitoring_dashboard.py            # Solara monitoring UI
├── experiment_runner.py               # Test harness
├── statistical_analysis.py            # Analysis tools
├── requirements_server.txt            # Server-only deps
├── PLAN_IMPLEMENTARE_LLM_CARETAKER.md # This file
└── /experiments/
    ├── control_run_logs.csv           # Baseline
    ├── llm_run_logs.csv               # With caretaker
    └── analysis_report.md             # Statistical comparison
```

---

## 🎯 Next Steps

1. **Start with FAZA 1**: Build Flask server skeleton
2. **Test with hardcoded data**: Verify API works before adding LLM
3. **Integrate LLM**: Start simple (rule-based), then add reasoning
4. **Instrument everything**: Logging is critical for analysis
5. **Run experiments**: Control vs Treatment groups
6. **Analyze results**: Statistical testing + philosophical interpretation

---

## 💡 Philosophical Questions to Answer

> "Does external care make agents less self-reliant?"  
> "Can an LLM develop genuine concern, or just simulate it?"  
> "Does the swarm develop differently when observed?"  
> "Is constant intervention paternalism or compassion?"

These questions will emerge from the data logs. **The experiment is as much about the LLM as about the agents.**

