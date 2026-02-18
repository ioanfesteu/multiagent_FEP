# Memorie Asociativă Termică în Agenți FEP
## Motivație Teoretică și Implementare

> **Proiect:** MINDWORM // Artificial Phenomenology — Experiment 02  
> **Referință teoretică principală:** Solms & Friston (2023), *How and Why Consciousness Arises*  
> **Modificare:** Adăugarea modulului `thermal_memory` în `agents.py`

---

## 1. Problema de pornire: agenți fără referințe spațiale

Agenții din simulare nu au acces la coordonate spațiale absolute. Nu știu *unde* se află pe hartă, nu pot stoca sau compara poziții GPS. Singurul context intern stabil pe care îl au la dispoziție este **temperatura internă** (`T_int`) — o variabilă fiziologică care se modifică lent, prin conductivitate termică cu mediul:

```
T_int(t+1) = T_int(t) + eta * (T_env - T_int(t))
```

Aceasta înseamnă că `T_int` **codifică implicit mediul termic traversat**: un agent care a petrecut timp în zone reci va fi mai rece decât unul care s-a aflat în zone calde. Temperatura internă devine astfel un **proxy interoceptiv al contextului ambiental**.

Întrebarea care motivează această modificare este: *poate un agent să formeze preferințe de navigare bazate exclusiv pe ce a simțit termic atunci când a găsit hrană?*

---

## 2. Ancorarea în ecuațiile lui Solms & Friston

### Ecuația 2 — Energia liberă și eroarea de predicție

$$F = \frac{1}{2}(e \cdot \omega \cdot e - \log(\omega)), \quad e = \varphi(M) - \psi(Q)$$

Această ecuație definește energia liberă ca o tensiune între eroarea de predicție (`e`) și precizia (`ω`) cu care sistemul îi acordă importanță. Memoria asociativă intervine la nivelul termenului `ψ(Q)` — **predicția**: un agent cu memorie termică formează predicții despre valoarea unui context termic viitor, nu numai despre starea senzorială imediată.

Formal, `ψ(Q)` în modulul de memorie devine:

$$\psi_{mem}(T_{int}) = \hat{V}(T_{int}) = \frac{\sum_k r_k \cdot w_k \cdot \mathcal{K}(T_{int}, T_k)}{\sum_k w_k \cdot \mathcal{K}(T_{int}, T_k) + \varepsilon}$$

Aceasta este estimarea neparametrică a valorii așteptate a hranei la o stare termică internă dată — o **hartă de valoare termică** construită din experiență.

### Ecuația 3 — Energia liberă așteptată și precizia

$$E[F] \approx H[P(\varphi)] = -\frac{1}{2}\log(\omega)$$

Ecuația 3 arată că energia liberă așteptată scade proporțional cu log-ul preciziei. Cu cât precizia (`ω`) este mai mare, cu atât incertitudinea (și, implicit, suferința existențială a sistemului) este mai mică.

Modulul de memorie introduce o **precizie specifică memoriei**, `ω_M`, care modulează lărgimea kernel-ului gaussian (`σ_T`):

$$\omega_M = \omega_0 \cdot \sigma(\gamma \cdot \beta_t)$$

unde `β_t` este precizia globală a agentului (derivată din valența afectivă). Efectul: când agentul este în stare bună (valență pozitivă), `ω_M` crește → `σ_T` scade → kernel mai îngust → **generalizare termică mai strictă, memorie mai precisă**.

---

## 3. Arhitectura modulului de memorie

### 3.1 Structura de date

Memoria asociativă `M_assoc` este o listă de trace-uri:

$$\mathcal{M} = \{(T_k, r_k, \tau_k)\}_{k=1}^{N}$$

| Câmp | Simbol | Descriere |
|------|--------|-----------|
| `T_k` | temperatura internă | Contextul termic la momentul mâncatului |
| `r_k` | recompensa | Cantitatea de energie absorbită |
| `τ_k` | pasul simulării | Momentul evenimentului (pentru decay temporal) |

### 3.2 Funcția de interogare — kernel density estimator

```
V_hat(T_query) = Σ_k [ r_k * w_k * K(T_query, T_k) ] / (Σ_k [ w_k * K(T_query, T_k) ] + ε)
```

cu:
- **Kernel gaussian:** `K(T, T_k) = exp(-(T - T_k)² / (2σ_T²))`
- **Decay temporal:** `w_k = (1 + age_k)^(-λ)`
- **Parametri:** `σ_T` (lărgime), `λ = 0.015` (rata de uitare)

Proprietăți emergente ale acestei formulări:
- Un agent care a mâncat la `T_int = 18°C` va fi atras și de stări termice de `16°C` sau `20°C`, cu intensitate proporțională cu kernel-ul
- Trăsăturile vechi devin treptat irelevante prin decay-ul `(1 + age)^(-λ)`
- Lipsa memoriei produce `V_hat = 0` — **prior plat, fără bias** (comportament virgin corect)

### 3.3 Politica de evicțiune (capacitate limitată N_max = 20)

Când memoria este plină, se elimină trace-ul cu cea mai mică valoare decăzută:

$$\text{evict} = \arg\min_k \left( r_k \cdot (1 + \text{age}_k)^{-\lambda} \right)$$

Aceasta păstrează trace-urile recente și cele cu recompensă mare — echivalentul biologic al **consolidării preferențiale a amintirilor semnificative**.

### 3.4 Modularea preciziei kernel-ului prin affect

$$\sigma_T(t) = \sigma_{max} - (\sigma_{max} - \sigma_{min}) \cdot \sigma(\gamma \cdot \text{valence\_integrated})$$

| Stare afectivă | valence_integrated | σ_T | Efect |
|---|---|---|---|
| Pozitivă (stare bună) | > 0 | → σ_min = 1.0°C | Memorie precisă, generalizare strictă |
| Neutră | ≈ 0 | ≈ σ_default = 3.0°C | Comportament standard |
| Negativă (stres, foame) | < 0 | → σ_max = 5.0°C | Memorie difuză, generalizare largă |

Rațiunea biologică: sub stres, precizia de codificare a memoriei scade — un fenomen documentat la nivel de consolidare hipocampală (cortizol ridicat → encodare mai puțin discriminativă).

---

## 4. Integrarea în Expected Free Energy G

Termenul de memorie se adaugă la suma de G:

$$G(a) = G_{pragmatic} + G_{epistemic} + G_{social} + G_{memory}$$

unde:

$$G_{memory}(a) = -\alpha \cdot \hat{V}(T_{pred}^{(a)})$$

- `T_pred` este temperatura internă **proiectată** dacă agentul ia acțiunea `a`: `T_pred = T_int + η * (T_env_next - T_int)`
- `α = 0.8` este ponderea relativă a memoriei în decizie
- Semnul negativ: G este minimizat, deci valoare mare → acțiune mai atractivă

**Condiție de activare:** `G_memory` este calculat doar când agentul este `is_hungry` și are cel puțin un trace în memorie. Aceasta previne ca memoria să suprascrie homeostaza unui agent sătul sau să blocheze explorarea în faza inițială (memorie goală = prior plat).

---

## 5. Comportamente emergente așteptate

### 5.1 Preferință termică derivată din experiență

Fără niciun cod explicit de navigare termică, agenții vor dezvolta **preferințe contextuale** — vor tinde spre zone termice în care au găsit anterior hrană. Aceasta nu este o regulă hard-coded, ci o consecință a `V_hat` crescut în acel domeniu termic.

### 5.2 Tensiunea memorie vs. curiozitate

`G_epistemic` trage agentul spre zone neexplorate (noi). `G_memory` trage spre zone termic familiare (sigure). Tensiunea dintre cele două va produce un comportament de foraging mai realist: agenții tineri (fără memorie) explorează agresiv; agenții cu experiență balansează explorarea cu exploatarea contextelor termice cunoscute.

### 5.3 Uitare adaptativă și re-explorare

Pe măsură ce trace-urile vechi se diminuează prin `(1 + age)^(-λ)`, un agent care nu mai găsește hrană într-un context termic familiar va vedea scăzând `V_hat` pentru acea regiune → memoria eliberează treptat acel prior → agentul revine la explorare. Aceasta implementează o formă de **flexibilitate cognitivă** prin uitare.

### 5.4 Afectul modulează calitatea memoriei

Un agent care găsește hrană imediat după o perioadă de stres (valență negativă, σ_T mare) va forma o memorie **difuză** — va generaliza larg de la acel context termic. Un agent în stare bună va forma o memorie **precisă** — mai discriminativă, mai greu de activat de stări termice diferite. Aceasta introduce **heterogeneitate individuală** emergentă în stilurile de foraging.

---

## 6. Parametrii noi și ghid de tunare

| Parametru | Default | Efect al creșterii |
|---|---|---|
| `MEMORY_MAX_TRACES` | 20 | Memorie mai lungă, cost computațional mai mare |
| `MEMORY_SIGMA_T` | 3.0°C | Kernel mai larg → generalizare mai mare |
| `MEMORY_LAMBDA_DECAY` | 0.015 | Uitare mai lentă → memorie mai persistentă |
| `MEMORY_ALPHA` | 0.8 | Memorie mai dominantă față de celelalte G-uri |
| `MEMORY_SIGMA_MIN` | 1.0°C | Precizie maximă sub affect pozitiv |
| `MEMORY_SIGMA_MAX` | 5.0°C | Difuziune maximă sub stres |
| `MEMORY_GAMMA` | 0.5 | Sensibilitate mai mare a σ_T la affect |

**Recomandare pentru experimente:**
- Setați `MEMORY_ALPHA = 0.0` pentru a dezactiva complet memoria și a compara cu baseline-ul original
- Setați `MEMORY_SIGMA_MIN = MEMORY_SIGMA_MAX` pentru a dezactiva modularea afectivă a preciziei
- Reduceți `MEMORY_LAMBDA_DECAY` spre 0 pentru memorie aproape permanentă (testarea impactului uitării)

---

## 7. Limitări și direcții viitoare

**Limitare 1 — Proxy imperfect:** `T_int` este un proxy al contextului termic, nu al contextului precis. Doi agenți pot ajunge la aceeași `T_int` prin rute termice diferite. Memoria termică nu distinge aceste cazuri.

**Limitare 2 — Absența memoriei sociale:** Trace-urile termice sunt private. O extensie naturală ar fi partajarea contextului termic prin semnalele de feromoni (`food_scent`) — un agent ar putea emite și temperatura sa internă la momentul mâncării, permițând altor agenți să-și actualizeze memoria din experiențele altora.

**Limitare 3 — Linearizare:** Relația dintre valență și `σ_T` este o aproximare sigmoidală. O implementare mai fidelă față de Ecuația 3 ar folosi `σ_T ∝ exp(-1/2 * log(ω_M))`, dar aceasta introduce instabilitate numerică la valorile extreme de affect.

**Direcție viitoare — Memorie episodică minimală:** Combinarea `T_int` cu `E_int` la momentul evenimentului ar forma un **vector de stare internă** bidimensional — o reprezentare mai bogată a contextului homeostat în care a apărut recompensa.

---

## 8. Rezumat

Modulul de memorie asociativă termică implementează o formă de **memorie interoceptivă contextuală** compatibilă cu framework-ul Free Energy Principle. Fără a utiliza coordonate spațiale, agenții pot forma preferințe de navigare bazate pe contextul termic al experiențelor trecute de succes. Precizia acestei memorii este modulată dinamic de starea afectivă curentă, cuplând sistemul mnemonic cu infrastructura FEP existentă (valență, precizie, inferență activă).

Aceasta extinde simularea dinspre **homeostaza reactivă** (răspuns la starea prezentă) spre **homeostazie predictivă** (navigare ghidată de experiența trecută) — un pas spre formele mai complexe de adaptare comportamentală descrise de Solms în conceptul de *conștiință afectivă*.
