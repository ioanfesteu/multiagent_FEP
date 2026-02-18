# Memorie Asociativă Termică în Agenți FEP
## Motivație Teoretică și Implementare

> **Proiect:** MINDWORM // Artificial Phenomenology — Experiment 02  
> **Referință teoretică principală:** Solms & Friston (2023), *How and Why Consciousness Arises*  
> **Actualizare:** Implementarea câmpului de memorie termică asociativă (`thermal_memory`)

---

## 1. Problema de pornire: agenți fără referințe spațiale

Agenții din simulare nu au acces la coordonate spațiale absolute. Nu știu *unde* se află pe hartă, nu pot stoca sau compara poziții GPS. Pentru a naviga eficient către sursele de hrană, ei trebuie să se bazeze pe indicii locale.

Inițial, am considerat folosirea temperaturii interne ($T_{int}$) ca un proxy pentru context. Totuși, din cauza inerției termice simulate (factorul $\eta$), $T_{int}$ are un decalaj față de mediu. Un agent care vine din frig și găsește hrană într-o zonă caldă ar putea avea încă o temperatură internă scăzută, ceea ce ar duce la o învățare eronată ("mâncarea e la frig").

**Soluția adoptată:** Agenții memorează **temperatura mediului** ($T_{env}$) în momentul hrănirii. Aceasta le permite să învețe corect corelația dintre temperatură și resurse, independent de starea lor fiziologică tranzitorie.

---

## 2. Ancorarea în ecuațiile Active Inference

### Ecuația 2 — Energia liberă și eroarea de predicție

$$F = \frac{1}{2}(e \cdot \omega \cdot e - \log(\omega)), \quad e = \varphi(M) - \psi(Q)$$

Memoria asociativă intervine la nivelul termenului $\psi(Q)$ — **predicția**. Un agent cu memorie termică formează predicții despre disponibilitatea hranei într-un anumit context termic.

Formal, estimarea valorii așteptate a aportului energetic ($I$) la o temperatură dată $T$ este:

$$\hat{I}(T) = \frac{\sum_k I_k \cdot \mathcal{K}(T, T_k)}{\sum_k \mathcal{K}(T, T_k) + \varepsilon}$$

Unde:
- $I_k$ este **aportul energetic** (intake) obținut la evenimentul $k$.
- $T_k$ este **temperatura mediului** înregistrată la evenimentul $k$.
- $\mathcal{K}$ este un kernel Gaussian de similaritate.

---

## 3. Arhitectura modulului de memorie

### 3.1 Structura de date

Memoria asociativă este o listă FIFO (First-In-First-Out) de trace-uri:

$$\mathcal{M} = \{(T_k, I_k)\}_{k=1}^{N_{max}}$$

| Câmp | Simbol | Descriere |
|------|--------|-----------|
| `T_k` | temperatura ambientală | Contextul termic extern la momentul hrănirii |
| `I_k` | intake (aport) | Cantitatea de energie absorbită |

### 3.2 Funcția de interogare — Kernel Density Estimation

Nu folosim un decay temporal explicit. Relevanța amintirilor este determinată pur de similaritatea termică. Dacă temperatura curentă (sau viitoare) este departe de orice temperatură la care s-a găsit hrană, valoarea estimată scade natural spre zero datorită formei Gaussiene a kernel-ului.

**Kernel Gaussian:** 
$$\mathcal{K}(T, T_k) = \exp\left(-\frac{(T - T_k)^2}{2\sigma_T^2}\right)$$

Parametrul $\sigma_T$ (`MEMORY_SIGMA_T`) controlează generalizarea. Un $\sigma_T$ mic înseamnă că agentul este foarte specific ("mănânc doar la exact 20°C"), în timp ce un $\sigma_T$ mare permite o generalizare mai largă ("mănânc cam pe unde e călduț").

### 3.3 Integrarea în Expected Free Energy G

Termenul de memorie se adaugă la suma de G, influențând decizia de mișcare:

$$G(a) = G_{pragmatic} + G_{epistemic} + G_{social} + G_{memory}$$

unde:

$$G_{memory}(a) = -\alpha \cdot \hat{I}(T_{env\_next}^{(a)})$$

- $T_{env\_next}^{(a)}$ este temperatura celulei unde agentul intenționează să se mute.
- $\alpha$ (`MEMORY_ALPHA`) este ponderea memoriei în decizie.
- Semnul minus indică faptul că o valoare estimată mare a hranei reduce $G$ (face acțiunea mai atractivă).

**Condiție de activare:** $G_{memory}$ este calculat doar când agentul este în stare de foame (`is_hungry`) și are amintiri relevante.

---

## 4. Vizualizare: Izobarele Verzi (Thermal Memory Field)

Pentru a înțelege "gândirea" agentului, vizualizăm câmpul $\hat{I}(T)$ sub formă de izobare verzi.

- **Ce reprezintă:** Aceste linii de contur arată zonele de pe hartă unde agentul *se așteaptă* să găsească hrană, bazat pe experiența sa.
- **Forma:** Deoarece temperatura variază continuu pe hartă, zonele cu $T \approx T_{memorata}$ formează benzi sau inele.
- **Interpretare:** Dacă un agent a găsit mâncare la 20°C, izobarele vor marca toate zonele cu 20°C de pe hartă. Agentul va tinde să navigheze de-a lungul acestor linii ("surfing termic"), căutând resurse în condiții familiare.

---

## 5. Rezumat

Prin trecerea de la memorarea $T_{int}$ la $T_{env}$, am eliminat confuzia cauzată de inerția termică. Agentul învață acum o corelație robustă între parametrii obiectivi ai mediului (temperatura) și disponibilitatea resurselor. Izobarele verzi oferă o fereastră directă în "mintea" agentului, arătându-ne ipotezele sale despre distribuția spațială a hranei.
