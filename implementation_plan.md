# Implementation Plan: Epistemic Utility through Associative Memory

## Context
În modelul original, agenții explorau mediul pentru a evita repetiția (epistemic value negativ), dar fără o direcție clară legată de supraviețuire. Acest plan implementează o formă de "conștiință senzorială" primară, unde explorarea devine utilă prin corelarea temperaturii cu prezența hranei.

## Obiective
1. Eliminarea nevoii de coordonate spațiale (X, Y) în memoria agentului.
2. Transformarea "curiozității" într-un mecanism de reducere a incertitudinii față de resurse.
3. Implementarea unei memorii bazate pe stări interne (afect/temperatură).

## Pași de Implementare

### 1. Definirea Memoriei Asociative (`agents.py`)
- S-a adăugat variabila `self.T_food_memory` în constructorul agentului.
- Aceasta stochează "semătura termică" a zonelor cu hrană.
- S-a adăugat un flag `self.has_learned_food_temp` pentru a preveni ghidarea eronată înainte de prima masă.

### 2. Mecanismul de Învățare (Learning)
- În `update_internal_state`, atunci când agentul consumă o cantitate semnificativă de hrană (`intake > 1.0`), acesta își actualizează memoria.
- Se folosește o medie mobilă (`LEARNING_RATE_MEMORY`) pentru a permite adaptarea dacă mediul se schimbă (ex: sursele de hrană se mută în zone mai reci).

### 3. Integrarea în Inferența Activă (`choose_action`)
- S-a introdus `G_associative` în calculul Energiei Libere Expected ($G$).
- **Logica:** Când agentului îi este foame (`is_hungry`), acesta evaluează celulele vecine nu doar prin prisma confortului termic actual, ci și prin prisma potrivirii cu memoria hranei.
- **Funcția de utilitate:** O curbă Gaussiană care atinge maximul atunci când temperatura vecinătății este egală cu temperatura din memorie.

### 4. Parametrii de Control
- `WEIGHT_ASSOCIATIVE`: Controlează cât de mult "disperarea" foamei forțează agentul să urmeze amintirea termică.
- `SIGMA_ASSOC`: Definește cât de selectiv este agentul (o valoare mică îl face să caute strict acea temperatură, o valoare mare îi permite o explorare mai largă în jurul valorii reținute).

## Rezultat Teoretic
Explorarea nu mai este o rătăcire haotică. Agenții vor învăța, de exemplu, că "hrana se găsește de obicei în zonele de 30 grade". Chiar dacă nu văd hrana și nu simt mirosul lăsat de alții, ei vor naviga spre zonele care "se simt corect", optimizând astfel supraviețuirea grupului.