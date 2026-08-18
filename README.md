# KINVI-DIST

Framework de calcul distribué multi-GPU open source optimisé pour l'IA.
Plusieurs GPU forment un réseau qui travaille comme un seul GPU plus
puissant.

## Règle physique centrale

```
T_transfert(packet N+1) ≤ T_calcul(packet N)
```

Si cette condition tient → GPU idle time < 1 % → scaling parfait.
Chaque décision architecturale découle de cette règle : chaque GPU
calcule **et** reçoit en même temps, en permanence. Personne n'attend
personne.

**Objectif de scaling :** 2 GPU ≈ ×2 / 3 GPU ≈ ×3 / N GPU ≈ ×N.

## Architecture

- **CPU Scheduler** (thread léger, boucle 5 ms) : mesure T_calcul et
  T_transfert en temps réel, calcule la taille optimale des packets
  (`bandwidth × T_calcul × 0.85`, clampée 1 MB–512 MB), le lead time
  (`T_transfer + 2 ms`), le load balancing (`share ∝ 1/T_calcul`) et le
  work stealing. Ne fait aucun calcul IA, ne bloque jamais les GPU.
- **GPU 0 = MASTER** : 3 CUDA Streams (compute / send / receive).
- **GPU 1..N = WORKERS** : 3 CUDA Streams + **triple buffer**
  (A = calcul, B = réception DMA, C = prefetch prêt). Rotation cyclique :
  le GPU ne voit aucune interruption entre packets.

## Règles absolues

1. idle_time < 1 % sur chaque GPU — métrique principale du succès
2. Zéro transfert bloquant — tout passe par CUDA async + CUDA Events
3. Tous les tensors en transit = pinned memory obligatoire
4. Workers = processus séparés, pas des threads (GIL Python)
5. Le Scheduler mesure avant d'ajuster — jamais de valeurs hardcodées
6. Thread-safety partout : PacketPool, Profiler, Scheduler
7. 1 GPU détecté → mode local pur, overhead absolument zéro
8. Python 3.9+ / PyTorch 2.0+ / CUDA 11.8+

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from kinvi_dist import KinviDist
import torch

kd = KinviDist()
result = kd.run(tensor, lambda t: torch.matmul(t, w))
model  = kd.wrap(my_model)          # forward distribué

@kd.distribute
def heavy(tensor):
    return tensor @ w

with kd.session() as s:
    r1 = s.run(t1, op1)
    r2 = s.run(t2, op2)
```

## Tolerance aux pannes

- Timeout par packet : 30 s, retransmission auto (max 3 tentatives) ;
  après 3 échecs, le Master reprend le chunk.
- Heartbeat worker toutes les 500 ms ; silence > 2 s → worker déclaré
  mort, packets redistribués.

## Monitoring

```python
kd = KinviDist(monitor=True)   # dashboard CLI rafraîchi toutes les 200 ms
```

```
╔══════════════════════════════════════════════╗
║              KINVI-DIST  v1.0                ║
╠══════════════════════════════════════════════╣
║ GPU 0 MASTER  ████████░░  82%                ║
║  Compute : 14.2 ms/pkt │ BW : 13.8 GB/s      ║
║  Packet size : 196.0 MB (auto) │ Idle : 0.3% ║
╠══════════════════════════════════════════════╣
║ Packets total : 4821  │ Stolen : 23          ║
╚══════════════════════════════════════════════╝
```

## Benchmark

```bash
python examples/benchmark_scaling.py   # affiche le speedup réel mesuré
```

## Tests

```bash
pytest tests/
```

Les tests fonctionnent aussi sur machine sans GPU (mode CPU / local).

## Structure

```
kinvi-dist/
├── kinvi_dist/      # 10 modules + utils
├── examples/        # basic_matmul, model_inference, benchmark_scaling
├── tests/           # 7 suites de tests
├── setup.py
└── requirements.txt
```
