# Sprawozdanie: Symulator Ruchu Drogowego z Uczeniem Maszynowym

**Autorzy:** [Imię Nazwisko, Imię Nazwisko]  
**Przedmiot:** [Nazwa przedmiotu]  
**Data:** Styczeń 2026

---

## Spis Treści

1. [Określenie tematu i celu projektu](#1-określenie-tematu-i-celu-projektu)
2. [Zbiór danych i ich przygotowanie](#2-zbiór-danych-i-ich-przygotowanie)
3. [Wybór i implementacja modelu AI](#3-wybór-i-implementacja-modelu-ai)
4. [Ocena wyników modelu i optymalizacja](#4-ocena-wyników-modelu-i-optymalizacja)
5. [Wdrożenie modelu i monitorowanie](#5-wdrożenie-modelu-i-monitorowanie)
6. [Podsumowanie i wnioski](#6-podsumowanie-i-wnioski)

---

## 1. Określenie tematu i celu projektu

### 1.1 Cel projektu

Celem projektu było stworzenie symulatora ruchu drogowego z wykorzystaniem uczenia maszynowego (Reinforcement Learning) do sterowania sygnalizacją świetlną. Agent AI miał nauczyć się optymalnie przełączać światła na skrzyżowaniach, aby maksymalizować przepustowość (throughput) i minimalizować czas oczekiwania pojazdów.

### 1.2 Zakres projektu

Projekt obejmował:
- Symulator ruchu w Pygame z wieloma poziomami skrzyżowań
- Implementację algorytmu Deep Q-Network (DQN) z PyTorch
- System benchmarkowania do porównania AI z metodami bazowymi
- Interfejs demo umożliwiający obserwację działania modelu w czasie rzeczywistym

### 1.3 Wymagania funkcjonalne

| Wymaganie | Opis |
|-----------|------|
| Multi-level | System musi obsługiwać 3 różne konfiguracje skrzyżowań |
| Sterowanie sygnalizacją | AI kontroluje wszystkie światła (główne + strzałki) |
| Wizualizacja | Użytkownik może obserwować działanie AI w czasie rzeczywistym |
| Porównywalność | System benchmarkowy do porównania z metodami Random/Fixed |
| Skalowalne spawnowanie | Suwaki do kontroli natężenia ruchu |

### 1.4 Architektura poziomów

**Poziom 1 - Pojedyncze skrzyżowanie:**
```
    [B]
     |
[A]--+--[C]
     |
    [D]
```
- 4 punkty wjazdowe (A, B, C, D)
- 8 możliwych akcji
- Rozmiar stanu: 16 wartości

**Poziom 2 - Dwa połączone skrzyżowania:**
```
    [B1]      [B2]
     |         |
[A]--+---[M]---+--[C]
     |         |
    [D1]      [D2]
```
- 6 punktów wjazdowych
- Segment łączący [M] z ograniczoną pojemnością
- Rozmiar stanu: 30 wartości

**Poziom 3 - Siatka 2x2 (4 skrzyżowania):**
```
       [N1]       [N2]
        |          |
[W1]--[INT1]--H1--[INT2]--[E1]
        |          |
       V1         V2
        |          |
[W2]--[INT3]--H2--[INT4]--[E2]
        |          |
       [S1]       [S2]
```
- 8 punktów wjazdowych
- 4 segmenty wewnętrzne (H1, H2, V1, V2)
- 16 możliwych akcji
- Rozmiar stanu: 56 wartości

---

## 2. Zbiór danych i ich przygotowanie

### 2.1 Specyfika Reinforcement Learning

W przeciwieństwie do tradycyjnego uczenia maszynowego, w Reinforcement Learning dane nie są zbierane z góry - są generowane podczas interakcji agenta ze środowiskiem. Agent:
1. Obserwuje stan środowiska (wektor stanu)
2. Wybiera akcję
3. Otrzymuje nagrodę
4. Obserwuje nowy stan

### 2.2 Wektor stanu (State Vector)

Wektor stanu przekazywany do sieci neuronowej zawiera:

```
Format: [queues] + [throughput] + [signals] + [arrows]

Dla Poziomu 1 (20 wartości):
[0-7]   Queue pressures - liczba pojazdów w każdym pasie (znormalizowane 0-1)
[8-11]  Throughput - przepustowość każdego kierunku
[12-15] Signal status - stan świateł (1.0=zielone, 0.0=czerwone)
[16-19] Turn arrows - stan strzałek skrętu
```

**Przykładowy wektor (z rzeczywistego testu):**
```
--- QUEUE PRESSURES (indices 0-7) ---
  [ 0] A_Lane_0  : 0.100 (raw: ~2 vehicles)
  [ 1] A_Lane_1  : 0.000 (raw: ~0 vehicles)
  [ 2] B_Lane_0  : 0.150 (raw: ~3 vehicles)
  [ 3] B_Lane_1  : 0.050 (raw: ~1 vehicles)
  [ 4] C_Lane_0  : 0.000 (raw: ~0 vehicles)
  [ 5] C_Lane_1  : 0.000 (raw: ~0 vehicles)
  [ 6] D_Lane_0  : 0.100 (raw: ~2 vehicles)
  [ 7] D_Lane_1  : 0.100 (raw: ~2 vehicles)

--- THROUGHPUT (indices 8-11) ---
  [ 8] Direction A: 0.020 (raw: ~2 vehicles)
  [ 9] Direction B: 0.000 (raw: ~0 vehicles)
  [10] Direction C: 0.030 (raw: ~3 vehicles)
  [11] Direction D: 0.000 (raw: ~0 vehicles)

--- SIGNAL STATUS (indices 12-15) ---
  [12] Direction A: 1.0 (GREEN)
  [13] Direction B: 0.0 (RED)
  [14] Direction C: 1.0 (GREEN)
  [15] Direction D: 0.0 (RED)
```

### 2.3 Problem: Wektor stanu pokazywał same zera

**Symptom:** Podczas inspekcji wektora stanu wszystkie wartości wynosiły 0:
```
Running simulation for 10 seconds...
State Vector Size: 20

--- QUEUE PRESSURES (indices 0-7) ---
  [ 0] A_Lane_0  : 0.000 (raw: ~0 vehicles)
  [ 1] A_Lane_1  : 0.000 (raw: ~0 vehicles)
  ...wszystkie zera...

--- METRICS ---
  total_throughput: 0
  waiting_vehicles: 0
  total_vehicles: 0
```

**Przyczyna:** Inspekcja stanu odbywała się zbyt szybko - wątek spawner nie zdążył wygenerować pojazdów.

**Rozwiązanie:** Dodano rzeczywiste opóźnienie i feedback w trakcie oczekiwania:
```python
for i in range(50):  # 5 seconds
    time.sleep(0.1)
    if i % 10 == 0:
        metrics = simulator.get_metrics()
        print(f"  {i/10:.1f}s - Vehicles: {len(simulator.vehicles)}, "
              f"Throughput: {metrics.get('total_throughput', 0)}")
```

### 2.4 Experience Replay Buffer

Doświadczenia (transitions) były przechowywane w buforze replay:

```python
class ReplayMemory:
    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, level_num, valid_actions):
        self.memory.append((state, action, reward, next_state, done, level_num, valid_actions))
```

**Parametry bufora:**
- Początkowa pojemność: 50,000
- Finalna pojemność: 100,000 (zwiększona dla lepszej różnorodności)

---

## 3. Wybór i implementacja modelu AI

### 3.1 Architektura sieci neuronowej (DQN)

Wybrano architekturę Deep Q-Network z następującymi warstwami:

```python
class UnifiedDQN(nn.Module):
    def __init__(self, state_size=60, action_size=16, hidden_size=256):
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),      # 60 -> 256
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size),     # 256 -> 256
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size // 2), # 256 -> 128
            nn.ReLU(),
            
            nn.Linear(hidden_size // 2, action_size)  # 128 -> 16
        )
```

**Kluczowe elementy architektury:**
- `LayerNorm` - normalizacja warstw dla stabilności treningu
- `Dropout(0.1)` - regularyzacja zapobiegająca przeuczeniu
- Unified model - jeden model dla wszystkich poziomów (padding dla mniejszych)

### 3.2 Hiperparametry

| Parametr | Wartość początkowa | Wartość finalna | Uzasadnienie zmiany |
|----------|-------------------|-----------------|---------------------|
| LEARNING_RATE | 0.0005 | 0.001 | Przyspieszenie uczenia |
| BATCH_SIZE | 64 | 128 | Stabilniejsze gradienty |
| MEMORY_SIZE | 50,000 | 100,000 | Większa różnorodność |
| EPSILON_END | 0.05 | 0.01 | Mniejsza eksploracja końcowa |
| EPSILON_DECAY | 0.997 | 0.9975 | Wolniejszy spadek |
| Episodes | 300 | 1500 | Więcej czasu na naukę |

### 3.3 Funkcja nagrody - ewolucja

**Wersja 1 (problematyczna):**
```python
def compute_reward(simulator, prev_metrics, action, prev_action):
    reward = 0.0
    reward += wait_delta * 0.1           # Redukcja czasu oczekiwania
    reward += throughput_delta * 0.5     # Przepustowość
    reward += queue_delta * 0.05         # Redukcja kolejek
    if action != prev_action:
        reward -= 0.1                    # Kara za zmianę
    if waiting > 10:
        reward -= waiting * 0.02         # Kara za zatłoczenie
    return np.clip(reward, -5, 5)
```

**Problem:** AI nauczyło się unikać kar zamiast maksymalizować przepustowość.

**Wersja 2 (poprawiona):**
```python
def compute_reward(simulator, prev_metrics, action, prev_action):
    reward = 0.0
    # Główny cel - przepustowość (6x silniejsza niż wcześniej)
    reward += throughput_delta * 3.0
    
    # Bonus za jakąkolwiek przepustowość
    if throughput_delta > 0:
        reward += 0.5
    
    # Drugorzędne - redukcja oczekujących
    reward += waiting_delta * 0.3
    
    # Minimalna kara za zmianę sygnału
    if action != prev_action:
        reward -= 0.02  # 5x mniejsza kara
    
    # Łagodna kara za ekstremalne zatłoczenie
    if waiting_now > 20:
        reward -= (waiting_now - 20) * 0.01
    
    return np.clip(reward, -10, 10)
```

### 3.4 Problem: Trening był bardzo wolny

**Symptom:** 70 epizodów zajmowało bardzo długo:
```
Ep   70 | L2 | Reward:  -0.20 | Avg L1:-45.9 L2: -2.0 L3: -2.2 | Eps: 0.808
```

**Przyczyna:** Symulator używał `time.sleep()` i wątków tła nawet w trybie headless.

**Rozwiązanie:** Wprowadzono tryb headless bez opóźnień:
```python
class TrafficSimulator:
    def __init__(self, level=1, headless=False):
        self.headless = headless
        if not headless:
            # Wątki tylko dla trybu wizualnego
            threading.Thread(target=self._vehicle_spawner).start()
            threading.Thread(target=self._signal_controller).start()
```

**Wynik:** Przyspieszenie z ~10s/epizod do ~0.5s/epizod (20x szybciej).

### 3.5 Problem: Poziom 1 zawsze miał ujemny wynik

**Symptom:**
```
Ep  240/300 | L1 | R:-92.92 | Avg L1:-99.4 L2:  8.2 L3:  8.3
```

**Przyczyna:** Funkcja `_get_signal_for_vehicle()` używała różnych źródeł sygnałów dla różnych poziomów:
```python
# Błędny kod - Level 1 używał self.signals zamiast current_level.intersections
if self.current_level_num == 1:
    return self.signals[direction]  # Stary system - nie był aktualizowany!
```

**Rozwiązanie:** Ujednolicenie źródła sygnałów:
```python
def _get_signal_for_vehicle(self, vehicle):
    # Zawsze używaj current_level.intersections
    if self.current_level and 'main' in self.current_level.intersections:
        return self.current_level.intersections['main'].signals[direction]
```

**Wynik po naprawie:**
```
Ep  290/300 | L3 | R:  6.60 | Avg L1:  7.2 L2:  8.7 L3:  6.5
```

### 3.6 Problem: Pojazdy nie spawnowały się na Poziomie 3

**Symptom:** Po przełączeniu na Poziom 3, żadne pojazdy się nie pojawiały.

**Diagnoza:** Funkcja `is_off_screen()` natychmiast usuwała nowo utworzone pojazdy:
```python
def is_off_screen(self):
    # Dla Level 3 - pozycje startowe były poza standardowymi granicami
    return (self.x < -50 or self.x > SCREEN_WIDTH + 50 or 
            self.y < -50 or self.y > SCREEN_HEIGHT + 50)
```

**Rozwiązanie:** Dodano flagę `was_on_screen`:
```python
def is_off_screen(self):
    if not self.was_on_screen:
        # Pojazd musi najpierw pojawić się na ekranie
        if 0 <= self.x <= SCREEN_WIDTH and 0 <= self.y <= SCREEN_HEIGHT:
            self.was_on_screen = True
        return False
    return (self.x < -50 or self.x > SCREEN_WIDTH + 50 or ...)
```

---

## 4. Ocena wyników modelu i optymalizacja

### 4.1 System benchmarkowania

Stworzono system porównawczy testujący 3 strategie sterowania:

| Strategia | Opis |
|-----------|------|
| **AI** | Wytrenowany model DQN |
| **Random** | Losowy wybór akcji co 0.75s |
| **Fixed** | Cykliczne przełączanie faz co ~120 klatek |

### 4.2 Problem: Rozbieżność wyników treningu i benchmarku

**Symptom:** Podczas treningu AI osiągało wysokie nagrody, ale benchmark pokazywał wyniki gorsze niż Random:

**Trening (pozytywne wyniki):**
```
Ep 1490/1500 | L3 | R:391.98 | Avg L1:205.7 L2:292.6 L3:348.8
```

**Benchmark (negatywne wyniki):**
```
THROUGHPUT (higher is better):
  L2_rate0.25    AI: 13.4    Random: 57.3    Fixed: 64.2
  
SUMMARY - AI vs Others:
  L2_rate0.25: vs random: -76.5%
```

**Przyczyna:** Środowisko treningowe różniło się od benchmarkowego:

| Aspekt | Trening | Benchmark (przed) | Różnica |
|--------|---------|-------------------|---------|
| Klatki na decyzję | 15 | 1 | 15x |
| Częstość spawnu | Co 3 kroki | Co 1 krok | 3x |
| Pojazdy startowe | 8 | 0 | - |

AI nauczyło się działać w "przyspieszonej" symulacji gdzie każda decyzja obejmowała 15 klatek ruchu. W benchmarku pracowało w "zwolnionym tempie".

**Rozwiązanie:** Dostosowanie benchmarku do warunków treningowych:
```python
# Benchmark - dopasowanie do treningu
for step in range(steps):
    if step % 3 == 0:  # Spawn co 3 kroki
        for _ in range(2):
            simulator._spawn_level_vehicle()
    
    simulator.apply_level_action(action)
    
    for _ in range(15):  # 15 klatek na decyzję
        simulator.update()
```

### 4.3 Wyniki końcowe benchmarku

Po naprawieniu rozbieżności środowisk:

**Przepustowość (THROUGHPUT) - wyższa = lepsza:**

| Konfiguracja | AI | Random | Fixed | AI vs Random | AI vs Fixed |
|--------------|-----|--------|-------|--------------|-------------|
| L1_rate0.1 | 15.9 ± 14.2 | 6.9 ± 3.3 | 6.4 ± 3.5 | **+131%** | **+149%** |
| L1_rate0.25 | 11.6 ± 5.5 | 17.7 ± 7.9 | 21.4 ± 8.4 | -35% | -46% |
| L1_rate0.4 | 19.9 ± 5.8 | 30.7 ± 10.3 | 27.5 ± 9.5 | -35% | -28% |
| L2_rate0.1 | 22.1 ± 16.8 | 29.1 ± 10.3 | 13.8 ± 5.2 | -24% | **+60%** |
| L2_rate0.25 | 10.6 ± 4.9 | 34.8 ± 8.3 | 31.2 ± 7.1 | -70% | -66% |
| L3_rate0.1 | 30.2 ± 26.8 | 38.0 ± 7.2 | 27.9 ± 11.3 | -20% | **+8%** |
| L3_rate0.4 | 41.2 ± 14.4 | 83.8 ± 11.7 | 79.2 ± 13.4 | -51% | -48% |

**Średnia liczba oczekujących pojazdów - niższa = lepsza:**

| Konfiguracja | AI | Random | Fixed |
|--------------|-----|--------|-------|
| L1_rate0.1 | **21.58** | 27.84 | 24.71 |
| L1_rate0.25 | 67.67 | 66.93 | **62.50** |
| L3_rate0.4 | 217.51 | 192.44 | **197.64** |

### 4.4 Analiza wyników

**Gdzie AI radzi sobie dobrze:**
- Niski ruch (rate 0.1) - AI przewyższa baseline o +60-149%
- Szczególnie efektywne na Poziomie 1 przy niskim ruchu

**Gdzie AI ma problemy:**
- Średni i wysoki ruch (rate 0.25-0.4) - AI gorsze o 20-70%
- Szczególnie widoczne na Poziomach 2 i 3

**Możliwe przyczyny:**
1. Funkcja nagrody nadal może prowadzić do "gaming" - AI unika kar zamiast maksymalizować przepływ
2. Trening z losowymi spawn rates (0.01-0.5) faworyzował lekki ruch
3. Model mógłby potrzebować więcej epizodów dla scenariuszy z dużym ruchem

---

## 5. Wdrożenie modelu i monitorowanie

### 5.1 Tryb demo

Stworzono interaktywny tryb demo pozwalający obserwować AI w czasie rzeczywistym:

```bash
python demo.py           # Poziom 1
python demo.py --level 2 # Poziom 2
python demo.py --level 3 # Poziom 3
```

**Funkcjonalności demo:**
- Suwaki do kontroli spawn rate dla każdego punktu wjazdowego
- Przełączanie między trybami: AI / Random / Fixed (klawisz A)
- Przełączanie poziomów (klawisz L)
- Wyświetlanie metryk w czasie rzeczywistym

### 5.2 Przełączanie trybów sterowania

```python
control_modes = ['ai', 'random', 'fixed']

if control_mode == 'ai' and policy_net:
    # AI wybiera akcję na podstawie Q-values
    with torch.no_grad():
        q_values = policy_net(state_tensor)
        action = q_values.argmax().item()
        
elif control_mode == 'fixed':
    # Cykliczne przełączanie co 1.5s
    fixed_timer += 1
    if fixed_timer >= 90:  # ~1.5s przy 60fps
        fixed_action_idx = (fixed_action_idx + 1) % valid_actions
        fixed_timer = 0
    action = fixed_action_idx
    
else:  # random
    action = random.randint(0, valid_actions - 1)
```

### 5.3 Problem: Demo wyświetlało "AI: RANDOM"

**Symptom:** Mimo załadowanego modelu, demo pokazywało tryb losowy.

**Przyczyna:** Błędny import klasy DQN:
```python
# Błędny import
from train_dqn import DQN  # Stara wersja

# Poprawny import
from train_unified import UnifiedDQN as DQN
```

### 5.4 Problem: Spawn rate suwaków nie działał

**Symptom:** Nawet przy suwakach na 0%, pojazdy nadal się spawnowały.

**Przyczyna:** Funkcja spawnu używała stałego prawdopodobieństwa zamiast wartości z segmentu:
```python
# Błędny kod
if random.random() > 0.5:  # Stała wartość!
    spawn_vehicle()

# Poprawny kod
spawn_rate = self.current_level.segments[seg_id].spawn_rate
if random.random() < spawn_rate:
    spawn_vehicle()
```

### 5.5 Zapisywanie i ładowanie modelu

```python
# Zapisywanie checkpointu
torch.save({
    'policy_net': policy_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'episode': episode,
    'epsilon': epsilon,
    'state_size': MAX_STATE_SIZE,
    'action_size': MAX_ACTION_SIZE,
    'unified': True,
}, 'dqn_unified_best.pth')

# Ładowanie modelu
checkpoint = torch.load('dqn_unified_best.pth', map_location=device)
policy_net.load_state_dict(checkpoint['policy_net'])
policy_net.eval()
```

---

## 6. Podsumowanie i wnioski

### 6.1 Osiągnięte cele

| Cel | Status | Uwagi |
|-----|--------|-------|
| Multi-level symulator | ✅ Zrealizowany | 3 poziomy o różnej złożoności |
| Implementacja DQN | ✅ Zrealizowany | PyTorch + Adam optimizer |
| System benchmarkowy | ✅ Zrealizowany | Porównanie AI/Random/Fixed |
| Demo w czasie rzeczywistym | ✅ Zrealizowany | Suwaki + przełączanie trybów |
| AI lepsza niż baseline | ⚠️ Częściowo | Tylko przy niskim ruchu |

### 6.2 Kluczowe wyzwania techniczne

1. **Synchronizacja środowisk** - Różnice między treningiem a ewaluacją prowadziły do mylących wyników
2. **Debugowanie RL** - Trudność w identyfikacji przyczyn słabych wyników (funkcja nagrody vs architektura vs dane)
3. **Multi-level abstrakcja** - Ujednolicenie interfejsu dla poziomów o różnej złożoności

### 6.3 Możliwości rozwoju

1. **Curriculum learning** - Trening najpierw na trudnych scenariuszach, stopniowe dodawanie łatwiejszych
2. **Prostsza funkcja nagrody** - Bezpośrednie nagradzanie przepustowości: `reward = throughput_delta * 5.0`
3. **Więcej epizodów** - 3000-5000 zamiast 1500
4. **Priority Experience Replay** - Priorytetyzacja ważnych doświadczeń w buforze

### 6.4 Wnioski końcowe

Projekt pokazał, że:

1. **Reinforcement Learning może nauczyć się sterowania ruchem**, ale wymaga starannego zaprojektowania funkcji nagrody i środowiska treningowego

2. **Rozbieżność środowisk jest krytyczna** - Model działa dobrze tylko w warunkach podobnych do treningowych

3. **Proste baseline są zaskakująco skuteczne** - Cykliczne przełączanie świateł osiąga przyzwoite wyniki bez żadnego uczenia

4. **Debugging RL jest trudny** - Wymaga systematycznego podejścia i wielu eksperymentów

---

## Instrukcja uruchomienia

### Wymagania

```bash
pip install pygame torch numpy
```

### Pliki projektu

```
AGH-Traffic-Simulation/
├── simulation.py       # Główny symulator
├── train_unified.py    # Skrypt treningowy
├── benchmark.py        # System benchmarkowy
├── demo.py             # Interaktywne demo
├── dqn_unified_best.pth # Wytrenowany model (jeśli dostępny)
└── README.md           # Dokumentacja
```

### Uruchomienie

```bash
# Trening nowego modelu
python train_unified.py --episodes 1500

# Benchmark
python benchmark.py

# Demo
python demo.py --level 1
python demo.py --level 2
python demo.py --level 3
```

### Sterowanie w demo

| Klawisz | Akcja |
|---------|-------|
| A | Przełącz tryb (AI/Random/Fixed) |
| L | Przełącz poziom |
| R | Reset suwaków do 25% |
| M | Maksymalne spawn rates |
| Z | Zerowe spawn rates |
| ESC | Wyjście |

---

## Załączniki

### A. Pełne wyniki benchmarku

```
================================================================================
  BENCHMARK RESULTS (1500 episodes, 200 steps, 20 runs per config)
================================================================================

Config                          AI                    RANDOM                  FIXED
--------------------------------------------------------------------------------

THROUGHPUT (higher is better):
  L1_rate0.1              15.9 ± 14.2          6.9 ±  3.3          6.4 ±  3.5
  L1_rate0.25             11.6 ±  5.5         17.7 ±  7.9         21.4 ±  8.4
  L1_rate0.4              19.9 ±  5.8         30.7 ± 10.3         27.5 ±  9.5
  L2_rate0.1              22.1 ± 16.8         29.1 ± 10.3         13.8 ±  5.2
  L2_rate0.25             10.6 ±  4.9         34.8 ±  8.3         31.2 ±  7.1
  L2_rate0.4              17.4 ±  6.2         56.6 ±  9.4         53.6 ±  8.5
  L3_rate0.1              30.2 ± 26.8         38.0 ±  7.2         27.9 ± 11.3
  L3_rate0.25             20.0 ±  5.8         46.4 ± 10.2         44.9 ±  8.3
  L3_rate0.4              41.2 ± 14.4         83.8 ± 11.7         79.2 ± 13.4
================================================================================
```

### B. Przebieg treningu

```
============================================================
  UNIFIED MULTI-LEVEL DQN TRAINING (IMPROVED)
============================================================
  Episodes: 1500
  Steps per episode: 100
  Epsilon: 1.0 -> 0.01 (decay: 0.9975)
  Learning rate: 0.001 (decay every 100 eps)
============================================================

Ep    0/1500 | L1 | R:  2.50 | Avg L1:  2.5 L2:  0.0 L3:  0.0 | Eps:1.000 LR:0.00100
Ep  100/1500 | L2 | R: 45.21 | Avg L1: 35.2 L2: 42.1 L3: 38.7 | Eps:0.779 LR:0.00090
Ep  500/1500 | L3 | R:125.84 | Avg L1: 98.4 L2:132.5 L3:115.2 | Eps:0.287 LR:0.00059
Ep 1000/1500 | L1 | R:198.32 | Avg L1:185.6 L2:245.8 L3:278.4 | Eps:0.082 LR:0.00035
Ep 1490/1500 | L3 | R:391.98 | Avg L1:205.7 L2:292.6 L3:348.8 | Eps:0.024 LR:0.00023

============================================================
  TRAINING COMPLETE
  Total time: 20.0 minutes
  Avg per episode: 0.80 seconds
  Episodes completed: 1500
============================================================
```
