# Kolokwium 1 — Grupa 4
### Scenariusz: Rok 2031. Wirus zombie DAK-0-W5K1 rozprzestrzenia się coraz szybciej

Służby medyczne i logistyczne potrzebują narzędzi analitycznych do zarządzania kryzysem.

> **Twoje rozwiązanie musi zawierać:**
> - przygotowanie danych: obsługę braków, kodowanie zmiennych kategorycznych, podział na zbiór treningowy i testowy
> - pipeline baseline oraz dwa pipeline'y autorskie, każdy z jednozdaniowym uzasadnieniem wyboru algorytmu
> - wybraną metrykę oceny wraz z jednozdaniowym uzasadnieniem jej doboru
> - omówienie wyników wszystkich pipeline'ów, wskazanie najlepszego i wyjaśnienie przyczyn jego rezultatów

Rozwiązanie w formie pliku Jupyter Notebook (`.ipynb`) prześlij na maila.
W trakcie kolokwium możesz korzystać tylko z materiałów zgromadzonych w swojej instancji JupyterHub.

---

## Klasyfikacja: Certyfikaty bezpieczeństwa

**Kontekst:**
Pracujesz dla komitetu ds. bezpieczeństwa stref zielonych.
Osoby wchodzące do strefy chronionej muszą otrzymać certyfikat zdrowia.
Certyfikat wydaje się tylko wtedy, gdy model jest pewny, że dana osoba jest zdrowa —
fałszywy certyfikat dla osoby zakażonej oznacza katastrofę dla całej strefy.
Zakażeni stanowią znaczną część badanych.

**Dane:**
Siedem cech fizjologicznych: temperatura ciała (°C), tętno (bpm), poziom kortyzolu (µg/dL),
ciśnienie krwi skurczowe (mmHg), saturacja (%), czas reakcji (ms), spójność mowy (skala 0–10).
Kolumna tekstowa `strefa_pochodzenia` (`"zielona"` / `"czerwona"`).
Wybrane kolumny zawierają braki danych. Część cech jest ze sobą skorelowana.

**Zadanie:**
Wytrenuj trzy pipeline'y klasyfikacji binarnej (zakażony / zdrowy):
**Gaussian Naive Bayes** jako baseline oraz dwa modele autorskie spośród dostępnych klasyfikatorów.
Dla każdego modelu autorskiego napisz jednym zdaniem, dlaczego wybrałeś ten algorytm do tego problemu.
Przed porównaniem modeli wybierz jedną metrykę i uzasadnij wybór jednym zdaniem.
Oceń skuteczność modeli, wskaż najlepszy i uzasadnij decyzję.
Co Twoim zdaniem jest przyczyną wyników najlepszego modelu?

**Model baseline:** `GaussianNB()`

```python
import pandas as pd

df = pd.read_csv("dane_grupa4.csv")
df.head()
```
