# Kolokwium 1 — Grupa 1
### Scenariusz: Rok 2031. Wirus zombie DAK-0-W5K1 rozprzestrzenia się coraz szybciej

Służby medyczne i logistyczne potrzebują narzędzi analitycznych do zarządzania kryzysem.

> **Twoje rozwiązanie musi zawierać:**
> - przygotowanie danych: obsługę braków, kodowanie zmiennych kategorycznych, podział na zbiór treningowy i testowy
> - pipeline baseline oraz dwa pipeline'y autorskie, każdy z jednozdaniowym uzasadnieniem wyboru algorytmu
> - wybraną metrykę oceny wraz z jednozdaniowym uzasadnieniem jej doboru
> - omówienie wyników wszystkich pipeline'ów, wskazanie najlepszego i wyjaśnienie przyczyn jego rezultatów

Gotowy notebook, zapisz, pobierz na komputer i prześlij tutaj:
https://forms.office.com/e/a8TnNyvgXG

---

## Klasyfikacja: Triage zakażonych

**Kontekst:**
Pracujesz dla mobilnego szpitala polowego. Przy bramach obozu przeprowadzany jest
szybki pomiar siedmiu parametrów fizjologicznych każdego przybysza.
Twój model ma wspomagać lekarzy w kierowaniu pacjentów na dalsze badania —
każda osoba zakażona, która przejdzie przez bramę jako zdrowa, zagraża całemu obozowi.
Zakażeni stanowią znaczną część przybyszy.

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

df = pd.read_csv("dane_grupa1.csv")
df.head()
```
