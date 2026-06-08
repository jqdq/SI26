# Kolokwium 1 — Grupa 2
### Scenariusz: Rok 2031. Wirus zombie DAK-0-W5K1 rozprzestrzenia się coraz szybciej

Służby medyczne i logistyczne potrzebują narzędzi analitycznych do zarządzania kryzysem.

> **Twoje rozwiązanie musi zawierać:**
> - przygotowanie danych: obsługę braków, kodowanie zmiennych kategorycznych, skalowanie cech (w co najmniej jednej metodzie)
> - model baseline oraz dwa modele autorskie, każdy z jednozdaniowym uzasadnieniem wyboru algorytmu
> - uzasadnienie wybranej liczby klastrów (na podstawie metody łokciową lub silhouette)
> - omówienie wyników wszystkich modeli, wskazanie najlepszego i wyjaśnienie przyczyn jego rezultatów

---

## Klastrowanie: Segmentacja obozów ocalałych

**Kontekst:**
Koordynujesz działania służb kryzysowych w sieci obozów ocalałych rozsianych po kraju.
Zasoby są ograniczone — nie można stosować tej samej strategii wszędzie.
Służby proszą cię o pogrupowanie obozów według ich charakterystyki,
aby móc opracować dedykowane plany działania dla każdego typu obozu.
Liczba grup nie jest z góry znana — musisz ją uzasadnić.

**Dane:**
Siedem cech opisujących każdy obóz: współrzędna geograficzna N (szerokość, stopnie),
współrzędna geograficzna E (długość, stopnie), liczba mieszkańców,
poziom zapasów żywności (dni), odległość od najbliższej strefy zakażeń (km),
poziom fortyfikacji (skala 0–10), wskaźnik zachorowań w ostatnim tygodniu (%).
Kolumna tekstowa `typ_terenu` (`"miejski"` / `"wiejski"`).
Wybrane kolumny zawierają braki danych. Część cech jest ze sobą skorelowana.

**Zadanie:**
Wytrenuj trzy pipeline'y klastrowania:
**KMeans** jako baseline oraz dwa modele autorskie spośród dostępnych algorytmów klastrowania.
Dla każdego modelu autorskiego napisz jednym zdaniem, dlaczego wybrałeś ten algorytm do tego problemu.
Liczbę klastrów wyznacz metodą łokciową lub współczynnikiem silhouette i uzasadnij wybór jednym zdaniem.
Oceń skuteczność modeli, wskaż najlepszy i uzasadnij decyzję.
Co Twoim zdaniem jest przyczyną wyników najlepszego modelu?

**Model baseline:** `KMeans(n_clusters=8, random_state=42)`

```python
import pandas as pd

df = pd.read_csv("dane_grupa2.csv")
df.head()
```
