# Kolokwium 1 — Grupa 3
### Scenariusz: Rok 2031. Wirus zombie DAK-0-W5K1 rozprzestrzenia się coraz szybciej.

Służby medyczne i logistyczne potrzebują narzędzi analitycznych do zarządzania kryzysem.

> **Twoje rozwiązanie musi zawierać:**
> - przygotowanie danych: obsługę braków, kodowanie zmiennych kategorycznych, skalowanie cech (w co najmniej jednej metodzie)
> - model baseline oraz dwa modele autorskie, każdy z jednozdaniowym uzasadnieniem wyboru algorytmu
> - uzasadnienie wybranej liczby klastrów (na podstawie metody łokciową lub silhouette)
> - omówienie wyników wszystkich modeli, wskazanie najlepszego i wyjaśnienie przyczyn jego rezultatów

Rozwiązanie w formie pliku Jupyter Notebook (`.ipynb`) prześlij na maila.
W trakcie kolokwium możesz korzystać tylko z materiałów zgromadzonych w swojej instancji JupyterHub.

---

## Klastrowanie: Taksonomia szczepów wirusa

**Kontekst:**
Pracujesz w laboratorium wirusologicznym. Próbki pobrane od zakażonych wykazują
zróżnicowane właściwości biologiczne — naukowcy podejrzewają, że wirus DAK-0-W5K1 zmutował
w kilka odrębnych szczepów. Zrozumienie tej struktury jest kluczowe dla opracowania leków.
Dane są zaszumione — pobieranie próbek w warunkach polowych wprowadza błędy pomiarowe,
a granice między szczepami są rozmyte.

**Dane:**
Siedem cech biologicznych próbki: tempo replikacji (kopie/h), powinowactwo do receptora ACE-3
(skala 0–10), ładunek wirusowy (log skala), odporność na temperaturę (°C),
aktywność proteazy (jednostki/ml), długość sekwencji RNA (kb), wskaźnik mutacji (%).
Kolumna tekstowa `zrodlo_probki` (`"laboratorium_A"` / `"laboratorium_B"`).
Wybrane kolumny zawierają braki danych. Część cech jest ze sobą skorelowana.

**Zadanie:**
Wytrenuj trzy pipeline'y klastrowania:
**KMeans** jako baseline oraz dwa modele autorskie spośród dostępnych algorytmów klastrowania.
Dla każdego pipeline'u autorskiego napisz jednym zdaniem, dlaczego wybrałeś ten algorytm do tego problemu.
Oceń z pomocą metryk skuteczność modeli, wskaż, które klastrowanie jest najbardziej użyteczne (w tym ile klastrów można wyróżnić) i uzasadnij wybór jednym zdaniem.
Co Twoim zdaniem jest przyczyną wyników najlepszego modelu?

**Model baseline:** `KMeans(n_clusters=3, random_state=7)`

```python
import pandas as pd

df = pd.read_csv("dane_grupa3.csv")
df.head()
```
