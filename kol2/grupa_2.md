# Kolokwium — Sieci Neuronowe
## Runway Magazine, 2026

Print umiera, a konkurencja stawia na algorytmy. *Runway* — kiedyś
najważniejszy magazyn mody świata — dostał ultimatum: albo zacznie podejmować
trafniejsze decyzje niż cyfrowa konkurencja, albo traci budżet reklamowy.
**Miranda Priestly**, redaktor naczelna, oczekuje wyników.

Jesteś internem data science w *Runway*. Masz 90 minut, żeby udowodnić,
że redakcja wciąż wie, co robi.

> **Twoje rozwiązanie musi zawierać:**
> - przygotowanie danych: obsługę braków, kodowanie zmiennych kategorycznych, skalowanie cech)
> - model baseline zbudowany według opisu
> - dwa modele z early stoppingiem oraz wybranym mechanizmem zapobiegającym overfittingowi (regularyzacja, dropout)
> - uzasadnieniem wyboru architektury
> - wybraną metrykę oceny wraz z jednozdaniowym uzasadnieniem jej doboru
> - omówienie wyników wszystkich pipeline'ów, wskazanie najlepszego i wyjaśnienie przyczyn rezultatów jednego z nich.

**Kontekst:**
Redakcja decyduje, które produkty z e-commerce *Runway* trafią do drukowanego
katalogu sezonu. Katalog ma ograniczoną liczbę stron — a każda strona to
opłacona powierzchnia reklamowa. Produkt, który trafi do katalogu, a okaże się
słaby sprzedażowo, to zmarnowana opłacona strona i argument przeciw drukowi.

**Dane:**
Dane e-commerce *Runway* — 2 000 produktów, 13 cech liczbowych i 2 kategoryczne.
Cechy: cena, marża, rabat, koszt, odsłony strony, click rate, conversion rate,
wishlist adds, średnia ocena, liczba recenzji, wskaźnik zwrotów,
dni w magazynie, liczba dostawień. Dział i sezon jako zmienne kategoryczne.
Wybrane kolumny zawierają braki danych. Cechy są ze sobą skorelowane.
Target: `w_katalogu` (1 = produkt trafia do katalogu, 0 = odpada).

**Model baseline:**
Model ma składać się z dwóch warstw gęstych: pierwszej z 64 neuronami, drugiej z 32 neuronami oraz warstwy wyjściowej z 1 neuronem. Trenowanie powinno trwać 30 epok.

> Każda strona katalogu to opłacona powierzchnia —
> zastanów się, który rodzaj błędu marnuje budżet reklamowy
> i jak to przekłada się na wybór metryki.

**Wczytanie danych:**
```python
import pandas as pd

df = pd.read_csv("data_group_2.csv")
print(df.shape)   # (2000, 16)
print(df.head())
```

*That's all.*
