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
> - dwa modele z early stoppingiem różniące się w budowie sieci czymś poza regularyzacją
> - w jednym wybrany mechanizmem zapobiegającym overfittingowi (regularyzacja, dropout)
> - uzasadnienie wyboru architektury
> - wybraną metrykę oceny wraz z jednozdaniowym uzasadnieniem jej doboru
> - omówienie wyników wszystkich pipeline'ów, wskazanie najlepszego i wyjaśnienie przyczyn rezultatów jednego z nich.

**Kontekst:**
Redakcja przygotowuje nowy numer *Runway* i musi przypisać każdy produkt do jednej
z trzech ścieżek wydawniczych: odrzucenia, katalogu standardowego lub katalogu prestiżowego.
Dział sprzedaży dysponuje danymi e-commerce z poprzednich sezonów.
Trafna klasyfikacja przekłada się bezpośrednio na przychód z reklam.

**Dane:**
Dane e-commerce *Runway* — 2 000 produktów, 13 cech liczbowych i 2 kategoryczne.
Cechy: cena, marża, rabat, koszt, odsłony strony, click rate, conversion rate,
wishlist adds, średnia ocena, liczba recenzji, wskaźnik zwrotów,
dni w magazynie, liczba dostarczeń. Kolekcja oraz linia marki jako zmienne kategoryczne
(wartości: `nowa` / `archiwalna` oraz `premium` / `masowa`).
Wybrane kolumny zawierają braki danych. Cechy są ze sobą skorelowane.
Target: `kategoria` (`odrzucony`, `standardowy`, `prestiżowy`).

**Model baseline:**
Model ma składać się z dwóch warstw gęstych: pierwszej z 64 neuronami, drugiej z 32 neuronami oraz warstwy wyjściowej. Trenowanie powinno trwać 30 epok.

> Błędy klasyfikacji nie są równoważne — zastanów się, który rodzaj pomyłki
> jest najbardziej kosztowny dla redakcji i jak to przekłada się na wybór metryki.

**Wczytanie danych:**
```python
import pandas as pd

df = pd.read_csv("data_group_3.csv")
print(df.shape)   # (2000, 16)
print(df.head())
```

*That's all.*
