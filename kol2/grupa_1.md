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
*Runway* digitalizuje archiwum 20 lat zdjęć produktowych przed kluczową sesją.
Redakcja potrzebuje modułu wykrywającego **obuwie** w katalogu — system
inwentaryzacji ma osobne działy dla butów i dla odzieży, więc but błędnie
wrzucony do działu odzieży psuje cały magazyn.
Sesja jest za tydzień. Nie ma czasu na ręczną korektę.

**Dane:**
42 000 zdjęć produktów (28×28 px, skala szarości).
Dane w pliku `data_group_1.npz` są już zbinaryzowane — `y` zawiera `0` lub `1`.
Target: `y` (1 = obuwie: Sandal, Sneaker, Ankle boot, 0 = reszta).

**Model baseline:**
Model ma składać się z warstwy spłaszczającej, po której następuje warstwa ukryta z 128 neuronami oraz warstwa wyjściowa z 1 neuronem. Trenowanie powinno trwać 20 epok.

> But, który wymknął się systemowi, trafia do złego działu tuż przed sesją.
> Zastanów się, który rodzaj błędu redakcja może sobie pozwolić,
> a którego absolutnie nie — i jak to przekłada się na wybór metryki.

**Wczytanie danych:**
```python
import numpy as np

data = np.load("data_group_1.npz")
X, y = data["X"], data["y"]   # X: (42000, 28, 28) float32, y: (42000,) labels 0/1
print(X.shape, y.shape)
```

*That's all.*
