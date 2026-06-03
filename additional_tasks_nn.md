# Przykładowe zadania na kolokwium z Sieci Neuronowych

Poprosiłem Chada Gepette o wygenerowanie kilku zadań na bazie tego, co spotkacie na kolokwium. Oto one:

## Vino del Sud

Włoska winiarnia *Vino del Sud* wystawia swoje czerwone wina na prestiżowych
targach eksportowych we Francji. Każda butelka oznaczona jako **premium**
trafia bezpośrednio na stoisko degustacyjne — i jest oglądana przez
dystrybutorów z całej Europy.

Dział jakości dostał nowe zalecenia: zanim wino trafi na etykietę „premium",
musi przejść przez automatyczny filtr oparty na pomiarach fizykochemicznych.
Ręczna ocena każdej partii zajmuje zbyt dużo czasu przed targami.

Jesteś data scientist w laboratorium winiarni. Masz 90 minut, żeby zbudować
system, który odróżni wina godne eksportu od reszty.

> **Twoje rozwiązanie musi zawierać:**
> - przygotowanie danych: obsługę braków, kodowanie zmiennych kategorycznych, skalowanie cech)
> - model baseline zbudowany według opisu
> - dwa modele z early stoppingiem oraz wybranym mechanizmem zapobiegającym overfittingowi (regularyzacja, dropout)
> - uzasadnieniem wyboru architektury
> - wybraną metrykę oceny wraz z jednozdaniowym uzasadnieniem jej doboru
> - omówienie wyników wszystkich pipeline'ów, wskazanie najlepszego i wyjaśnienie przyczyn rezultatów jednego z nich.

**Kontekst:**
Laboratorium musi oznaczyć wina o wysokiej jakości (wynik ≥ 7 w skali 1–10)
jako godne eksportu. Błędna etykieta „premium" na winie przeciętnym
kompromituje markę na targach; pominięcie naprawdę dobrego wina oznacza
utratę przychodu z najwyższej półki.

**Dane:**
```python
import pandas as pd
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
    sep=';'
)
df['high_quality'] = (df['quality'] >= 7).astype(int)
```
1 599 próbek, 11 cech liczbowych (kwasowość stała i lotna, kwas cytrynowy,
cukier resztkowy, chlorki, wolny i całkowity dwutlenek siarki, gęstość, pH,
siarczany, zawartość alkoholu).
Target: `high_quality` (1 = wino godne eksportu, 0 = przeciętne).

**Model baseline:**
Model ma składać się z dwóch warstw gęstych: pierwszej z 64 neuronami,
drugiej z 32 neuronami oraz warstwy wyjściowej z 1 neuronem.

> Etykieta „premium" na przeciętnym winie trafia na stoisko degustacyjne —
> zastanów się, który rodzaj błędu niszczy reputację marki w oczach dystrybutorów,
> a który oznacza jedynie utracony przychód, i jak to przekłada się na wybór metryki.

## NewsWatch, Monitoring Mediów, 2026

Agencja prasowa *NewsWatch* archiwizuje miliony zdjęć ze światowych
serwisów informacyjnych. Redaktorzy muszą ręcznie tagować każde zdjęcie
przed publikacją — co przy obecnym wolumenie zajmuje kilka godzin dziennie.

Dział technologii dostał budżet na automatyzację: system ma sam rozpoznawać,
czy na zdjęciu widnieje konkretna osoba publiczna, i wstępnie tagować
zdjęcie bez udziału redaktora. Pilotaż rusza za miesiąc.

Jesteś ML engineerem w *NewsWatch*. Masz 90 minut, żeby zbudować
klasyfikator, który odróżni zdjęcia z twarzą **George'a W. Busha**
od zdjęć z innymi osobami.

> **Twoje rozwiązanie musi zawierać:**
> - przygotowanie danych: obsługę braków, kodowanie zmiennych kategorycznych, skalowanie cech)
> - model baseline zbudowany według opisu
> - dwa modele z early stoppingiem oraz wybranym mechanizmem zapobiegającym overfittingowi (regularyzacja, dropout)
> - uzasadnieniem wyboru architektury
> - wybraną metrykę oceny wraz z jednozdaniowym uzasadnieniem jej doboru
> - omówienie wyników wszystkich pipeline'ów, wskazanie najlepszego i wyjaśnienie przyczyn rezultatów jednego z nich.

**Kontekst:**
Błędny tag „Bush" na zdjęciu innej osoby trafia do archiwum i może zostać
opublikowany — narażając agencję na sprostowanie lub kompromitację;
pominięcie prawdziwego zdjęcia Busha to jedynie brak automatycznego tagu
i kilka sekund pracy redaktora.

**Dane:**
```python
from sklearn.datasets import fetch_lfw_people
import numpy as np

lfw = fetch_lfw_people(min_faces_per_person=70, color=False)
X = lfw.images                                         # (n, 62, 47), skala szarości
bush_idx = list(lfw.target_names).index('George W Bush')
y = (lfw.target == bush_idx).astype(int)
```
Około 1 140 zdjęć (62×47 px, skala szarości), naturalnie niezbalansowanych
(~530 zdjęć Busha vs. ~610 innych osób).
Target: `bush` (1) vs. `inna osoba` (0).

**Model baseline:**
Model ma składać się z warstwy spłaszczającej, po której następuje
warstwa ukryta z 128 neuronami oraz warstwa wyjściowa z 1 neuronem.
Trenowanie powinno trwać 20 epok.

> Zdjęcie z błędnym tagiem trafia do agencyjnego archiwum i może zostać
> użyte w artykule — zastanów się, który rodzaj błędu kompromituje agencję
> publicznie, a który jest niewidoczny dla czytelnika,
> i jak to przekłada się na wybór metryki.
