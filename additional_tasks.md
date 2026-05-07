# Zadania laboratoryjne - Machine Learning

---

## Zadanie 1 - Klasyfikacja (SVC baseline)

**Kontekst:** Pracujesz jako analityk danych w przychodni onkologicznej. Lekarze proszą cię o model wstępnego przesiewu guzów piersi - system ma flagować przypadki wymagające biopsji.

**Zadanie:** Na bazie zbioru Breast Cancer Wisconsin wytrenuj trzy pipeline'y (preprocessing + model): **SVC** jako baseline oraz dwa wybrane spośród KNN i Gaussian Naive Bayes. Oceń ich skuteczność. Który model zachował się lepiej? Uzasadnij podjęte decyzje. Co Twoim zdaniem jest przyczyną jego wyników?

**Model baseline:** `SVC(kernel='linear', C=1.0)`

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

---


## Zadanie 2 - Klasyfikacja szóstek (KNN baseline)

**Kontekst:** Pracujesz w archiwum narodowym digitalizującym historyczne dokumenty pocztowe. Skaner produkuje obrazy cyfr ze znaczków i kodów pocztowych. System musi wykrywać cyfrę 6, która jest szczególnie trudna do odróżnienia od innych cyfr i powoduje największą liczbę. Pracownikom zależy na zminimalizowaniu manualnej pracy z odrzuceniami i niepoprawnie zaklasyfikowanymi dokumentami.

**Zadanie:** Na bazie zbioru Digits wytrenuj trzy pipeline'y klasyfikacji binarnej (6 vs. reszta): **KNN** jako baseline oraz dwa wybrane spośród SVC i Gaussian Naive Bayes. Oceń ich skuteczność. Który model zachował się lepiej? Uzasadnij decyzję. Co Twoim zdaniem jest przyczyną jego wyników?

**Model baseline:** `KNeighborsClassifier(n_neighbors=5)`

```python
from sklearn.datasets import load_digits
data = load_digits()
X, y = data.data, (data.target == 6).astype(int)
```

> Zwróć uwagę na bilans zbiorów ;)

---

## Zadanie 3 - Regresja liniowa (LinearRegression baseline)

**Kontekst:** Pracujesz dla towarzystwa ubezpieczeniowego. Aktuariusze proszą cię o model szacujący roczne koszty leczenia pacjenta na podstawie jego danych demograficznych i stylu życia. W danych historycznych znajdują się rekordy z ekstremalnymi kosztami - nam zależy, aby model działał przede wszystkim na typowych przypadkach.

**Zadanie:** Na bazie zbioru Medical Cost Personal wytrenuj trzy pipeline'y regresji: **StandardScaler + LinearRegression** jako baseline oraz dwie autorskie propozycje. Oceń ich skuteczność. Który pipeline zachował się lepiej? Uzasadnij podjęte decyzje. Co Twoim zdaniem jest przyczyną jego wyników?

**Model baseline:** `LinearRegression()` poprzedzony `StandardScaler()`

```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
```

---

## Zadanie 4 - Klastrowanie (KMeans baseline)

**Kontekst:** Jesteś konsultantem dla sieci hurtowni spożywczych obsługującej restauracje i hotele w Portugalii. Zarząd chce podzielić klientów na segmenty według wzorców zakupowych, żeby dopasować oferty handlowe. Nie wiadomo z góry ile segmentów istnieje - zarząd prosi o uzasadnienie tej liczby.

**Zadanie:** Na bazie zbioru Wholesale Customers wytrenuj trzy pipeline'y klastrowania: **KMeans** poprzedzony **MinMaxScaler** jako baseline oraz dwa inne pipeline'y. Oceń ich skuteczność. Który model zachował się lepiej dla tego zadania? Uzasadnij podjęte decyzje. Co Twoim zdaniem jest przyczyną jego wyników?

**Model baseline:** `KMeans(n_clusters=3)` z `MinMaxScaler()`

```python
import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv")
```

---

## Zadanie 5 - Klastrowanie (DBSCAN baseline)

**Kontekst:** Pracujesz w zespole analitycznym operatora sieci komórkowej. Dział bezpieczeństwa prosi o wykrycie anomalnych wzorców aktywności użytkowników - potencjalnych botów lub kont przejętych przez oszustów. Dane o aktywności są gęsto skupione wokół typowych zachowań, ale zawierają nieliczne, rozproszone punkty odstające.

**Zadanie:** Na bazie zbioru Wine wytrenuj trzy pipeline'y klastrowania (ignorując etykiety): **DBSCAN** jako baseline oraz KMeans i Agglomerative Clustering. Oceń ich skuteczność. Który model lepiej nadaje się do tego zadania? Uzasadnij podjęte decyzje. Co Twoim zdaniem jest przyczyną jego wyników?

**Model baseline:** `DBSCAN(eps=0.5, min_samples=5)`

```python
from sklearn.datasets import load_wine
data = load_wine()
```

---

## Zadanie 6 - Klasyfikacja (Decision Tree baseline)

**Kontekst:** Pracujesz jako analityk w dziale HR dużej korporacji. Menedżerowie proszą cię o model przewidujący, czy pracownik odejdzie z firmy w ciągu najbliższego roku. Dział prawny zaznacza, że decyzje podjęte na podstawie modelu muszą być wytłumaczalne pracownikowi na jego prośbę. Możesz usunąć zmienne kategorialne.

**Zadanie:** Na bazie zbioru IBM HR Analytics wytrenuj trzy pipeline'y: **Decision Tree** jako baseline oraz dwa wybrane. Oceń ich skuteczność. Który model zachował się lepiej? Uzasadnij podjęte decyzje. Co Twoim zdaniem jest przyczyną jego wyników?

**Model baseline:** `DecisionTreeClassifier(max_depth=4)`

```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/prernagoel/IBM-Employee-Attrition-Analysis-in-Python/master/WA_Fn-UseC_-HR-Employee-Attrition.csv")
```

---

## Zadanie 7 - Regresja liniowa (PCA baseline)

**Kontekst:** Pracujesz dla agencji środowiskowej monitorującej jakość powietrza w miastach. Analitycy chcą przewidywać stężenie tlenku węgla na podstawie odczytów sensorów rozmieszczonych w różnych punktach miasta. Inżynierowie ostrzegają, że sensory są umieszczone blisko siebie i mierzą częściowo te same zjawiska - ich odczyty są ze sobą silnie skorelowane.

**Zadanie:** Na bazie zbioru Air Quality wytrenuj trzy pipeline'y regresji: **PCA + LinearRegression** jako baseline oraz dwie autorskie propozycje. Oceń ich skuteczność. Który pipeline zachował się lepiej? Uzasadnij podjęte decyzje. Co Twoim zdaniem jest przyczyną jego wyników?

**Model baseline:** `PCA(n_components=5)` a po nim `LinearRegression()`

```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/asharvi1/UCI-Air-Quality-Data/refs/heads/master/AirQualityUCI.csv", delimiter=";", decimal=",")
# target: 'CO(GT)'
```

---

## Zadanie 8 - Klastrowanie (Agglomerative baseline)

**Kontekst:** Pracujesz jako konsultant dla winnicy w Piemoncie. Enolodzy chcą pogrupować próbki win według składu chemicznego, żeby sprawdzić czy da się automatycznie odtworzyć podział na odmiany stosowany przez sommelierów. Zależy im na hierarchicznej strukturze grup - chcą wiedzieć, które odmiany są do siebie chemicznie najbliższe.

**Zadanie:** Na bazie zbioru Wine wytrenuj trzy pipeline'y klastrowania (ignorując etykiety): **Agglomerative Clustering** jako baseline oraz KMeans i DBSCAN. Oceń ich skuteczność. Który model zachował się lepiej dla tego zadania? Uzasadnij podjęte decyzje. Co Twoim zdaniem jest przyczyną jego wyników?

**Model baseline:** `AgglomerativeClustering(n_clusters=3)`

```python
from sklearn.datasets import load_wine
data = load_wine()
```
