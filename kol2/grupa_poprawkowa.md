# Kolokwium poprawkowe — ML i Sieci Neuronowe

Twój manager zleca Ci zbudowanie modelu do przewidywania odejścia klienta od operatora telekomunikacyjnego. Będzie on używany, aby zaproponować klientom lepsze oferty i zatrzymać ich w firmie. Firma nie ma jednak budżetu, aby oferować je ponad potrzeby.

> **Twoje rozwiązanie musi zawierać:**
> - przygotowanie danych: obsługę braków, kodowanie zmiennych kategorycznych, skalowanie cech
> - model baseline zbudowany według opisu
> - dwa modele autorskie: jeden klasyczny model ML, jeden sieć neuronową
> - w modelu autorskim — wybrany mechanizm zapobiegający overfittingowi (regularyzacja, dropout, early stopping)
> - uzasadnienie wyboru algorytmu/architektury dla każdego modelu autorskiego
> - wybraną metrykę oceny wraz z jednozdaniowym uzasadnieniem jej doboru
> - omówienie wyników wszystkich pipeline'ów, wskazanie najlepszego i wyjaśnienie przyczyn rezultatów jednego z nich

**Dane:**
Dane klientów operatora telekomunikacyjnego — 7 043 rekordy, 12 cech.
Cechy numeryczne: staż klienta, miesięczne opłaty, łączne opłaty.
Cechy kategoryczne (tekstowe, binarne): płeć, posiadanie partnera, osoby na utrzymaniu,
usługa telefoniczna, wiele linii, ochrona online, wsparcie techniczne, rozliczenia elektroniczne.
Wybrane kolumny zawierają braki danych.
Target: `Churn` (`Yes` = klient odszedł, `No` = pozostał).

**Model baseline:**
Pipeline złożony z dwóch kroków: redukcji wymiarowości PCA zachowującej **85% wariancji**,
a następnie SVM z jądrem wielomianowym stopnia 3 i współczynnikiem regularyzacji **C=0.1**.

**Wczytanie danych:**
```python
import pandas as pd
import numpy as np

df = pd.read_csv(
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

drop_cols = [
    "customerID", "InternetService", "Contract", "PaymentMethod",
    "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies",
]
df = df.drop(columns=drop_cols)

df["MultipleLines"] = df["MultipleLines"].replace("No phone service", np.nan)
df["OnlineSecurity"] = df["OnlineSecurity"].replace("No internet service", np.nan)
df["TechSupport"] = df["TechSupport"].replace("No internet service", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
```
