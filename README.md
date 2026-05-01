# Agent-oceny-zdolnosci-kredytowej
Projekt agenta AI wspomagającego ocenę zdolności kredytowej, 
zrealizowany w ramach pracy magisterskiej.

## Opis
System łączy model klasyfikacyjny XGBoost z modelem językowym Claude, 
tworząc agenta konwersacyjnego przeznaczonego dla doradcy bankowego.

## Dane
Projekt wykorzystuje zbiór danych Credit Score Classification 
dostępny publicznie na platformie Kaggle:
https://www.kaggle.com/datasets/parisrohan/credit-score-classification

Aby uruchomić projekt należy pobrać plik train.csv 
i umieścić go w folderze data/.

## Struktura repozytorium
- data/ — folder na dane (train.csv pobierz z Kaggle)
- notebooks/ — notebook do analizy i przygotowania danych
- model/ — skrypty do trenowania modelu XGBoost
- app/ — aplikacja Streamlit z agentem
