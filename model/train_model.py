import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import shap
import matplotlib.pyplot as plt

# Wczytanie czystych danych
df = pd.read_csv('data/train_clean.csv')

# Usuniecie outlierow w Annual_Income
q99 = df['Annual_Income'].quantile(0.99)
df = df[df['Annual_Income'] <= q99]

print(f"Rekordów po usunięciu outlierów: {len(df)}")
print(f"Maksymalne Annual_Income po czyszczeniu: {df['Annual_Income'].max():,.0f}")

# Sprawdzenie korelacji zmiennych z targetem
correlations = df.corr()['Credit_Score'].sort_values(ascending=False)
print("\nKorelacja zmiennych z Credit_Score:")
print(correlations)

print(f"\nPozostałe cechy: {df.columns.tolist()}")

# Podział na cechy i target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

print(f"\nKształt danych: {X.shape}")
print(f"Rozkład klas przed SMOTE:")
print(y.value_counts())

# Podział na zbiór treningowy i testowy (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nRozkład po podziale:")
print(f"Zbiór treningowy: {X_train.shape[0]} rekordów")
print(f"Zbiór testowy: {X_test.shape[0]} rekordów")

# Wyrównanie klas przez SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nRozkład klas po SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Trenowanie modelu XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train_smote, y_train_smote)

# Ocena modelu na zbiorze testowym
y_pred = model.predict(X_test)

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred,
      target_names=['Poor', 'Standard', 'Good']))

print("\nMacierz pomyłek:")
print(confusion_matrix(y_test, y_pred))

# Zapis modelu
with open('model/credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel zapisany do pliku credit_model.pkl")

# Generowanie wykresu SHAP
print("\nGenerowanie wykresu SHAP...")

# Polskie nazwy cech
feature_names_pl = {
    'Age': 'Wiek klienta',
    'Occupation': 'Zawód',
    'Annual_Income': 'Roczny dochód',
    'Monthly_Inhand_Salary': 'Miesięczna pensja netto',
    'Num_Bank_Accounts': 'Liczba kont bankowych',
    'Num_Credit_Card': 'Liczba kart kredytowych',
    'Num_of_Loan': 'Liczba aktywnych pożyczek',
    'Outstanding_Debt': 'Całkowite zadłużenie',
    'Total_EMI_per_month': 'Miesięczna suma rat',
    'Delay_from_due_date': 'Opóźnienie spłaty (dni)',
    'Num_of_Delayed_Payment': 'Liczba opóźnionych płatności',
    'Credit_History_Age': 'Długość historii kredytowej',
    'Payment_of_Min_Amount': 'Spłata minimalnej kwoty',
    'Credit_Utilization_Ratio': 'Wykorzystanie limitu kredytowego',
    'Credit_Mix': 'Jakość portfela kredytowego',
    'Monthly_Balance': 'Miesięczne saldo konta'
}

X_test_pl = X_test.rename(columns=feature_names_pl)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_pl)

fig = plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test_pl,
    plot_type="bar",
    class_names=['Poor', 'Standard', 'Good'],
    show=False
)

plt.xlabel('Średni wpływ na decyzję modelu (wartość SHAP)', fontsize=11)
plt.tight_layout()
plt.savefig('model/shap_importance.png',
            dpi=200,
            bbox_inches='tight',
            transparent=True)
plt.close()
print("Wykres SHAP zapisany do model/shap_importance.png")