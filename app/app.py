import streamlit as st
import anthropic
import pickle
import pandas as pd
import numpy as np
import shap
from dotenv import load_dotenv
import os
import json
import re

# Wczytanie klucza API
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Wczytanie modelu
@st.cache_resource
def load_model():
    with open('model/credit_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Konfiguracja strony
st.set_page_config(
    page_title="Agent oceny zdolności kredytowej",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Agent oceny zdolności kredytowej")
st.caption("System wspomagający doradcę bankowego")

# Mapowanie zawodów na liczby
occupation_mapping = {
    'Accountant': 0, 'Architect': 1, 'Developer': 2,
    'Doctor': 3, 'Engineer': 4, 'Entrepreneur': 5,
    'Journalist': 6, 'Lawyer': 7, 'Manager': 8,
    'Mechanic': 9, 'Media_Manager': 10, 'Musician': 11,
    'Scientist': 12, 'Teacher': 13, 'Writer': 14
}

# Polskie nazwy cech
feature_names_pl = {
    'Age': 'Wiek klienta',
    'Occupation': 'Zawód',
    'Annual_Income': 'Roczny dochód (USD)',
    'Monthly_Inhand_Salary': 'Miesięczna pensja netto (USD)',
    'Num_Bank_Accounts': 'Liczba kont bankowych',
    'Num_Credit_Card': 'Liczba kart kredytowych',
    'Num_of_Loan': 'Liczba aktywnych pożyczek',
    'Outstanding_Debt': 'Całkowite zadłużenie (USD)',
    'Total_EMI_per_month': 'Miesięczna suma rat (USD)',
    'Delay_from_due_date': 'Średnie opóźnienie spłaty (dni)',
    'Num_of_Delayed_Payment': 'Liczba opóźnionych płatności',
    'Credit_History_Age': 'Długość historii kredytowej (miesiące)',
    'Payment_of_Min_Amount': 'Spłata minimalnej kwoty (1=Tak, 0=Nie)',
    'Credit_Utilization_Ratio': 'Wykorzystanie limitu kredytowego (%)',
    'Credit_Mix': 'Jakość portfela (0=Bad, 1=Standard, 2=Good)',
    'Monthly_Balance': 'Miesięczne saldo konta (USD)'
}

def wywolaj_model(dane):
    """Przyjmuje slownik z danymi klienta, zwraca wynik i SHAP"""
    try:
        wiek = dane.get('wiek', 30)
        zawod = dane.get('zawod', 'Engineer')
        roczny_dochod = dane.get('roczny_dochod', 50000)
        miesieczna_pensja = dane.get('miesieczna_pensja', 4000)
        liczba_kont = dane.get('liczba_kont', 2)
        liczba_kart = dane.get('liczba_kart', 2)
        liczba_pozyczek = dane.get('liczba_pozyczek', 1)
        zadluzenie = dane.get('zadluzenie', 1000)
        miesieczne_raty = dane.get('miesieczne_raty', 200)
        opoznienie_dni = dane.get('opoznienie_dni', 0)
        liczba_opoznien = dane.get('liczba_opoznien', 0)
        historia = dane.get('historia_kredytowa', 60)
        splata_min = dane.get('splata_minimum', 'Tak')
        wykorzystanie = dane.get('wykorzystanie_limitu', 30)
        jakosc = dane.get('jakosc_portfela', 'Standard')
        saldo = dane.get('saldo_miesieczne', 500)
        kwota_kredytu = dane.get('kwota_kredytu', 0)
        okres_kredytowania = dane.get('okres_kredytowania', 12)

        df_input = pd.DataFrame([{
            'Age': float(wiek),
            'Occupation': occupation_mapping.get(zawod, 4),
            'Annual_Income': float(roczny_dochod),
            'Monthly_Inhand_Salary': float(miesieczna_pensja),
            'Num_Bank_Accounts': int(liczba_kont),
            'Num_Credit_Card': int(liczba_kart),
            'Num_of_Loan': int(liczba_pozyczek),
            'Outstanding_Debt': float(zadluzenie),
            'Total_EMI_per_month': float(miesieczne_raty),
            'Delay_from_due_date': float(opoznienie_dni),
            'Num_of_Delayed_Payment': float(liczba_opoznien),
            'Credit_History_Age': float(historia),
            'Payment_of_Min_Amount': 1 if splata_min == 'Tak' else 0,
            'Credit_Utilization_Ratio': float(wykorzystanie),
            'Credit_Mix': {'Good': 2, 'Standard': 1, 'Bad': 0}.get(jakosc, 1),
            'Monthly_Balance': float(saldo)
        }])

        # Predykcja
        wynik = model.predict(df_input)[0]
        klasy = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        wynik_nazwa = klasy[wynik]

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        feature_names = df_input.columns.tolist()

        if isinstance(shap_values, list):
            shap_vals_raw = shap_values[wynik][0]
        else:
            shap_vals_raw = shap_values[0, :, wynik]

        # Top 3 cechy z kierunkiem i wartością
        top_indices = np.argsort(np.abs(shap_vals_raw))[-3:][::-1]
        top_features = []
        for i in top_indices:
            nazwa_en = feature_names[i]
            nazwa_pl = feature_names_pl.get(nazwa_en, nazwa_en)
            wartosc = float(df_input[nazwa_en].iloc[0])
            wplyw = float(shap_vals_raw[i])
            kierunek = "pozytywny" if wplyw > 0 else "negatywny"
            top_features.append({
                'nazwa': nazwa_pl,
                'wartosc': wartosc,
                'wplyw': abs(wplyw),
                'kierunek': kierunek
            })

        # Obliczenie DTI
        if okres_kredytowania > 0 and miesieczna_pensja > 0:
            szacowana_rata = float(kwota_kredytu) / float(okres_kredytowania)
            dti = (szacowana_rata / float(miesieczna_pensja)) * 100
        else:
            szacowana_rata = 0
            dti = 0

        return wynik_nazwa, top_features, float(kwota_kredytu), float(okres_kredytowania), szacowana_rata, dti

    except Exception as e:
        return None, str(e), 0, 0, 0, 0

# Prompt systemowy
SYSTEM_PROMPT = """Jesteś asystentem doradcy bankowego wspierającym ocenę zdolności kredytowej klienta.
Twoim zadaniem jest zebranie danych o kliencie zadając pytania jedno po drugim.

Zbierz następujące informacje w tej kolejności:
1. Kwota wnioskowanego kredytu w USD (liczba dodatnia)
2. Okres kredytowania w miesiącach (liczba dodatnia)
3. Wiek klienta (liczba całkowita, 18-100)
4. Zawód klienta (wybierz z listy: Scientist, Teacher, Engineer, Accountant, Developer, Doctor, Lawyer, Manager, Mechanic, Musician, Journalist)
5. Roczny dochód klienta w USD (liczba dodatnia)
6. Miesięczna pensja netto w USD (liczba dodatnia)
7. Liczba kont bankowych (0-20)
8. Liczba kart kredytowych (0-20)
9. Liczba aktywnych pożyczek (0-20)
10. Całkowite zadłużenie w USD (liczba nieujemna)
11. Miesięczna suma rat w USD (liczba nieujemna)
12. Średnie opóźnienie spłaty w dniach (liczba nieujemna)
13. Liczba opóźnionych płatności (liczba nieujemna)
14. Długość historii kredytowej w miesiącach (liczba nieujemna)
15. Czy klient regularnie spłaca minimalne kwoty? (Tak/Nie/Brak danych)
16. Wskaźnik wykorzystania limitu kredytowego w % (0-100)
17. Jakość portfela kredytowego (Good/Standard/Bad)
18. Miesięczne saldo konta w USD (liczba nieujemna)

Zasady:
- Zadawaj jedno pytanie na raz
- Waliduj odpowiedzi — jeśli odpowiedź jest nieprawidłowa poproś o poprawienie
- Dla zawodu — jeśli podany zawód nie jest na liście zaproponuj najbliższą kategorię
- Prowadź rozmowę po polsku
- Bądź uprzejmy i profesjonalny
- Gdy zbierzesz wszystkie dane napisz dokładnie: DANE_KOMPLETNE i wypisz zebrane wartości w formacie JSON używając DOKŁADNIE tych kluczy bez polskich znaków:
{
  "kwota_kredytu": ,
  "okres_kredytowania": ,
  "wiek": ,
  "zawod": ,
  "roczny_dochod": ,
  "miesieczna_pensja": ,
  "liczba_kont": ,
  "liczba_kart": ,
  "liczba_pozyczek": ,
  "zadluzenie": ,
  "miesieczne_raty": ,
  "opoznienie_dni": ,
  "liczba_opoznien": ,
  "historia_kredytowa": ,
  "splata_minimum": ,
  "wykorzystanie_limitu": ,
  "jakosc_portfela": ,
  "saldo_miesieczne": 
}"""

# Inicjalizacja historii czatu
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.data_collected = False

# Wyświetlenie historii czatu
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Pierwsze powitanie
if len(st.session_state.messages) == 0:
    welcome = "Dzień dobry. Jestem asystentem oceny zdolności kredytowej. Aby rozpocząć analizę proszę odpowiadać na moje pytania.\n\nJaką kwotę kredytu wnioskuje klient (w USD)?"
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    with st.chat_message("assistant"):
        st.markdown(welcome)

# Pole do wpisywania
if prompt := st.chat_input("Wpisz odpowiedź..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Wywołanie Claude API
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=st.session_state.messages
    )

    assistant_message = response.content[0].text
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    with st.chat_message("assistant"):
        st.markdown(assistant_message)

    # Sprawdzenie czy agent zebrał wszystkie dane
    if "DANE_KOMPLETNE" in assistant_message and not st.session_state.data_collected:
        st.session_state.data_collected = True
        try:
            cleaned = assistant_message.replace('```json', '').replace('```', '')
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)

            if json_match:
                dane = json.loads(json_match.group())
                wynik, top_features, kwota, okres, rata, dti = wywolaj_model(dane)

                if wynik and isinstance(top_features, list):
                    cechy_opis = "\n".join([
                        f"- {f['nazwa']}: wartość={f['wartosc']}, "
                        f"wpływ={f['wplyw']:.3f}, kierunek={f['kierunek']}"
                        for f in top_features
                    ])

                    interpretacja_prompt = f"""Na podstawie zebranych danych model ocenił klienta jako: {wynik}

Trzy czynniki które najbardziej wpłynęły na tę decyzję:
{cechy_opis}

Dodatkowe informacje o wniosku kredytowym:
- Kwota wnioskowanego kredytu: {kwota:,.0f} USD
- Okres kredytowania: {okres:.0f} miesięcy
- Szacowana miesięczna rata: {rata:,.0f} USD
- Wskaźnik obciążenia dochodu ratą (DTI): {dti:.1f}%

Ważne: kierunek "pozytywny" oznacza że cecha poprawiła ocenę, "negatywny" że ją pogorszyła.
Interpretuj wartości zgodnie z ich rzeczywistym znaczeniem.

Przedstaw doradcy bankowemu:
1. Wynik oceny kredytowej wraz z rekomendacją:
   - Good = przyznać kredyt na standardowych warunkach
   - Standard = rozważyć z dodatkowymi warunkami lub zabezpieczeniem
   - Poor = odmówić lub wymagać silnego zabezpieczenia
2. Ocenę czy kwota i okres kredytowania są adekwatne do profilu klienta (uwzględnij DTI)
3. Wyjaśnienie które czynniki wpłynęły na decyzję i w jaki sposób
4. Ewentualne sugestie co klient mógłby poprawić

Ważne zasady przy formułowaniu sugestii dla klienta:
- Sugeruj poprawę tylko tych cech które faktycznie wymagają poprawy
- Jeśli cecha ma już optymalną wartość np. liczba pożyczek = 1, opóźnienia = 0, 
  wykorzystanie limitu poniżej 30% — NIE sugeruj jej poprawy, zamiast tego 
  pochwal klienta za utrzymanie tej wartości
- Sugestie muszą być logicznie spójne z rzeczywistymi wartościami klienta
- Nie sugeruj działań które klient już wykonał prawidłowo

Ważne ograniczenie: opieraj się WYŁĄCZNIE na danych zebranych podczas rozmowy 
oraz czynnikach SHAP które otrzymałeś. Nie wspominaj o żadnych zmiennych 
których nie zbierałeś — takich jak własność nieruchomości, stan cywilny, 
wykształcenie, certyfikaty zawodowe czy inne dane spoza analizy. 
Jeśli chcesz wyjaśnić dlaczego ocena jest Standard a nie Good — odwołuj się 
tylko do zebranych danych.

Odpowiedź po polsku, profesjonalnie ale zrozumiale."""

                    interpretacja = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1000,
                        messages=[{
                            "role": "user",
                            "content": interpretacja_prompt
                        }]
                    )

                    wynik_message = interpretacja.content[0].text
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": wynik_message
                    })
                    with st.chat_message("assistant"):
                        st.markdown(wynik_message)
                else:
                    st.error(f"Błąd modelu: {top_features}")

        except Exception as e:
            st.error(f"Błąd podczas analizy danych: {str(e)}")