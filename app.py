{\rtf1\ansi\ansicpg1252\cocoartf2865
\cocoatextscaling1\cocoaplatform1{\fonttbl\f0\fnil\fcharset0 .SFUI-Semibold;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f0\b\fs24 \cf0 # app.py - Der vollst\'e4ndige Back-End-Code f\'fcr die Mean-CVaR-Optimierung\
\
from flask import Flask, request, jsonify\
import numpy as np\
import pandas as pd\
from scipy.optimize import minimize \
\
# =========================================================================\
# I. GESCH\'c4FTSLOGIK (Das "Mini-Aladdin"-Modell)\
# =========================================================================\
\
# --- A1. DATEN UND PARAMETER-BERECHNUNG ---\
def lade_historische_kurse():\
    """Simuliert das Laden von realen Schlusskursen f\'fcr 2 Assets."""\
    # HINWEIS: Bei realen Daten m\'fcsste diese Funktion eine API nutzen (z.B. yfinance)\
    np.random.seed(42)\
    tage = 500\
    # Simulierte t\'e4gliche Renditen: Asset A (h\'f6heres Risiko) und Asset B (niedrigeres Risiko)\
    rendite_a = np.random.normal(loc=0.0005, scale=0.012, size=tage) \
    kurse_a = 100 * np.exp(np.cumsum(rendite_a))\
    rendite_b = np.random.normal(loc=0.0002, scale=0.005, size=tage)\
    # Korrelation simulieren\
    kurse_b = 50 * np.exp(np.cumsum(rendite_b + 0.3 * rendite_a)) \
    daten = pd.DataFrame(\{'Asset_A': kurse_a, 'Asset_B': kurse_b\})\
    return daten\
\
def berechne_historische_parameter(kurse_df):\
    """Berechnet t\'e4gliche Renditen, Mittelwerte und die Kovarianzmatrix."""\
    renditen = np.log(kurse_df / kurse_df.shift(1)).dropna()\
    mittelwerte = renditen.mean().values \
    kovarianzmatrix = renditen.cov().values\
    return renditen, mittelwerte, kovarianzmatrix, renditen.columns.tolist()\
\
# --- A2. MONTE-CARLO-RISIKOFUNKTION (CVaR) ---\
def berechne_cvar_fuer_optimierung(gewichtungen, mittelwerte, kovarianzmatrix, tage, simulationen, konfidenzniveau=0.99):\
    """Berechnet den CVaR f\'fcr eine gegebene Gewichtung mittels Monte-Carlo."""\
    anzahl_assets = len(mittelwerte)\
    try:\
        # Cholesky-Zerlegung zur Ber\'fccksichtigung der Korrelationen\
        L = np.linalg.cholesky(kovarianzmatrix)\
    except np.linalg.LinAlgError:\
        return 1000.0 # R\'fcckgabe eines hohen Wertes, wenn Matrix nicht positiv-definit ist\
\
    P0 = 1\
    end_portfolio_werte = np.zeros(simulationen)\
\
    for i in range(simulationen):\
        z = np.random.normal(size=(anzahl_assets, tage))\
        renditen_pfad = mittelwerte[:, None] + L @ z\
        portfolio_rendite_pfad = (gewichtungen @ renditen_pfad).sum(axis=0)\
        end_portfolio_werte[i] = P0 * np.prod(1 + portfolio_rendite_pfad)\
\
    end_renditen = (end_portfolio_werte - P0) / P0\
    alpha = 1.0 - konfidenzniveau\
    var_prozent = np.percentile(end_renditen, alpha * 100) \
    \
    # Der CVaR ist der Durchschnitt aller Verluste, die den VaR \'fcberschreiten\
    extreme_verluste = end_renditen[end_renditen <= var_prozent]\
    \
    if len(extreme_verluste) > 0:\
        cvar_prozent = extreme_verluste.mean()\
    else:\
        cvar_prozent = var_prozent \
\
    return abs(cvar_prozent * 100)\
\
# --- A3. OPTIMIERUNGS-ROUTINE ---\
def optimiere_portfolio(mittelwerte, kovarianzmatrix, R_ziel_tag, tage, simulationen, konfidenzniveau):\
    """F\'fchrt die Mean-CVaR-Optimierung durch."""\
    anzahl_assets = len(mittelwerte)\
    \
    # Zielfunktion: Minimiere den CVaR\
    ziel_funktion = lambda w: berechne_cvar_fuer_optimierung(w, mittelwerte, kovarianzmatrix, tage, simulationen, konfidenzniveau)\
\
    constraints = [\
        \{'type': 'eq', 'fun': lambda w: np.sum(w) - 1\}, # Constraint: Summe der Gewichte = 1\
        \{'type': 'ineq', 'fun': lambda w: np.dot(w, mittelwerte) - R_ziel_tag\} # Constraint: Rendite >= Ziel\
    ] \
    \
    grenzen = tuple((0, 1) for asset in range(anzahl_assets)) # Constraint: Keine Short-Positionen (0% bis 100%)\
    start_gewichtungen = np.array(anzahl_assets * [1. / anzahl_assets])\
    \
    optimale_ergebnisse = minimize(\
        ziel_funktion,\
        start_gewichtungen,\
        method='SLSQP',\
        bounds=grenzen,\
        constraints=constraints,\
        options=\{'disp': False\}\
    )\
    return optimale_ergebnisse\
\
# =========================================================================\
# II. FLASK API-STRUKTUR (Der Webserver)\
# =========================================================================\
\
app = Flask(__name__, static_folder='.', static_url_path='')\
\
# Globale Parameter und historische Daten einmalig beim Start der App laden\
KURSE = lade_historische_kurse()\
RENDITEN, MITTELWERTE, KOVARIANZMATRIX, ASSET_NAMEN = berechne_historische_parameter(KURSE)\
\
\
@app.route('/')\
def serve_index():\
    """Gibt die Front-End-Datei (index.html) aus."""\
    return app.send_static_file('index.html')\
\
\
@app.route('/optimieren', methods=['POST'])\
def optimiere_effizienzgrenze():\
    """API-Endpunkt zur Berechnung der Effizienzgrenze."""\
    # 1. Daten aus der POST-Anfrage extrahieren\
    try:\
        data = request.get_json()\
        anzahl_punkte = int(data.get('punkte', 50))\
        konfidenzniveau = float(data.get('konfidenz', 0.99))\
        N_SIM = int(data.get('simulationen', 10000))\
        T = 1\
    except Exception:\
        return jsonify(\{"fehler": "Ung\'fcltiger Parameterwert im Request-Body."\}), 400\
\
    # 2. Renditebereich definieren und Schleife vorbereiten\
    min_rendite_tag = MITTELWERTE.min()\
    max_rendite_tag = MITTELWERTE.max()\
    rendite_ziel_array_tag = np.linspace(min_rendite_tag, max_rendite_tag * 1.5, anzahl_punkte)\
    \
    cvar_werte = []\
    rendite_werte_jahr = []\
    gewichtungen_liste = []\
\
    # 3. Iteration und Optimierung (Berechnung der Effizienzgrenze)\
    for R_ziel_tag in rendite_ziel_array_tag:\
        optimale_ergebnisse = optimiere_portfolio(\
            MITTELWERTE, KOVARIANZMATRIX, R_ziel_tag, T, N_SIM, konfidenzniveau\
        )\
        \
        if optimale_ergebnisse.success:\
            optimale_gewichtungen = optimale_ergebnisse.x\
            minimaler_cvar = optimale_ergebnisse.fun\
            \
            # Skalierung der Rendite auf Jahresbasis f\'fcr die Ausgabe\
            erwartete_portfolio_rendite_tag = np.dot(optimale_gewichtungen, MITTELWERTE)\
            rendite_jahr = (1 + erwartete_portfolio_rendite_tag)**252 - 1\
            \
            cvar_werte.append(minimaler_cvar) \
            rendite_werte_jahr.append(rendite_jahr * 100) # In Prozent\
            gewichtungen_liste.append(optimale_gewichtungen.tolist())\
            \
    # 4. JSON-Antwort erstellen und zur\'fcckgeben\
    ergebnis = \{\
        "effizienzgrenze": \{\
            "cvar": cvar_werte,\
            "rendite": rendite_werte_jahr,\
            "gewichtungen": gewichtungen_liste\
        \},\
        "asset_namen": ASSET_NAMEN,\
        "konfidenzniveau": konfidenzniveau\
    \}\
\
    return jsonify(ergebnis)\
\
\
if __name__ == '__main__':\
    # Flask-App starten (wichtig f\'fcr Replit/Render)\
    app.run(host='0.0.0.0', port=8080)\
}