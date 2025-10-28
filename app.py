from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math
import random

# --- Konfiguration und Initialisierung ---

# Initialisierung der Flask-App
# static_folder='.' ist ESSENZIELL, damit Flask die index.html im Root findet.
app = Flask(__name__, static_folder='.', static_url_path='')

# Globale Konstanten
NUM_SIMULATIONEN = 1000
RISIKO_LEVEL = 0.05  # 5% für CVaR (Conditional Value at Risk)

# --- Hilfsfunktionen für die Finanzmathematik ---

def lade_historische_kurse():
    """
    Simuliert historische Schlusskurse für vier Assets.
    In einem realen Projekt würde hier ein API-Aufruf erfolgen.
    """
    random.seed(42) # Für reproduzierbare Simulationen
    
    start_datum = pd.to_datetime('2020-01-01')
    end_datum = pd.to_datetime('2024-01-01')
    datumsbereich = pd.date_range(start=start_datum, end=end_datum, freq='D')
    
    np.random.seed(42)
    
    # Startpreise und tägliche Volatilität/Drift
    start_preise = np.array([100, 50, 200, 75])
    volatilitaeten = np.array([0.015, 0.02, 0.01, 0.03])
    drifts = np.array([0.0002, 0.0001, 0.0003, 0.00005])
    
    # Erzeuge Kurse mit geometrischer Brown'scher Bewegung
    kurse = np.zeros((len(datumsbereich), 4))
    kurse[0] = start_preise
    
    for i in range(1, len(datumsbereich)):
        # Zufällige Bewegungen
        zufall = np.random.normal(0, 1, 4)
        kurse[i] = kurse[i-1] * np.exp(drifts + volatilitaeten * zufall)
        
    df_kurse = pd.DataFrame(kurse, index=datumsbereich, columns=['Asset A (Tech)', 'Asset B (Energy)', 'Asset C (Bonds)', 'Asset D (Gold)'])
    
    return df_kurse.asfreq('B').ffill() # Nur Börsentage, fehlende Daten vorfüllen

def berechne_historische_parameter(df_kurse):
    """
    Berechnet tägliche Renditen, Mittelwerte und die Kovarianzmatrix.
    """
    renditen = df_kurse.pct_change().dropna()
    mittelwerte = renditen.mean()
    kovarianzmatrix = renditen.cov()
    asset_namen = renditen.columns.tolist()
    
    # Skalierung auf jährliche Basis für Mittelwerte (252 Handelstage)
    jaehrliche_mittelwerte = mittelwerte * 252
    
    return renditen, jaehrliche_mittelwerte, kovarianzmatrix, asset_namen

# --- Core-Optimierungsfunktionen ---

def portfolio_return(gewichte, mittelwerte):
    """Berechnet die erwartete jährliche Portfoliorendite."""
    return np.sum(gewichte * mittelwerte)

def portfolio_volatility(gewichte, kovarianzmatrix):
    """Berechnet die jährliche Portfolio-Volatilität (Standardabweichung)."""
    # Kovarianz ist tägliche Kovarianz, muss auf jährlich skaliert werden (sqrt(252))
    return np.sqrt(np.dot(gewichte.T, np.dot(kovarianzmatrix * 252, gewichte)))

def portfolio_value_at_risk(gewichte, renditen, risiko_level):
    """
    Berechnet den historischen Value at Risk (VaR) des Portfolios.
    VaR ist das Perzentil der schlechtesten Renditen.
    """
    # Rendite des Portfolios für jeden Tag in der Historie
    portfolio_renditen = renditen.dot(gewichte)
    
    # VaR ist das Perzentil der Renditen. 
    # Wir nehmen den Wert, bei dem die schlechtesten (risiko_level * 100)% der Fälle liegen.
    var = np.percentile(portfolio_renditen, risiko_level * 100)
    return var

def portfolio_cvar(gewichte, renditen, risiko_level):
    """
    Berechnet den Conditional Value at Risk (CVaR) des Portfolios.
    CVaR ist der Durchschnitt der Verluste, die schlimmer sind als VaR.
    """
    portfolio_renditen = renditen.dot(gewichte)
    
    # VaR: Das Rendite-Level, das nur in (risiko_level * 100)% der Fälle unterschritten wird
    var = portfolio_value_at_risk(gewichte, renditen, risiko_level)
    
    # CVaR: Durchschnitt aller Renditen, die kleiner oder gleich VaR sind
    cvar = portfolio_renditen[portfolio_renditen <= var].mean()
    
    # Da CVaR ein Verlust ist, geben wir es als positiven Wert für die Optimierung zurück (Ziel ist Minimierung)
    return -cvar

def optimiere_portfolio(ziel_rendite, renditen, mittelwerte, kovarianzmatrix, risiko_level):
    """
    Führt die CVaR-Minimierung für eine bestimmte Zielrendite durch.
    """
    num_assets = len(mittelwerte)
    
    # 1. Nebenbedingungen (Constraints)
    # C1: Summe der Gewichte muss 1 ergeben (Vollständige Investition)
    constraints = ({'type': 'eq', 'fun': lambda gewichte: np.sum(gewichte) - 1})
    
    # C2: Die erwartete Portfoliorendite muss mindestens der Zielrendite entsprechen
    constraints += ({'type': 'ineq', 'fun': lambda gewichte: portfolio_return(gewichte, mittelwerte) - ziel_rendite})
    
    # 2. Bindung (Bounds): Gewichte müssen zwischen 0 und 1 liegen (kein Leerverkauf)
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # 3. Startwerte: Gleichgewichtete Aufteilung
    initial_gewichte = np.array([1 / num_assets] * num_assets)
    
    # 4. Minimierungsfunktion: CVaR soll minimiert werden
    # Führt die Optimierung durch (Sequential Least Squares Programming)
    ergebnis = minimize(
        lambda gewichte: portfolio_cvar(gewichte, renditen, risiko_level),
        initial_gewichte, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    # Prüfe auf Erfolg
    if ergebnis.success:
        gewichte = ergebnis.x
        volatilitaet = portfolio_volatility(gewichte, kovarianzmatrix)
        cvar_wert = portfolio_cvar(gewichte, renditen, risiko_level)
        
        return {
            "erfolgreich": True,
            "gewichte": gewichte.tolist(),
            "rendite": portfolio_return(gewichte, mittelwerte),
            "volatilitaet": volatilitaet,
            # CVaR wird hier als negativer Wert zurückgegeben, da die Funktion ihn für die Minimierung negativ macht.
            # Wir geben den tatsächlichen Wert (Verlust) zurück.
            "cvar": cvar_wert 
        }
    else:
        return {"erfolgreich": False, "fehlermeldung": ergebnis.message}

def berechne_cvar_effizienzgrenze(renditen, mittelwerte, kovarianzmatrix, risiko_level, schritte=50):
    """
    Berechnet die CVaR-Effizienzgrenze durch Optimierung für eine Reihe von Zielrenditen.
    """
    # Definiere den Bereich der Zielrenditen
    min_rendite = np.min(mittelwerte) * 0.9 # Etwas niedriger als der schlechteste Asset
    max_rendite = np.max(mittelwerte) * 1.1 # Etwas höher als der beste Asset
    
    ziel_renditen = np.linspace(min_rendite, max_rendite, schritte)
    
    effizienzgrenze = []
    
    for ziel in ziel_renditen:
        optimierung = optimiere_portfolio(ziel, renditen, mittelwerte, kovarianzmatrix, risiko_level)
        
        if optimierung["erfolgreich"]:
            effizienzgrenze.append({
                "rendite": optimierung["rendite"],
                "risiko": optimierung["cvar"],
                "gewichte": optimierung["gewichte"]
            })
            
    return effizienzgrenze

# --- Globale Initialisierung der Parameter (wird beim Import durch Gunicorn ausgeführt) ---

try:
    KURSE = lade_historische_kurse()
    RENDITEN, MITTELWERTE, KOVARIANZMATRIX, ASSET_NAMEN = berechne_historische_parameter(KURSE)
    
    # Berechne die Effizienzgrenze nur einmal beim Start
    EFFIZIENZ_GRENZE = berechne_cvar_effizienzgrenze(RENDITEN, MITTELWERTE, KOVARIANZMATRIX, RISIKO_LEVEL)
    
    print("FINANZDATEN ERFOLGREICH INITIALISIERT UND CVAR-GRENZE BERECHNET.")

except Exception as e:
    # Fehler beim Laden/Berechnen der Daten, was den Gunicorn-Start verhindern würde
    print(f"KRITISCHER FEHLER BEIM INITIALISIEREN DER FINANZDATEN: {e}")
    # In einer realen Umgebung würde man hier einen Health Check Failed zurückgeben

# --- Flask-Routen ---

@app.route('/')
def serve_index():
    """Gibt die Front-End-Datei (index.html) aus."""
    return app.send_static_file('index.html')

@app.route('/optimieren', methods=['POST'])
def optimieren_api():
    """
    API-Endpunkt zur Optimierung eines Portfolios basierend auf der vom Benutzer
    gewünschten Zielrendite.
    """
    
    if not EFFIZIENZ_GRENZE:
         return jsonify({"error": "Finanzdaten konnten nicht initialisiert werden. Bitte Server-Logs prüfen."}), 500

    try:
        daten = request.get_json()
        ziel_rendite_prozent = daten.get('zielRendite', 0.10)  # Standard: 10%
        
        # Rendite von Prozent (z.B. 0.10) in Dezimalzahl (0.10) umwandeln
        ziel_rendite = float(ziel_rendite_prozent)
        
        # Finde den nächstgelegenen Punkt auf der Effizienzgrenze für die Zielrendite
        # Dies ist schneller, als jedes Mal neu zu optimieren
        
        beste_option = None
        min_differenz = float('inf')
        
        for punkt in EFFIZIENZ_GRENZE:
            differenz = abs(punkt['rendite'] - ziel_rendite)
            if differenz < min_differenz:
                min_differenz = differenz
                beste_option = punkt
        
        if beste_option:
            # Bereite die finale Antwort auf
            gewichtung_dict = {
                ASSET_NAMEN[i]: round(gewichte * 100, 2) 
                for i, gewichte in enumerate(beste_option["gewichte"])
            }
            
            antwort = {
                "erwarteteRendite": round(beste_option["rendite"] * 100, 2),
                "cvarRisiko": round(abs(beste_option["risiko"]) * 100, 2), # CVaR als positiven Verlust anzeigen
                "gewichtung": gewichtung_dict,
                "assetNamen": ASSET_NAMEN,
                "effizienzGrenze": [
                    {"rendite": round(p['rendite'] * 100, 2), "risiko": round(abs(p['risiko']) * 100, 2)} 
                    for p in EFFIZIENZ_GRENZE
                ]
            }
            return jsonify(antwort)
            
        else:
            return jsonify({"error": "Optimierung fehlgeschlagen oder Zielrendite nicht erreichbar."}), 400

    except Exception as e:
        # Rückgabe eines generischen Fehlers
        print(f"Fehler in der API-Verarbeitung: {e}")
        return jsonify({"error": f"Interner Serverfehler: {str(e)}"}), 500

# Wenn die App lokal gestartet wird (nicht über Gunicorn)
if __name__ == '__main__':
    app.run(debug=True)

