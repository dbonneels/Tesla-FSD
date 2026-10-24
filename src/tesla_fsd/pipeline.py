"""Tesla-FSD pipeline — génère dashboard_data.json pour le dashboard Web.

Étapes :
1. Chargement des données (placeholder NHTSA à remplacer par les vrais fichiers)
2. Modèle de Poisson (ou binomiale négative) sur le nombre d'accidents
3. Export JSON lisible par dashboard.html
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data() -> pd.DataFrame:
    """Charge les données NHTSA.

    À remplacer par le chargement réel des fichiers NHTSA (CRASH, VEHICLE, PERSON).
    Retourne un DataFrame avec au minimum les colonnes :
    vehicle_id, fsd_enabled, fsd_version, mileage, accident, driver_age,
    driver_license_years, road_type, weather, speed_avg, region, vehicle_model
    """
    np.random.seed(42)
    n = 50_000

    df = pd.DataFrame({
        "vehicle_id": np.arange(n),
        "fsd_enabled": np.random.choice([0, 1], n, p=[0.55, 0.45]),
        "fsd_version": np.random.choice(
            ["base", "fsd_supervised", "fsd_beta"], n, p=[0.5, 0.35, 0.15]
        ),
        "mileage": np.random.exponential(15_000, n).round(0),
        "driver_age": np.random.normal(45, 12, n).clip(18, 85).round(0),
        "driver_license_years": np.random.normal(20, 10, n).clip(0, 60).round(0),
        "road_type": np.random.choice(
            ["autoroute", "urbaine", "rurale"], n, p=[0.4, 0.35, 0.25]
        ),
        "weather": np.random.choice(
            ["clair", "pluie", "neige", "brouillard"], n, p=[0.65, 0.25, 0.06, 0.04]
        ),
        "speed_avg": np.random.normal(90, 20, n).clip(30, 140).round(0),
        "region": np.random.choice(["CA", "TX", "FL", "NY", "WA"], n),
        "vehicle_model": np.random.choice(
            ["Model 3", "Model Y", "Model S", "Model X"], n, p=[0.45, 0.35, 0.12, 0.08]
        ),
    })

    # Taux d'accident de base par 100 000 km (illustratif, volontairement élevé
    # pour que le modèle converge sur ce jeu synthétique)
    base_per_100k = 15.0  # 15 accidents par 100 000 km
    fsd_effect = 0.75  # FSD réduit le risque de 25 % (illustratif)

    rate_no_fsd = base_per_100k / 100_000
    rate_fsd = base_per_100k / 100_000 * fsd_effect

    fsd_mask = df["fsd_enabled"] == 1

    # Nombre d'accidents par véhicule (loi de Poisson, adapté aux événements rares)
    lam_no = rate_no_fsd * df.loc[~fsd_mask, "mileage"]
    lam_yes = rate_fsd * df.loc[fsd_mask, "mileage"]

    df.loc[~fsd_mask, "accident"] = np.random.poisson(lam_no, int((~fsd_mask).sum()))
    df.loc[fsd_mask, "accident"] = np.random.poisson(lam_yes, int(fsd_mask.sum()))

    return df


def fit_model(df: pd.DataFrame) -> dict:
    """Ajuste une régression de Poisson et retourne les résultats clés."""
    df_model = df.copy()
    # Exclure les kilométrages nuls (log(0) = -inf)
    df_model = df_model[df_model["mileage"] > 0].copy()
    df_model["exposition"] = df_model["mileage"]

    try:
        mod = smf.poisson(
            formula=(
                "accident ~ fsd_enabled + driver_age + driver_license_years "
                "+ C(road_type) + C(weather) + speed_avg + C(region) + C(vehicle_model)"
            ),
            data=df_model,
            exposure=np.log(df_model["exposition"]),
        )
        res = mod.fit(cov_type="HC1", maxiter=200)

        rr = float(np.exp(res.params["fsd_enabled"]))
        ci = res.conf_int().loc["fsd_enabled"]
        ci_low = float(np.exp(ci[0]))
        ci_high = float(np.exp(ci[1]))
        converged = bool(res.mle_retvals.get("converged", True))
    except Exception:  # modèle qui ne converge pas — fallback sur les taux bruts
        converged = False
        total_km = float(df["mileage"].sum())
        total_acc = int(df["accident"].sum())
        km_fsd = float(df.loc[df["fsd_enabled"] == 1, "mileage"].sum())
        km_no = float(df.loc[df["fsd_enabled"] == 0, "mileage"].sum())
        acc_fsd = int(df.loc[df["fsd_enabled"] == 1, "accident"].sum())
        acc_no = int(df.loc[df["fsd_enabled"] == 0, "accident"].sum())

        rate_fsd = acc_fsd / km_fsd if km_fsd else 0.0
        rate_no = acc_no / km_no if km_no else 0.0
        rr = rate_fsd / rate_no if rate_no else 1.0
        ci_low = rr * 0.8
        ci_high = rr * 1.2

    # Si le modèle retourne des NaN (non-convergence numérique), fallback sur les taux bruts
    if rr != rr:  # NaN check
        total_km = float(df["mileage"].sum())
        km_fsd = float(df.loc[df["fsd_enabled"] == 1, "mileage"].sum())
        km_no = float(df.loc[df["fsd_enabled"] == 0, "mileage"].sum())
        acc_fsd = int(df.loc[df["fsd_enabled"] == 1, "accident"].sum())
        acc_no = int(df.loc[df["fsd_enabled"] == 0, "accident"].sum())
        rate_fsd = acc_fsd / km_fsd if km_fsd else 0.0
        rate_no = acc_no / km_no if km_no else 0.0
        rr = rate_fsd / rate_no if rate_no else 1.0
        ci_low = rr * 0.8
        ci_high = rr * 1.2
        converged = False

    rr = float(rr)
    ci_low = float(ci_low)
    ci_high = float(ci_high)

    return {
        "modele": "poisson_hc1" if converged else "taux_bruts_fallback",
        "source_donnees": "NHTSA (USA) — placeholder synthétique",
        "date_calcul": pd.Timestamp.now().isoformat(),
        "risque_relatif_fsd": rr,
        "ic_inf": ci_low,
        "ic_sup": ci_high,
        "variation_risque_pct": (rr - 1) * 100,
        "nb_vehicules": int(len(df)),
        "kilometrage_total_km": float(df["mileage"].sum()),
        "nb_accidents": int(df["accident"].sum()),
        "modele_converge": converged,
        "transfert_europe_hypothese": "H1 — non transférable sans récalibration",
        "recommandation": (
            "Recalibrer sur données européennes avant toute tarification UE"
        ),
    }


def main() -> None:
    out_path = Path("dashboard_data.json")
    df = load_data()
    results = fit_model(df)
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Exporté vers {out_path.resolve()}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
