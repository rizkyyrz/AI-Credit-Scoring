from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap

app = FastAPI(title="PayLater AI Credit Scoring API")

# Wajib agar bisa diakses dari GitHub Pages (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model ML dan SHAP Explainer saat server nyala
model = joblib.load("model_paylater.pkl")
explainer = shap.TreeExplainer(model)

class ApplicantData(BaseModel):
    umur: int
    gaji: float
    cicilan: float
    lama_kerja: int
    telat_bayar: int
    utilisasi: float

@app.get("/")
def home():
    return {"status": "Online", "message": "PayLater AI Scoring API is running!"}

@app.post("/predict")
def predict_credit(data: ApplicantData):
    cols = [
        'Umur', 'Pendapatan_Bulanan_Juta', 'Total_Cicilan_Aktif_Juta',
        'Lama_Bekerja_Tahun', 'Riwayat_Telat_Bayar_Kali', 'Pemakaian_Limit_Sebelumnya_Persen'
    ]
    input_df = pd.DataFrame([[
        data.umur, data.gaji, data.cicilan, data.lama_kerja, data.telat_bayar, data.utilisasi
    ]], columns=cols)

    # 1. Prediksi Probabilitas Gagal Bayar dari ML
    prob_default = float(model.predict_proba(input_df)[0][1])
    credit_score = int(850 - (prob_default * 550))

    # 2. Kalkulasi Finansial & Limit PayLater
    dti = (data.cicilan / data.gaji) * 100 if data.gaji > 0 else 100
    sisa_kapasitas = max(0.0, (data.gaji * 0.35) - data.cicilan)

    if credit_score >= 700 and dti <= 40:
        status = "DISETUJUI"
        tier = "Tier Gold (Low Risk)"
        pengali = 3.5
    elif credit_score >= 580 and dti <= 50:
        status = "DISETUJUI BERSYARAT"
        tier = "Tier Silver (Medium Risk)"
        pengali = 1.5
    else:
        status = "DITOLAK"
        tier = "High Risk / Over-Leverage"
        pengali = 0.0

    limit_rp = int(round(sisa_kapasitas * pengali * 1_000_000, -4))

    # 3. Explainable AI (SHAP Values)
    shap_vals = explainer.shap_values(input_df)
    sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    labels = [
        f"Umur ({data.umur} th)",
        f"Gaji (Rp {data.gaji} Jt)",
        f"Cicilan (Rp {data.cicilan} Jt)",
        f"Masa Kerja ({data.lama_kerja} th)",
        f"Telat Bayar ({data.telat_bayar}x)",
        f"Utilisasi ({data.utilisasi}%)"
    ]

    return {
        "status": status,
        "tier": tier,
        "credit_score": credit_score,
        "prob_default": round(prob_default * 100, 1),
        "dti": round(dti, 1),
        "limit_rp": limit_rp,
        "shap_labels": labels,
        "shap_values": [round(float(x), 4) for x in sv]
    }
