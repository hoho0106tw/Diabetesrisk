# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 20:47:55 2026

@author: Administrator
"""

# 3_test_risk.py
import pandas as pd
import pickle

# -----------------------------
# 載入模型
# -----------------------------
with open("cox_diabetes_model.pkl", "rb") as f:
    cph = pickle.load(f)

# -----------------------------
# 手動輸入一位病人的特徵
# -----------------------------
test_patient = pd.DataFrame([{
    "age": 25,
    "bmi": 15,
    "hba1c": 8.2,
    "family_dm": 1,
    "smoker": 0
}])

# -----------------------------
# 預測 5 年存活率 → 轉風險
# -----------------------------
surv_5y = cph.predict_survival_function(test_patient, times=[5.0])
risk_5y = 1 - surv_5y.iloc[0, 0]

print("=== 病人特徵 ===")
print(test_patient)

print(f"\n🩺 預測 5 年內得糖尿病機率：{risk_5y:.2%}")
