import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 設定 ---
PAYBACK_RATES = {1: 0.970, 2: 0.980, 3: 0.995, 4: 1.011, 5: 1.033, 6: 1.055}
TRUST_CONSTANT = 3000

# モデルの読み込み
@st.cache_resource
def load_model():
    return joblib.load('juggler_ai_model_pro_light.pkl')

model = load_model()

# --- UI部分 ---
st.set_page_config(page_title="Juggler AI Analyzer", layout="centered")
st.title("🎰 ジャグラー設定推測 AI")
st.caption("現場で即座に期待値を算出します")

with st.form("input_form"):
    st.subheader("現在の台データ入力")
    
    col1, col2 = st.columns(2)
    with col1:
        current_g = st.number_input("総回転数 (G)", min_value=0, value=1000, step=100)
        big = st.number_input("BIG回数", min_value=0, value=3, step=1)
        reg = st.number_input("REG回数", min_value=0, value=3, step=1)
    
    with col2:
        grape = st.number_input("ぶどう回数", min_value=0, value=160, step=1)
        max_hamari = st.number_input("最大ハマり (G)", min_value=0, value=200, step=10)
    
    submitted = st.form_submit_button("AI判定を実行")

if submitted:
    # 前処理
    reg_rate = current_g / max(1, reg)
    v_rate = current_g / max(1, (big + reg))
    
    features = np.array([[current_g, big, reg, grape, max_hamari, reg_rate, v_rate]])
    
    # 予測
    probs = model.predict_proba(features)[0]
    pred_setting = model.predict(features)[0]
    
    # 信頼度補正計算
    raw_payback = sum(probs[i] * PAYBACK_RATES[i+1] for i in range(6))
    confidence = min(1.0, current_g / TRUST_CONSTANT)
    adj_payback = (raw_payback * confidence) + (PAYBACK_RATES[1] * (1 - confidence))
    
    # --- 結果表示 ---
    st.divider()
    
    # メイン結果
    st.metric(label="AI予測設定", value=f"設定 {int(pred_setting)}")
    
    st.subheader("📊 解析詳細")
    c1, c2 = st.columns(2)
    c1.metric("期待機械割 (補正後)", f"{adj_payback*100:.2f}%")
    c2.metric("データ信頼度", f"{confidence*100:.1f}%")

    # 棒グラフで設定期待度を可視化
    st.write("各設定の期待度:")
    chart_data = pd.DataFrame({
        '設定': [f"設定{i}" for i in range(1, 7)],
        '確率(%)': [p * 100 for p in probs]
    })
    st.bar_chart(data=chart_data, x='設定', y='確率(%)')

    if adj_payback > 1.0:
        st.success(f"【続行推奨】期待値はプラスです（残り2000Gで約 {int((2000*3)*(adj_payback-1)*20):,}円）")
    else:

        st.error("【注意】期待値が100%を下回っています。")
