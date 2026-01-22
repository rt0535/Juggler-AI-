import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ページ設定
st.set_page_config(page_title="最強脳ジャグラーAI", layout="centered")

st.title("🎰 最強脳：実戦判定ツール")
st.caption("設定推測分布 × 期待収支シミュレーター")

# モデルの読み込み
@st.cache_resource
def load_model():
    return joblib.load('juggler_ai_model_pro_light.pkl')

try:
    model = load_model()
except:
    st.error("モデルファイルが見つかりません。GitHubにpklファイルがアップロードされているか確認してください。")
    st.stop()

# --- 入力フォーム ---
with st.form("input_form"):
    st.subheader("📊 現在の基本データ")
    col1, col2 = st.columns(2)
    with col1:
        g = st.number_input("総回転数 (G)", min_value=1, value=1500, step=100)
        big = st.number_input("BIG回数", min_value=0, value=5, step=1)
    with col2:
        reg = st.number_input("REG回数", min_value=0, value=5, step=1)
        grape_mode = st.radio("ぶどう入力方法", ["直接入力", "差枚数から逆算"])

    # モードによって入力項目を切り替え
    if grape_mode == "直接入力":
        grape_input = st.number_input("ぶどう回数", min_value=0, value=250)
    else:
        diff_input = st.number_input("現在の差枚数 (例: -100, +500)", value=0)
        
        # --- ベル・ピエロを除外した精密逆算ロジック ---
        total_in = g * 3
        bonus_out = (big * 252) + (reg * 96)
        repl_out = (g / 7.298) * 3  # リプレイ期待値
        cherry_out = (g / 35.62) * 2 # チェリー期待値
        
        # 逆算式: 払い出した枚数のうち、ぶどうによるものを抽出
        # ぶどう払い出し = (投入 + 差枚) - ボーナス - リプレイ - チェリー
        calc_grape_payout = (total_in + diff_input) - bonus_out - repl_out - cherry_out
        grape_input = int(max(0, calc_grape_payout / 8)) 
        
        # 入力中の目安を表示
        if grape_input > 0:
            st.info(f"算出されたぶどう: {grape_input} 回 (1/{g/grape_input:.2f})")
        else:
            st.info("データを入力してください")

    st.divider()
    st.subheader("📈 履歴と展開のデータ")
    history_str = st.text_area("ボーナス履歴（カンマ区切り）", "100, 250, 50, 400, 120")
    
    st.divider()
    st.subheader("📅 閉店までのシミュレーション")
    remaining_g = st.slider("残り予定回転数", 500, 8000, 3000)
    
    submit = st.form_submit_button("🔥 AI鑑定を実行")

# --- 判定ロジック ---
if submit:
    try:
        # 入力修正
        history_str = history_str.replace("、", ",").replace(" ", ",")
        history = [int(x.strip()) for x in history_str.split(",") if x.strip()]
        
        if not history:
            st.warning("履歴を1件以上入力してください。")
            st.stop()
            
        # 特徴量計算
        reg_r = reg / g
        v_r = grape_input / g
        diff_reg = reg_r - (1/255.0)
        std_dev = np.std(history) if len(history) > 1 else 0
        max_h = max(history)
        
        # ボラティリティ（差枚変動幅）の計算
        in_t = g * 3
        out_t = (big * 252) + (reg * 96) + (grape_input * 8)
        # 期待値ベースの差枚（リプ・チェリー考慮）
        current_diff_est = out_t - (in_t + (g/7.298*3) + (g/35.62*2))
        volatility = np.abs(current_diff_est) / (g / 100)

        # AI入力 (10項目)
        features = ['current_g', 'big', 'reg', 'grape', 'reg_rate', 'v_rate', 
                    'diff_from_target_reg', 'std_dev_bonus_interval', 'volatility', 'max_hamari']
        input_df = pd.DataFrame([[g, big, reg, grape_input, reg_r, v_r, diff_reg, std_dev, volatility, max_h]], columns=features)
        
        # AIの推論
        probs = model.predict_proba(input_df)[0]
        best_s = np.argmax(probs) + 1
        
        # 期待値計算
        pay_outs = np.array([0.970, 0.980, 0.991, 1.011, 1.033, 1.055])
        expected_rtp = np.sum(probs * pay_outs)
        exp_profit_yen = remaining_g * 3 * (expected_rtp - 1) * 20
        hourly_rate = (exp_profit_yen / remaining_g) * 800

        # --- 結果表示UI ---
        st.divider()
        st.header("🏁 判定結果")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("予想設定", f"設定{best_s}")
        m2.metric("期待時給", f"{hourly_rate:+,.0f}円")
        m3.metric("算出ぶどう", f"1/{g/grape_input:.2f}" if grape_input > 0 else "---")

        # 確率グラフ
        st.bar_chart(pd.DataFrame({"設定": [f"設定{i+1}" for i in range(6)], "確率(%)": probs * 100}).set_index("設定"))

        # アドバイス表示
        if hourly_rate >= 2000:
            st.success(f"### 💡 結論: 続行推奨 🔥\n時給 {hourly_rate:,.0f} 円の期待値。設定{best_s}濃厚か。")
        elif hourly_rate > 0:
            st.warning(f"### 💡 結論: 続行可能 👍\n時給 {hourly_rate:,.0f} 円。プラス圏内です。")
        else:
            st.error(f"### 💡 結論: 撤退推奨 ✋\n期待値マイナス。設定{best_s}以下に注意。")

        st.info(f"詳細: 期待機械割 {expected_rtp*100:.2f}% / 最大ハマリ {max_h}G")

    except Exception as e:
        st.error(f"計算エラー: {e}")
