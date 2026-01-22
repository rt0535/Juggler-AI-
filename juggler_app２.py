import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ページ設定
st.set_page_config(page_title="最強脳ジャグラーAI", layout="centered")

st.title("🎰 最強脳：実戦判定ツール")
st.caption("設定推測分布 × 期待収支シミュレーター")

# モデルの読み込み（ファイル名は軽量化したものに合わせてください）
@st.cache_resource
def load_model():
    return joblib.load('juggler_ai_model_pro_light.pkl')

try:
    model = load_model()
except:
    st.error("モデルファイル 'juggler_ai_model_pro_light.pkl' が見つかりません。")
    st.stop()

# --- 入力フォーム ---
with st.form("input_form"):
    st.subheader("📊 現在の基本データ")
    col1, col2 = st.columns(2)
    with col1:
        g = st.number_input("総回転数", min_value=1, value=1500, step=100)
        big = st.number_input("BIG回数", min_value=0, value=5, step=1)
    with col2:
        reg = st.number_input("REG回数", min_value=0, value=5, step=1)
        grape = st.number_input("ぶどう回数", min_value=0, value=250, step=1)
    
    st.divider()
    st.subheader("📈 履歴と展開のデータ")
    # ここが重要！履歴をカンマ区切りで入力
    history_str = st.text_area("ボーナス履歴（カンマ区切りで入力）", 
                               "100, 250, 50, 400, 120", 
                               help="データ機の履歴を古い順、または新しい順にカンマで区切って入れてください")
    
    st.divider()
    st.subheader("📅 閉店までのシミュレーション")
    remaining_g = st.slider("残り予定回転数", 500, 8000, 3000)
    
    submit = st.form_submit_button("🔥 AI鑑定を実行")

# --- 判定ロジック ---
if submit:
    try:
        # 履歴をリストに変換
        history = [int(x.strip()) for x in history_str.split(",") if x.strip()]
        if not history:
            st.warning("履歴を1件以上入力してください。")
            st.stop()
            
        # --- 追加特徴量の自動計算 ---
        reg_r = reg / g
        v_r = grape / g
        diff_reg = reg_r - (1/255.0)
        std_dev = np.std(history) if len(history) > 1 else 0
        max_h = max(history)
        
        # 差枚数とボラティリティの概算
        in_t = g * 3
        out_t = (big * 240) + (reg * 96) + (grape * 8)
        current_diff = out_t - in_t
        volatility = np.abs(current_diff) / (g / 100)

        # AI入力用データフレーム作成（学習時と同じ順番）
        features = ['current_g', 'big', 'reg', 'grape', 'reg_rate', 'v_rate', 
                    'diff_from_target_reg', 'std_dev_bonus_interval', 'volatility', 'max_hamari']
        input_df = pd.DataFrame([[g, big, reg, grape, reg_r, v_r, diff_reg, std_dev, volatility, max_h]], columns=features)
        
        # AIの推論
        probs = model.predict_proba(input_df)[0]
        best_s = np.argmax(probs) + 1
        
        # 期待値計算（アイムジャグラー準拠）
        pay_outs = np.array([0.970, 0.980, 0.991, 1.011, 1.033, 1.055])
        expected_rtp = np.sum(probs * pay_outs)
        exp_profit_yen = remaining_g * 3 * (expected_rtp - 1) * 20
        hourly_rate = (exp_profit_yen / remaining_g) * 800

        # --- 結果表示UI ---
        st.divider()
        st.header("🏁 判定結果")
        
        # 4つの指標をカード表示
        m1, m2, m3 = st.columns(3)
        m1.metric("予想設定", f"設定{best_s}")
        m2.metric("期待時給", f"{hourly_rate:+,.0f}円")
        m3.metric("現在枚数", f"{current_diff:+.0f}枚")

        # 設定分布グラフ
        prob_df = pd.DataFrame({"設定": [f"設定{i+1}" for i in range(6)], "確率(%)": probs * 100})
        st.bar_chart(data=prob_df, x="設定", y="確率(%)")

        # アドバイス
        if hourly_rate >= 2000:
            st.success(f"### 💡 結論: 続行推奨 🔥\n時給 {hourly_rate:,.0f} 円の期待値があります。")
        elif hourly_rate > 0:
            st.warning(f"### 💡 結論: 続行可能 👍\n時給 {hourly_rate:,.0f} 円。波を掴みましょう。")
        else:
            st.error(f"### 💡 結論: 撤退推奨 ✋\n期待時給がマイナスです。深追いは厳禁。")

        st.info(f"計算詳細: 最大ハマリ {max_h}G / 標準偏差 {std_dev:.1f} / ぶどう確率 1/{1/v_r:.2f}")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
