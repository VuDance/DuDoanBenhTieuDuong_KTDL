import pandas as pd
import numpy as np
from apyori import apriori
import streamlit as st

# =============================
# 1️⃣ Load & Tiền xử lý dữ liệu
# =============================
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('data/diabetes.csv')

    # Xử lý giá trị 0 -> NaN -> median
    medical_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[medical_cols] = df[medical_cols].replace(0, np.nan)
    df[medical_cols] = df[medical_cols].fillna(df[medical_cols].median())

    # Rời rạc hóa (3 bins: low, medium, high)
    features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    bin_edges = {}
    for feature in features:
        df[f'{feature}_discretized'], edges = pd.cut(
            df[feature],
            bins=3,
            labels=['low', 'medium', 'high'],
            retbins=True,
            duplicates='drop'
        )
        bin_edges[feature] = edges

    # Outcome
    df['Outcome_discretized'] = df['Outcome'].map({0: 'No_Diabetes', 1: 'Diabetes'})

    # Tạo transactions
    transactions = []
    for _, row in df.iterrows():
        t = [f'{feat}_{row[f"{feat}_discretized"]}' for feat in features]
        t.append(row['Outcome_discretized'])
        transactions.append(t)

    return transactions, df, bin_edges


# =============================
# 2️⃣ Áp dụng Apriori để khai phá luật
# =============================
@st.cache_data
def mine_rules(transactions, min_support=0.1, min_conf=0.6):
    rules = apriori(
        transactions,
        min_support=min_support,
        min_confidence=min_conf,
        min_lift=1.0,
        min_length=2
    )

    results = []
    for rule in rules:
        support = rule.support
        for ordered_stat in rule.ordered_statistics:
            antecedent = ', '.join(list(ordered_stat.items_base))
            consequent = ', '.join(list(ordered_stat.items_add))
            confidence = ordered_stat.confidence
            lift = ordered_stat.lift
            results.append([antecedent, consequent, support, confidence, lift])

    rules_df = pd.DataFrame(
        results, columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']
    )

    # Chia rule cho 2 loại kết luận
    diabetes_rules = rules_df[rules_df['Consequent'] == 'Diabetes']
    no_diabetes_rules = rules_df[rules_df['Consequent'] == 'No_Diabetes']
    return diabetes_rules, no_diabetes_rules, rules_df


# =============================
# 3️⃣ Dự đoán theo luật
# =============================
def predict_diabetes(input_data, diabetes_rules, no_diabetes_rules, bin_edges, threshold=0.7):
    features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    discretized_input = {}
    for i, feat in enumerate(features):
        discretized_input[f'{feat}_discretized'] = pd.cut(
            [input_data[i]], bins=bin_edges[feat], labels=['low', 'medium', 'high']
        )[0]

    input_transaction = [f'{feat}_{discretized_input[f"{feat}_discretized"]}' for feat in features]

    def get_max_conf(rules):
        confs = []
        for _, rule in rules.iterrows():
            antecedent_items = rule['Antecedent'].split(', ')
            if all(item in input_transaction for item in antecedent_items):
                confs.append(rule['Confidence'])
        return max(confs) if confs else 0

    conf_pos = get_max_conf(diabetes_rules)
    conf_neg = get_max_conf(no_diabetes_rules)

    # So sánh confidence giữa 2 nhóm rule
    if conf_pos >= threshold and conf_pos > conf_neg:
        return f"🩸 Có nguy cơ TIỂU ĐƯỜNG (conf={conf_pos:.2f})"
    elif conf_neg >= threshold and conf_neg > conf_pos:
        return f"✅ Nguy cơ thấp (Không bệnh) (conf={conf_neg:.2f})"
    else:
        return f"🤔 Không xác định rõ (conf_pos={conf_pos:.2f}, conf_neg={conf_neg:.2f})"


# =============================
# 4️⃣ Streamlit UI
# =============================
def main():
    st.set_page_config(page_title="Chẩn đoán Tiểu Đường - Apriori", page_icon="🩺", layout="centered")
    st.title("🩺 Ứng Dụng Chẩn Đoán Bệnh Tiểu Đường (Apriori)")
    st.caption("Dựa trên dataset Pima Indians Diabetes (Kaggle)")

    # Load data & rules
    with st.spinner("🔄 Đang xử lý dữ liệu và khai phá luật..."):
        transactions, df, bin_edges = load_and_preprocess_data()
        diabetes_rules, no_diabetes_rules, rules_df = mine_rules(transactions)

    # Hiển thị thống kê
    st.subheader("📊 Thống kê dữ liệu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Số bệnh nhân", len(df))
    c2.metric("Tỷ lệ tiểu đường", f"{df['Outcome'].mean():.1%}")
    c3.metric("Tổng số luật", len(rules_df))

    st.subheader("🏷️ Top 5 Luật Tiểu Đường (Confidence cao)")
    st.dataframe(diabetes_rules.sort_values("Confidence", ascending=False).head(5))

    st.subheader("🏷️ Top 5 Luật Không Bệnh (Confidence cao)")
    st.dataframe(no_diabetes_rules.sort_values("Confidence", ascending=False).head(5))

    # Input form
    st.subheader("🧍‍♀️ Nhập Thông Tin Bệnh Nhân")
    feature_labels = {
        'Pregnancies': 'Số lần mang thai',
        'Glucose': 'Nồng độ glucose (mg/dL)',
        'BloodPressure': 'Huyết áp tâm trương (mm Hg)',
        'SkinThickness': 'Độ dày nếp gấp da (mm)',
        'Insulin': 'Nồng độ insulin (mu U/ml)',
        'BMI': 'Chỉ số khối cơ thể (kg/m²)',
        'DiabetesPedigreeFunction': 'Chỉ số di truyền tiểu đường',
        'Age': 'Tuổi (năm)'
    }

    # Giá trị mặc định “không bệnh”
    default_values = {
        'Pregnancies': 1,
        'Glucose': 90,
        'BloodPressure': 70,
        'SkinThickness': 25,
        'Insulin': 80,
        'BMI': 24.5,
        'DiabetesPedigreeFunction': 0.35,
        'Age': 25
    }

    input_data = []
    for feat in feature_labels.keys():
        input_data.append(
            st.number_input(feature_labels[feat], min_value=0.0, value=float(default_values[feat]), step=0.1)
        )

    threshold = st.slider("Ngưỡng độ tin cậy để kết luận", 0.5, 1.0, 0.7)

    if st.button("🔍 Dự đoán"):
        result = predict_diabetes(input_data, diabetes_rules, no_diabetes_rules, bin_edges, threshold)
        st.success(result)


if __name__ == "__main__":
    main()
