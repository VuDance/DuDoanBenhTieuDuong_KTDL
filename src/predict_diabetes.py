import pandas as pd
import numpy as np
from apyori import apriori
import streamlit as st

# Bước 1: Load và xử lý dữ liệu
@st.cache_data  # Cache để tăng tốc
def load_and_preprocess_data():
    # Load dataset (giả sử file diabetes.csv đã tải từ Kaggle)
    df = pd.read_csv('data/diabetes.csv')
    
    # Thay thế 0 ở các cột y tế bằng NaN rồi fill median (xử lý missing values ngầm)
    medical_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[medical_cols] = df[medical_cols].replace(0, np.nan)
    df[medical_cols] = df[medical_cols].fillna(df[medical_cols].median())
    
    # Rời rạc hóa các đặc trưng (3 bins: low, medium, high)
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for feature in features:
        df[f'{feature}_discretized'] = pd.cut(df[feature], bins=3, labels=['low', 'medium', 'high'], duplicates='drop')
    
    # Rời rạc hóa Outcome
    df['Outcome_discretized'] = df['Outcome'].map({0: 'No_Diabetes', 1: 'Diabetes'})
    
    # Tạo transactions: Mỗi row là list các item (tên đặc trưng + giá trị)
    transactions = []
    for _, row in df.iterrows():
        transaction = [f'{feat}_{row[f"{feat}_discretized"]}' for feat in features]
        transaction.append(row['Outcome_discretized'])
        transactions.append(transaction)
    
    return transactions, df

# Bước 2: Áp dụng Apriori để khai phá rules
@st.cache_data
def mine_rules(transactions):
    # Chạy Apriori: min_support=0.1, min_confidence=0.5, min_lift=1.0
    rules = apriori(transactions, min_support=0.1, min_confidence=0.5, min_lift=1.0, min_length=2)
    
    # Chuyển rules thành DataFrame để dễ xem
    results = []
    for rule in rules:
        # rule là một RelationRecord, chứa items, support và ordered_statistics
        support = rule.support
        for ordered_stat in rule.ordered_statistics:
            antecedent = ', '.join(list(ordered_stat.items_base))
            consequent = ', '.join(list(ordered_stat.items_add))
            confidence = ordered_stat.confidence
            lift = ordered_stat.lift
            results.append([antecedent, consequent, support, confidence, lift])
    
    rules_df = pd.DataFrame(results, columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
    # Lọc rules liên quan đến Diabetes
    diabetes_rules = rules_df[rules_df['Consequent'] == 'Diabetes']
    return diabetes_rules.sort_values('Confidence', ascending=False)

# Bước 3: Hàm dự đoán dựa trên rules
def predict_diabetes(input_data, diabetes_rules, threshold=0.7):
    discretized_input = {}
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for i, feat in enumerate(features):
        # Use pd.cut and extract the first category directly
        discretized_input[f'{feat}_discretized'] = pd.cut([input_data[i]], bins=3, labels=['low', 'medium', 'high'])[0]
    
    input_transaction = [f'{feat}_{discretized_input[f"{feat}_discretized"]}' for feat in features]
    
    matching_rules = []
    for _, rule in diabetes_rules.iterrows():
        antecedent_items = rule['Antecedent'].split(', ')
        if all(item in input_transaction for item in antecedent_items):
            matching_rules.append(rule['Confidence'])
    
    if matching_rules:
        max_conf = max(matching_rules)
        if max_conf >= threshold:
            return f"Có nguy cơ tiểu đường"
        else:
            return f"Nguy cơ thấp tiểu đường"
    else:
        return "Không phát hiện nguy cơ"

# Streamlit App
def main():
    st.title("Ứng Dụng Chẩn Đoán Bệnh Tiểu Đường Với Apriori")
    st.write("Dựa trên dataset Pima Indians Diabetes. Nhập dữ liệu để dự đoán nguy cơ tiểu đường.")

    # Load data và rules
    with st.spinner('Đang xử lý dữ liệu và khai phá rules...'):
        transactions, df = load_and_preprocess_data()
        diabetes_rules = mine_rules(transactions)

    # Hiển thị thống kê dataset
    st.subheader("Thống Kê Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Số bệnh nhân", len(df))
    col2.metric("Tỷ lệ tiểu đường", f"{df['Outcome'].mean():.1%}")
    col3.metric("Số rules Diabetes", len(diabetes_rules))

    # Hiển thị top rules
    st.subheader("Top 5 Rules Liên Quan Đến Tiểu Đường (Confidence cao nhất)")
    st.dataframe(diabetes_rules.head(5))

    # Input form cho dự đoán
    st.subheader("Nhập Thông Tin Bệnh Nhân")
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
    default_values = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    input_data = []
    for feat in feature_labels.keys():
        input_data.append(st.number_input(f"{feature_labels[feat]}", min_value=0.0, value=float(default_values[feat]), step=0.1))

    threshold = st.slider("Ngưỡng độ tin cậy để dự đoán", 0.5, 1.0, 0.56)

    if st.button("Dự đoán"):
        prediction = predict_diabetes(input_data, diabetes_rules, threshold)
        st.success(prediction)

if __name__ == "__main__":
    main()