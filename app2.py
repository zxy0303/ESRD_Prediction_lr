import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Clinical Decision Support System", layout="wide")
st.title("🩺 Clinical Decision Support System")


# ==========================================
# 1. 模型加载 (Model Loading)
# ==========================================
@st.cache_resource
def load_models():
    models_12 = {
        1: joblib.load('./selected_features/lr_1yr.pkl'),
        3: joblib.load('./selected_features/lr_3yr.pkl'),
        5: joblib.load('./selected_features/lr_5yr.pkl')
    }

    try:
        models_9 = {
            1: joblib.load('./nine_features/lr_1yr.pkl'),
            3: joblib.load('./nine_features/lr_3yr.pkl'),
            5: joblib.load('./nine_features/lr_5yr.pkl')
        }
    except FileNotFoundError:
        st.warning("⚠️ 未找到 9 特征模型文件，演示模式下暂时使用 12 特征模型替代。")
        models_9 = models_12

    return models_12, models_9


models_12, models_9 = load_models()


# ==========================================
# 2. 模式选择 (Mode Selection)
# ==========================================
st.markdown("### ⚙️ Settings")
model_mode = st.radio(
    "Select Feature Input Mode:",
    ("12 Features (Enriched model)", "9 Features (Simplified model)"),
    horizontal=True
)

is_full_mode = (model_mode == "12 Features (Enriched model)")

st.markdown(
    f"Current Mode: **{model_mode}**. "
    f"{'Includes all clinical features.' if is_full_mode else 'Excludes PAX2, Family History, and Prenatal Phenotype.'}"
)

left_col, right_col = st.columns([2, 2], gap="large")

cakut_subphenotype_list = {
    'renal hypodysplasia associated with puv': 1,
    'solitary kidney': 2,
    'bilateral renal hypodysplasia': 3,
    'unilateral renal hypodysplasia': 4,
    'multicystic dysplastic kidney': 5,
    'horseshoe kidney': 6,
    'others': 7
}


# ==========================================
# 3. 动态输入界面 (Dynamic UI)
# ==========================================
with left_col:
    st.subheader("🏥 Patient Characteristics")
    col1, col2 = st.columns(2, gap='medium')

    with col1:
        age_first_diagnose = st.number_input("Age At First Diagnose (yr)", min_value=0.0, max_value=18.0, value=0.0)
        gender = st.selectbox("Gender", ["Female", "Male"])

        if is_full_mode:
            family_history = st.selectbox("Family History", ["No", "Yes"])
        else:
            family_history = "No"

        ckd_stage_first_diagnose = st.selectbox("CKD Stage At First Diagnose", [1, 2, 3, 4, 5])
        short_stature = st.selectbox("Short Stature", ["No", "Yes"])
        cakut_subphenotype = st.selectbox("CAKUT Subphenotype", list(cakut_subphenotype_list.keys()))

    with col2:
        if is_full_mode:
            pax2 = st.selectbox("PAX2", ["No", "Yes"])
        else:
            pax2 = "No"

        if is_full_mode:
            prenatal_phenotype = st.selectbox("Prenatal Phenotype", ["No", "Yes"])
        else:
            prenatal_phenotype = "No"

        congenital_heart_disease = st.selectbox("Congenital Heart Disease", ["No", "Yes"])
        ocular = st.selectbox("Ocular", ["No", "Yes"])
        preterm_birth = st.selectbox("Preterm Birth", ["No", "Yes"])
        behavioral_cognitive_abnormalities = st.selectbox("Behavioral Cognitive Abnormalities", ["No", "Yes"])

    predict_btn = st.button("PREDICT", use_container_width=True)


# ==========================================
# 4. 数据构建 (Data Construction)
# ==========================================
def get_binary(val):
    return 0 if val == 'No' or val == 'Female' else 1


data_dict = {
    "gender (1/0)": [get_binary(gender)],
    "preterm_birth (1/0)": [get_binary(preterm_birth)],
    "cakut_subphenotype": [cakut_subphenotype_list[cakut_subphenotype]],
    "behavioral_cognitive_abnormalities (1/0)": [get_binary(behavioral_cognitive_abnormalities)],
    "congenital_heart_disease (1/0)": [get_binary(congenital_heart_disease)],
    "ocular (1/0)": [get_binary(ocular)],
    "age_first_diagnose": [age_first_diagnose],
    "ckd_stage_first_diagnose": [ckd_stage_first_diagnose],
    "short_stature (1/0)": [get_binary(short_stature)]
}

if is_full_mode:
    data_dict.update({
        'PAX2': [get_binary(pax2)],
        'family_history (1/0)': [get_binary(family_history)],
        'prenatal_phenotype (1/0)': [get_binary(prenatal_phenotype)]
    })

input_data = pd.DataFrame(data_dict)


# ==========================================
# 5. 预测逻辑 (Prediction Logic)
# ==========================================
def predict_probability(model, input_df):
    input_df = input_df.copy()

    model_features = None
    if hasattr(model, 'feature_names_'):
        model_features = model.feature_names_
    elif hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    elif hasattr(model, 'steps'):
        try:
            final_estimator = model.steps[-1][1]
            if hasattr(final_estimator, 'feature_names_'):
                model_features = final_estimator.feature_names_
            elif hasattr(final_estimator, 'feature_names_in_'):
                model_features = final_estimator.feature_names_in_
        except:
            pass

    if model_features is not None:
        missing_cols = set(model_features) - set(input_df.columns)
        if missing_cols:
            for c in missing_cols:
                input_df[c] = 0
        input_df = input_df[list(model_features)]

    prob = model.predict_proba(input_df)[0][1]
    return float(prob)


# ==========================================
# 6. 结果展示 (Results)
# ==========================================
with right_col:
    st.subheader("🤖 Predicted Results")

    if predict_btn:
        try:
            current_models = models_12 if is_full_mode else models_9

            prob_1 = predict_probability(current_models[1], input_data)
            prob_3 = predict_probability(current_models[3], input_data)
            prob_5 = predict_probability(current_models[5], input_data)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    label="1-Year Risk",
                    value=f"{prob_1:.2%}"
                )

            with c2:
                st.metric(
                    label="3-Year Risk",
                    value=f"{prob_3:.2%}"
                )

            with c3:
                st.metric(
                    label="5-Year Risk",
                    value=f"{prob_5:.2%}"
                )

            st.markdown("---")
            st.write("Input features used for prediction:")
            st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.write("Current Input Columns:", input_data.columns.tolist())
    else:
        st.info("Please enter patient characteristics and click **PREDICT**.")
