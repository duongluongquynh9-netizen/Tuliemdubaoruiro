import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# ==============================================================================
# CẤU HÌNH TRANG (BẮT BUỘC ĐỂ ĐẦU TIÊN)
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# Thêm CSS tùy chỉnh để định dạng In đậm và Nền Tím cho các giá trị dữ liệu Metric
st.markdown("""
    <style>
    /* Tùy chỉnh màu nền và in đậm cho các số liệu Metric */
    [data-testid="stMetricValue"] {
        background-color: #8e44ad; /* Màu tím */
        color: white;
        font-weight: 900 !important; /* In đậm */
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        display: inline-block;
        width: 100%;
    }
    [data-testid="stMetricLabel"] {
        font-weight: bold;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# CÁC HÀM TRỢ GIÚP & CACHE DỮ LIỆU
# ==============================================================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bộ nhớ bytes một cách an toàn"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# ==============================================================================
# THANH BÊN (SIDEBAR) - CẤU HÌNH & TẢI DỮ LIỆU
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải tập dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tập dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu chứa các biến đặc trưng (X) và biến mục tiêu 'default' (y)"
    )
    
    st.divider()
    
    # 2. Lựa chọn mô hình thuật toán (Đồng bộ theo Notebook)
    st.subheader("🤖 Thuật toán mô hình")
    model_choice = st.selectbox(
        "Chọn thuật toán huấn luyện:",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        index=0,
        help="Chọn thuật toán để xây dựng mô hình phân loại giao dịch gian lận."
    )
    
    # 3. Cấu hình tham số động (Đồng bộ giá trị random_state = 32 chuẩn theo notebook)
    st.subheader("🛠️ Tham số mô hình AI")
    params = {}
    
    if model_choice == "Random Forest":
        params['n_estimators'] = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=300, value=100, step=10)
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=10, step=1)
        params['random_state'] = st.number_input("Hạt giống ngẫu nhiên (random_state)", value=32, step=1)
        
    elif model_choice == "Decision Tree":
        params['criterion'] = st.selectbox("Tiêu chí phân tách (criterion)", options=["gini", "entropy"], index=0)
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=10, step=1)
        params['random_state'] = st.number_input("Hạt giống ngẫu nhiên (random_state)", value=32, step=1)
        
    elif model_choice == "Logistic Regression":
        params['C'] = st.slider("Hệ số điều hòa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
        params['max_iter'] = st.number_input("Số vòng lặp tối đa (max_iter)", value=200, step=50)
        params['random_state'] = st.number_input("Hạt giống ngẫu nhiên (random_state)", value=32, step=1)

    st.divider()
    
    # 4. Nút kích hoạt huấn luyện mô hình
    train_clicked = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==============================================================================
# GIAO DIỆN CHÍNH (MAIN PANEL)
# ==============================================================================
st.title("🛡️ Hệ thống Dự báo Rủi ro & Phát hiện Gian lận Giao dịch")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch, đánh giá hành vi bất thường và cảnh báo rủi ro gian lận.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tập dữ liệu (.csv hoặc .xlsx) tại thanh Sidebar bên trái để bắt đầu sử dụng ứng dụng.")
    st.stop()

# Đọc dữ liệu thô từ file đã upload
file_bytes = uploaded_file.getvalue()
df_raw = load_data(file_bytes, uploaded_file.name)

if df_raw is None:
    st.error("❌ Định dạng file không hợp lệ hoặc dữ liệu bị lỗi. Vui lòng kiểm tra lại.")
    st.stop()

st.caption(f"📁 Đang sử dụng tệp tin dữ liệu: **{uploaded_file.name}**")
st.divider()

# Xác định danh sách biến tự động dựa trên cấu trúc bảng dữ liệu
target_col = 'default'
if target_col not in df_raw.columns:
    st.error(f"❌ Không tìm thấy cột mục tiêu '{target_col}' trong file dữ liệu vừa tải lên.")
    st.stop()

feature_cols = [col for col in df_raw.columns if col != target_col]

# ==============================================================================
# KHỐI XỬ LÝ HUẤN LUYỆN (TÁI HIỆN CHÍNH XÁC PIPELINE NOTEBOOK - KHÔNG DÙNG SCALER)
# ==============================================================================
if train_clicked:
    with st.spinner("🔄 Đang xử lý huấn luyện mô hình trực tiếp trên dữ liệu thô..."):
        X = df_raw[feature_cols]
        y = df_raw[target_col]
        
        # Xử lý khuyết thiếu cơ bản phòng ngừa lỗi tập dữ liệu
        X = X.fillna(X.median())
        
        # Chia tập dữ liệu chuẩn xác theo tham số Notebook (test_size=0.2, random_state=32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=int(params.get('random_state', 32))
        )
        
        # Khởi tạo mô hình tương ứng
        if model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=int(params.get('random_state', 32))
            )
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(
                criterion=params.get('criterion', 'gini'),
                max_depth=params.get('max_depth', 10),
                random_state=int(params.get('random_state', 32))
            )
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(
                C=params.get('C', 1.0), 
                max_iter=int(params.get('max_iter', 200)),
                random_state=int(params.get('random_state', 32))
            )
        
        # Huấn luyện trực tiếp trên tập train thô (Đúng theo tinh thần notebook gốc)
        model.fit(X_train, y_train)
        
        # Dự đoán đánh giá kết quả trên tập kiểm thử mẫu
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Lưu kết quả kiểm định vào cấu trúc từ điển mẫu
        metrics_results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "cm": confusion_matrix(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        
        # Đồng bộ lưu trữ trạng thái vào session_state của ứng dụng
        st.session_state['trained_model'] = model
        st.session_state['metrics'] = metrics_results
        st.session_state['feature_names'] = feature_cols
        st.session_state['model_name'] = model_choice
        
    st.success(f"🎉 Huấn luyện thành công mô hình {model_choice}!")

# ==============================================================================
# HỆ THỐNG PHÂN CHIA TABS CHỨC NĂNG CHÍNH
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả & Kiểm định", 
    "🔮 Dự báo thực tế"
])

# ------------------------------------------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("📋 Phân tích Thống kê Dữ liệu Thô")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Số lượng dòng giao dịch", f"{df_raw.shape[0]:,}")
    with col_m2:
        st.metric("Số lượng cột đặc trưng", f"{df_raw.shape[1]}")
    with col_m3:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.metric("Dung lượng tệp tin", f"{file_size_mb:.2f} MB")
        
    st.write("##### 🔍 Xem trước 5 dòng dữ liệu đầu tiên:")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.write("##### 📊 Thống kê mô tả tổng thể các biến mô hình:")
    st.dataframe(df_raw[feature_cols + [target_col]].describe(), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Phân bổ dữ liệu & Quan hệ đặc trưng")
    
    # Tạo bản sao và ép kiểu cột mục tiêu sang dạng chuỗi (String) để tránh lỗi Plotly Color Map
    df_plot = df_raw.copy()
    df_plot[target_col] = df_plot[target_col].astype(str).map({'0': 'Hợp lệ (0)', '1': 'Gian lận (1)'})
    
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        st.write("**Phân phối tỷ lệ Biến mục tiêu (Gian lận - default)**")
        target_counts = df_plot[target_col].value_counts().reset_index()
        target_counts.columns = ['Trạng thái', 'Số lượng']
        fig_target = px.bar(target_counts, x='Trạng thái', y='Số lượng', color='Trạng thái', 
                            color_discrete_map={'Hợp lệ (0)': '#2ecc71', 'Gian lận (1)': '#e74c3c'},
                            height=350)
        st.plotly_chart(fig_target, use_container_width=True)
        
    with col_v2:
        st.write("**Lựa chọn biến phân tích tần suất nâng cao**")
        selected_features = st.multiselect(
            "Chọn các biến đặc trưng để hiển thị biểu đồ phân bổ tần suất:",
            options=feature_cols,
            default=feature_cols[:2] if len(feature_cols) >= 2 else feature_cols
        )
        
    if selected_features:
        st.write("**Lưới phân tích biểu đồ mật độ đặc trưng**")
        grid_cols = st.columns(2)
        for idx, feat in enumerate(selected_features):
            with grid_cols[idx % 2]:
                fig_feat = px.histogram(df_plot, x=feat, color=target_col, barmode='overlay',
                                        title=f"Mật độ phân bổ đặc trưng của biến: {feat}",
                                        color_discrete_map={'Hợp lệ (0)': '#3498db', 'Gian lận (1)': '#e67e22'},
                                        height=300)
                st.plotly_chart(fig_feat, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🎯 Đánh giá Hiệu năng Mô hình Kiểm định")
    
    if 'metrics' not in st.session_state:
        st.info("⚠️ Vui lòng quay lại bảng điều khiển Sidebar bên trái và ấn nút 'Huấn luyện mô hình' để xem kết quả đánh giá.")
    else:
        res = st.session_state['metrics']
        current_model_name = st.session_state['model_name']
        
        st.write(f"⚙️ Thuật toán hiện tại đang đánh giá: **{current_model_name}**")
        
        c_acc, c_pre, c_rec, c_f1 = st.columns(4)
        c_acc.metric("Độ chính xác (Accuracy)", f"{res['accuracy']:.4f}")
        c_pre.metric("Độ xác thực (Precision)", f"{res['precision']:.4f}")
        c_rec.metric("Độ nhạy thu hồi (Recall)", f"{res['recall']:.4f}")
        c_f1.metric("F1-Score", f"{res['f1']:.4f}")
        
        st.divider()
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.write("**Ma trận nhầm lẫn (Confusion Matrix)**")
            cm_labels = ['Hợp lệ (0)', 'Gian lận (1)']
            fig_cm = px.imshow(res['cm'], text_auto=True, x=cm_labels, y=cm_labels,
                               labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế"),
                               color_continuous_scale="Purples", height=400) # Đổi màu ma trận nhầm lẫn sang tông tím để đồng bộ
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_g2:
            st.write("**Đường cong đặc trưng động học ROC**")
            if res['y_proba'] is not None:
                fpr, tpr, thresholds = roc_curve(res['y_test'], res['y_proba'])
                roc_auc = auc(fpr, tpr)
                
                fig_roc = px.line(x=fpr, y=tpr, title=f'Đường cong ROC (AUC = {roc_auc:.4f})',
                                  labels=dict(x='Tỷ lệ báo động sai (FPR)', y='Tỷ lệ nhận diện đúng (TPR)'),
                                  height=400)
                fig_roc.add_shape(type='line', line=dict(dash='dash', color='red'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.warning("Mô hình được chọn hiện tại không hỗ trợ hàm tính toán xác suất đầu ra dự báo.")

# ------------------------------------------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO THỰC TẾ
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🔮 Ứng dụng Mô hình Chấm điểm Dự báo Thực tế")
    
    if 'trained_model' not in st.session_state:
        st.info("⚠️ Vui lòng huấn luyện mô hình ở Sidebar trước khi sử dụng chức năng dự báo thực tế.")
    else:
        model = st.session_state['trained_model']
        feature_names = st.session_state['feature_names']
        
        mode = st.radio("Chọn phương thức nhập dữ liệu đầu vào:", 
                        options=["Nhập chỉ số trực tiếp từ Form", "Tải tệp danh sách cần chấm điểm hàng loạt"],
                        horizontal=True)
        
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP TỪ FORM
        if mode == "Nhập chỉ số trực tiếp từ Form":
            st.write("✍️ Điền thông số chi tiết của giao dịch đơn lẻ cần thẩm định:")
            
            with st.form("single_prediction_form"):
                form_cols = st.columns(3)
                input_data = {}
                
                for idx, feat in enumerate(feature_names):
                    default_val = float(df_raw[feat].median())
                    
                    with form_cols[idx % 3]:
                        input_data[feat] = st.number_input(
                            f"{feat}",
                            value=default_val,
                            step=0.01,
                            format="%.4f"
                        )
                        
                submit_predict = st.form_submit_button("🔍 Tiến hành Dự báo", type="primary")
                
                if submit_predict:
                    df_single = pd.DataFrame([input_data])[feature_names]
                    
                    # Dự đoán trực tiếp tương thích chuẩn 100% notebook gốc
                    pred_class = model.predict(df_single)[0]
                    pred_prob = model.predict_proba(df_single)[0][1] if hasattr(model, "predict_proba") else None
                    
                    st.divider()
                    st.write("#### 📝 Kết quả phân tích hành vi giao dịch:")
                    if pred_class == 1:
                        st.error(f"🚨 CẢNH BÁO: Giao dịch có nguy cơ GIAN LẬN hoặc RỦI RO CAO (default = 1).")
                    else:
                        st.success(f"✅ AN TOÀN: Giao dịch được thẩm định ở mức độ HỢP LỆ (default = 0).")
                        
                    if pred_prob is not None:
                        st.info(f"📊 Xác suất xảy ra rủi ro gian lận phân tích từ mô hình: **{pred_prob*100:.2f}%**")

        # CHẾ ĐỘ 2 — TẢI FILE DANH SÁCH CHẤM ĐIỂM HÀNG LOẠT
        elif mode == "Tải tệp danh sách cần chấm điểm hàng loạt":
            st.write("📂 Tải lên file chứa cấu trúc các cột đặc trưng tương tự (Không bắt buộc phải có cột 'default').")
            
            batch_file = st.file_uploader("Tải tệp danh sách cần dự báo (.csv, .xlsx)", type=["csv", "xlsx"], key="batch_uploader")
            
            if batch_file is not None:
                df_batch = load_data(batch_file.getvalue(), batch_file.name)
                
                if df_batch is not None:
                    missing_cols = [col for col in feature_names if col not in df_batch.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Tệp tải lên thiếu các cột đặc trưng quan trọng sau: {missing_cols}")
                    else:
                        df_batch_features = df_batch[feature_names].fillna(df_raw[feature_names].median())
                        
                        # Dự báo trực tiếp trên luồng thô đồng bộ notebook
                        batch_preds = model.predict(df_batch_features)
                        df_batch['Dự_Báo_Kết_Quả'] = batch_preds
                        df_batch['Trạng_Thái_Rủi_Ro'] = df_batch['Dự_Báo_Kết_Quả'].map({0: 'Hợp lệ', 1: 'Cảnh báo gian lận'})
                        
                        if hasattr(model, "predict_proba"):
                            df_batch['Xác_Suất_Rủi_Ro'] = model.predict_proba(df_batch_features)[:, 1]
                        
                        st.write("##### 🎉 Danh sách kết quả dự báo vừa được xử lý thành công:")
                        st.dataframe(df_batch, use_container_width=True)
                        
                        # Xuất file kết quả đầu ra
                        csv_buffer = io.StringIO()
                        df_batch.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Tải xuống tệp kết quả dự báo toàn bộ (.CSV)",
                            data=csv_data,
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv"
                        )
