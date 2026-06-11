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
# CẤU HÌNH TRANG ĐẦU TIÊN (MẮT XÍCH BẮT BUỘC)
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# ==============================================================================
# CÁC HÀM CACHE DÙNG CHUNG
# ==============================================================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bộ nhớ bytes để tối ưu hóa hiệu năng và tránh lỗi hash"""
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
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải dữ liệu mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tập dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu chứa các biến đặc trưng (X) và biến mục tiêu 'default' (y)"
    )
    
    st.divider()
    
    # 2. Lựa chọn mô hình thuật toán (Từ notebook)
    st.subheader("🤖 Thuật toán mô hình")
    model_choice = st.selectbox(
        "Chọn thuật toán huấn luyện:",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        index=0,
        help="Chọn thuật toán để xây dựng mô hình phân loại giao dịch gian lận."
    )
    
    # 3. Cấu hình tham số động theo mô hình (Đã đồng bộ giá trị mặc định = 32 từ notebook)
    st.subheader("🛠️ Tham số mô hình AI")
    params = {}
    
    if model_choice == "Random Forest":
        params['n_estimators'] = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=300, value=100, step=10, help="Số lượng cây quyết định trong rừng.")
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=10, help="Độ sâu tối đa của mỗi cây quyết định.")
        params['random_state'] = st.number_input("Hạt giống ngẫu nhiên (random_state)", value=32, step=1, help="Đảm bảo tính nhất quán chuẩn theo notebook.")
        
    elif model_choice == "Decision Tree":
        params['criterion'] = st.selectbox("Tiêu chí phân tách (criterion)", options=["gini", "entropy"], index=0, help="Hàm đo lường chất lượng phân tách.")
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=10, help="Độ sâu tối đa của cây.")
        params['random_state'] = st.number_input("Hạt giống ngẫu nhiên (random_state)", value=32, step=1)
        
    elif model_choice == "Logistic Regression":
        params['C'] = st.slider("Hệ số điều hòa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1, help="Nghịch đảo của cường độ điều hòa.")
        params['max_iter'] = st.number_input("Số vòng lặp tối đa (max_iter)", value=200, step=50, help="Số vòng lặp tối đa cho các thuật toán hội tụ.")
        params['random_state'] = st.number_input("Hạt giống ngẫu nhiên (random_state)", value=32, step=1)

    st.divider()
    
    # 4. Nút bấm kích hoạt huấn luyện duy nhất
    train_clicked = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True, help="Bấm để bắt đầu huấn luyện quy trình dựa trên dữ liệu thô.")

# ==============================================================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==============================================================================
st.title("🛡️ Hệ thống Dự báo Rủi ro & Phát hiện Gian lận Giao dịch")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch, đánh giá hành vi bất thường và cảnh báo rủi ro gian lận (Đã đồng bộ hóa chuẩn xác pipeline từ Notebook gốc).")

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tập dữ liệu (.csv hoặc .xlsx) tại thanh Sidebar bên trái để bắt đầu sử dụng ứng dụng.")
    st.stop()

# Đọc dữ liệu khi file đã được tải
file_bytes = uploaded_file.getvalue()
df_raw = load_data(file_bytes, uploaded_file.name)

if df_raw is None:
    st.error("❌ Định dạng file không hợp lệ hoặc dữ liệu bị lỗi. Vui lòng kiểm tra lại.")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.divider()

# Xác định danh sách biến tự động dựa trên cấu trúc dữ liệu đã đọc
target_col = 'default'
if target_col not in df_raw.columns:
    st.error(f"❌ Không tìm thấy cột mục tiêu '{target_col}' trong file dữ liệu.")
    st.stop()

feature_cols = [col for col in df_raw.columns if col != target_col]

# ==============================================================================
# KHỐI XỬ LÝ HUẤN LUYỆN (TÁI HIỆN CHÍNH XÁC PIPELINE NOTEBOOK - KHÔNG SỬ DỤNG SCALER)
# ==============================================================================
if train_clicked:
    with st.spinner("🔄 Đang xử lý huấn luyện mô hình trực tiếp trên dữ liệu thô..."):
        X = df_raw[feature_cols]
        y = df_raw[target_col]
        
        # Điền giá trị thiếu cơ bản nếu có để tránh crash mô hình
        X = X.fillna(X.median())
        
        # Chia tập dữ liệu huấn luyện và kiểm thử chuẩn xác theo thông số Notebook (test_size=0.2, random_state=32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=params.get('random_state', 32)
        )
        
        # Khởi tạo mô hình theo lựa chọn
        if model_choice == "Random Forest":
            model = RandomForestClassifier(**params)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(**params)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(
                random_state=params.get('random_state', 32), 
                C=params.get('C', 1.0), 
                max_iter=int(params.get('max_iter', 200))
            )
        
        # Huấn luyện trực tiếp trên tập train thô (Đúng theo notebook)
        model.fit(X_train, y_train)
        
        # Dự đoán đánh giá kết quả mẫu kiểm thử
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Tính toán chỉ tiêu đo lường hiệu năng
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
        
        # Lưu trữ các thành phần cốt lõi vào session_state
        st.session_state['trained_model'] = model
        st.session_state['metrics'] = metrics_results
        st.session_state['feature_names'] = feature_cols
        st.session_state['model_name'] = model_choice
        
    st.success(f"🎉 Huấn luyện thành công mô hình {model_choice}!")

# ==============================================================================
# GIAO DIỆN CHÍNH PHÂN CHIA CÁC TABS CHỨC NĂNG
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
    
    st.write("##### 📊 Thống kê mô tả các biến mô hình:")
    st.dataframe(df_raw[feature_cols + [target_col]].describe(), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU (ĐÃ SỬA LỖI ĐỊNH DẠNG MÀU SẮC BIẾN MỤC TIÊU)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Phân bổ dữ liệu & Quan hệ đặc trưng")
    
    # Tạo bản sao để chuyển biến mục tiêu thành chuỗi phục vụ vẽ biểu đồ chuẩn xác
    df_plot = df_raw.copy()
    df_plot[target_col] = df_plot[target_col].map({0: 'Hợp lệ (0)', 1: 'Gian lận (1)'})
    
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
        st.write("**Lựa chọn biến phân tích chi tiết nâng cao**")
        selected_features = st.multiselect(
            "Chọn các biến đặc trưng để hiển thị biểu đồ phân bổ tần suất:",
            options=feature_cols,
            default=feature_cols[:2] if len(feature_cols) >= 2 else feature_cols
        )
        
    if selected_features:
        st.write("**Lưới phân tích biểu đồ trực quan hóa**")
        grid_cols = st.columns(2)
        for idx, feat in enumerate(selected_features):
            with grid_cols[idx % 2]:
                fig_feat = px.histogram(df_plot, x=feat, color=target_col, barmode='overlay',
                                        title=f"Phân phối tần suất đặc trưng của: {feat}",
                                        color_discrete_map={'Hợp lệ (0)': '#3498db', 'Gian lận (1)': '#e67e22'},
                                        height=300)
                st.plotly_chart(fig_feat, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🎯 Đánh giá Hiệu năng Mô hình Kiểm định")
    
    if 'metrics' not in st.session_state:
        st.info("⚠️ Vui lòng quay lại bảng điều khiển Sidebar bên trái và ấn nút 'Huấn luyện mô hình' để xem kết quả chi tiết phân tích.")
    else:
        res = st.session_state['metrics']
        current_model_name = st.session_state['model_name']
        
        st.write(f"⚙️ Thuật toán hiện tại đang định giá: **{current_model_name}**")
        
        c_acc, c_pre, c_rec, c_f1 = st.columns(4)
        c_acc.metric("Độ chính xác (Accuracy)", f"{res['accuracy']:.4f}")
        c_pre.metric("Độ xác thực chính xác (Precision)", f"{res['precision']:.4f}")
        c_rec.metric("Độ nhạy thu hồi (Recall)", f"{res['recall']:.4f}")
        c_f1.metric("F1-Score", f"{res['f1']:.4f}")
        
        st.divider()
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.write("**Ma trận nhầm lẫn (Confusion Matrix)**")
            cm_labels = ['Hợp lệ (0)', 'Gian lận (1)']
            fig_cm = px.imshow(res['cm'], text_auto=True, x=cm_labels, y=cm_labels,
                               labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế"),
                               color_continuous_scale="Blues", height=400)
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
# TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO THỰC TẾ (ĐÃ SỬA LỖI GIỚI HẠN FORM)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🔮 Ứng dụng Mô hình Chấm điểm Dự báo")
    
    if 'trained_model' not in st.session_state:
        st.info("⚠️ Vui lòng huấn luyện mô hình ở Sidebar trước khi sử dụng chức năng dự báo.")
    else:
        model = st.session_state['trained_model']
        feature_names = st.session_state['feature_names']
        
        mode = st.radio("Chọn phương thức nhập dữ liệu đầu vào:", 
                        options=["Nhập chỉ số trực tiếp từ Form", "Tải tệp danh sách cần chấm điểm hàng loạt"],
                        horizontal=True)
        
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP TỪ FORM (Mở rộng bounds để tránh lỗi StreamlitAPIException)
        if mode == "Nhập chỉ số trực tiếp từ Form":
            st.write("✍️ Điền thông số chi tiết của giao dịch cần kiểm tra:")
            
            with st.form("single_prediction_form"):
                form_cols = st.columns(3)
                input_data = {}
                
                for idx, feat in enumerate(feature_names):
                    default_val = float(df_raw[feat].median())
                    
                    with form_cols[idx % 3]:
                        # Loại bỏ giới hạn nghiêm ngặt bằng min/max động để tránh xung đột kiểu dữ liệu
                        input_data[feat] = st.number_input(
                            f"{feat}",
                            value=default_val,
                            format="%.4f"
                        )
                        
                submit_predict = st.form_submit_button("🔍 Tiến hành Dự báo", type="primary")
                
                if submit_predict:
                    df_single = pd.DataFrame([input_data])[feature_names]
                    
                    # Dự đoán trực tiếp không qua scaler tương thích chuẩn 100% notebook
                    pred_class = model.predict(df_single)[0]
                    pred_prob = model.predict_proba(df_single)[0][1] if hasattr(model, "predict_proba") else None
                    
                    st.divider()
                    st.write("#### 📝 Kết quả phân tích hành vi:")
                    if pred_class == 1:
                        st.error(f"🚨 CẢNH BÁO: Giao dịch có nguy cơ GIAN LẬN hoặc RỦI RO CAO (default = 1).")
                    else:
                        st.success(f"✅ AN TOÀN: Giao dịch được thẩm định ở mức độ HỢP LỆ (default = 0).")
                        
                    if pred_prob is not None:
                        st.info(f"📊 Xác suất xảy ra rủi ro gian lận phân tích từ mô hình: **{pred_prob*100:.2f}%**")

        # CHẾ ĐỘ 2 — TẢI FILE THEO CẤU TRÚC ĐỂ CHẤM ĐIỂM HÀNG LOẠT
        elif mode == "Tải tệp danh sách cần chấm điểm hàng loạt":
            st.write("📂 Tải lên file chứa cấu trúc các cột đặc trưng tương tự (Không nhất thiết cần cột 'default').")
            
            batch_file = st.file_uploader("Tải tệp danh sách cần dự báo (.csv, .xlsx)", type=["csv", "xlsx"], key="batch_uploader")
            
            if batch_file is not None:
                df_batch = load_data(batch_file.getvalue(), batch_file.name)
                
                if df_batch is not None:
                    missing_cols = [col for col in feature_names if col not in df_batch.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Tệp tải lên thiếu các cột đặc trưng quan trọng sau: {missing_cols}")
                    else:
                        df_batch_features = df_batch[feature_names].fillna(df_raw[feature_names].median())
                        
                        # Dự báo trực tiếp trên luồng thô
                        batch_preds = model.predict(df_batch_features)
                        df_batch['Dự_Báo_Kết_Quả'] = batch_preds
                        df_batch['Trạng_Thái_Rủi_Ro'] = df_batch['Dự_Báo_Kết_Quả'].map({0: 'Hợp lệ', 1: 'Cảnh báo gian lận'})
                        
                        if hasattr(model, "predict_proba"):
                            df_batch['Xác_Suất_Rủi_Ro'] = model.predict_proba(df_batch_features)[:, 1]
                        
                        st.write("##### 🎉 Danh sách kết quả dự báo vừa được xử lý thành công:")
                        st.dataframe(df_batch, use_container_width=True)
                        
                        csv_buffer = io.StringIO()
                        df_batch.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Tải xuống tệp kết quả dự báo toàn bộ (.CSV)",
                            data=csv_data,
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv"
                        )
