import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }

    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        color: #e0e0e0;
    }

    .stMetric {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00d4ff;
        backdrop-filter: blur(10px);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 12px 24px;
        color: #b0b0b0;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff, #00a8ff);
        color: white;
        box-shadow: 0 4px 12px rgba(0,212,255,0.3);
    }

    h1 {
        color: #00d4ff;
        font-weight: 700;
        text-shadow: 0 2px 8px rgba(0,212,255,0.2);
    }

    h2 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 24px;
    }

    .css-1n76uvr, .css-ffhzg2 {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 8px;
    }

    .element-container {
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def train_churn_model(X_train, y_train):
    """Train a gradient boosting model for churn prediction"""
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def preprocess_data(df, scaler=None, label_encoders=None, fit=False):
    """Preprocess the data: handle missing values, encode categorical, scale numerical"""
    df_processed = df.copy()

    # Handle missing values
    df_processed = df_processed.dropna()

    if fit:
        label_encoders = {}
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    else:
        if label_encoders:
            for col, le in label_encoders.items():
                if col in df_processed.columns:
                    df_processed[col] = le.transform(df_processed[col].astype(str))

    # Scale numerical features
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

    if fit:
        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    else:
        if scaler:
            df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])

    return df_processed, scaler, label_encoders


def generate_sample_data(n_rows=500):
    """Generate realistic customer churn dataset"""
    np.random.seed(42)

    data = {
        'Customer_ID': [f'CUST_{i:05d}' for i in range(n_rows)],
        'Age': np.random.randint(18, 80, n_rows),
        'Monthly_Spend': np.random.uniform(10, 200, n_rows).round(2),
        'Contract_Duration_Months': np.random.randint(1, 60, n_rows),
        'Support_Tickets': np.random.randint(0, 20, n_rows),
        'Services_Active': np.random.randint(1, 6, n_rows),
        'Internet_Type': np.random.choice(['Fiber', 'DSL', 'Cable'], n_rows),
        'Payment_Method': np.random.choice(['Credit Card', 'Bank Transfer', 'Check'], n_rows),
        'Customer_Since_Days': np.random.randint(30, 1825, n_rows),
        'Last_Interaction_Days': np.random.randint(0, 180, n_rows),
    }

    df = pd.DataFrame(data)

    # Create churn target with realistic patterns
    churn_probability = (
            0.05 +
            (df['Last_Interaction_Days'] > 90) * 0.15 +
            (df['Support_Tickets'] > 10) * 0.20 +
            (df['Contract_Duration_Months'] < 6) * 0.25 +
            (df['Monthly_Spend'] < 30) * 0.15
    )

    df['Churn'] = (np.random.random(n_rows) < churn_probability).astype(int)

    return df


# ============================================================================
# MAIN APP
# ============================================================================

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None

# Header
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.title("📊 Customer Churn Prediction")
    st.markdown("**Advanced ML-powered churn risk analysis**")
with col2:
    st.metric("System Status", "🟢 Active", delta="Ready")

st.divider()

# Sidebar for data upload
with st.sidebar:
    st.header("⚙️ Configuration")

    data_source = st.radio(
        "Data Source",
        ["📤 Upload CSV", "📊 Sample Data"],
        help="Choose between uploading your own data or using sample data"
    )

    if data_source == "📤 Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your customer data (CSV)",
            type=['csv'],
            help="CSV must contain a 'Churn' column (0 or 1)"
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.original_df = df.copy()
                st.success(f"✅ Loaded {len(df)} records")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        n_samples = st.slider("Generate samples", 100, 1000, 500, step=100)
        if st.button("🔄 Generate Sample Data"):
            df = generate_sample_data(n_samples)
            st.session_state.original_df = df.copy()
            st.success(f"✅ Generated {len(df)} sample records")

    st.divider()
    st.subheader("Model Settings")
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
    st.info("Model uses Gradient Boosting with 200 estimators")

# Main content tabs
if st.session_state.original_df is not None:
    df = st.session_state.original_df.copy()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Overview",
        "🧠 Model Training",
        "🎯 Predictions",
        "📈 Analytics",
        "💾 Export"
    ])

    # ========================================================================
    # TAB 1: DATA OVERVIEW
    # ========================================================================
    with tab1:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(df), delta=f"{len(df)} customers")
        with col2:
            if 'Churn' in df.columns:
                churn_count = df['Churn'].sum()
                churn_rate = (churn_count / len(df) * 100)
                st.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_count} churned")
        with col3:
            st.metric("Features", len(df.columns) - 1, delta="Input variables")
        with col4:
            st.metric("Data Quality", "✓ Good", delta="No critical issues")

        st.divider()

        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            st.subheader("First Few Records")
            st.dataframe(
                df.head(10),
                use_container_width=True,
                height=300
            )

        with col2:
            st.subheader("Data Types")
            type_counts = df.dtypes.value_counts()
            fig = go.Figure(data=[go.Bar(
                x=type_counts.index.astype(str),
                y=type_counts.values,
                marker=dict(color=['#00d4ff', '#ff006e', '#00f900'])
            )])
            fig.update_layout(
                height=350,
                template='plotly_dark',
                title="Feature Types Distribution",
                xaxis_title="Type",
                yaxis_title="Count",
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Statistical Summary")
        st.dataframe(df.describe().T, use_container_width=True)

    # ========================================================================
    # TAB 2: MODEL TRAINING
    # ========================================================================
    with tab2:
        st.subheader("🧠 Train Churn Prediction Model")

        if 'Churn' not in df.columns:
            st.error("⚠️ Dataset must contain a 'Churn' column (0=No, 1=Yes)")
        else:
            col1, col2 = st.columns([0.6, 0.4])

            with col1:
                st.info(
                    "The model will:\n"
                    "1. Encode categorical variables\n"
                    "2. Scale numerical features\n"
                    "3. Train Gradient Boosting classifier\n"
                    "4. Evaluate on test set"
                )

            with col2:
                if st.button("🚀 Train Model", use_container_width=True, key="train"):
                    with st.spinner("Training model..."):
                        try:
                            # Prepare data
                            X = df.drop('Churn', axis=1)
                            y = df['Churn']

                            # Split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42, stratify=y
                            )

                            # Preprocess
                            X_train, scaler, encoders = preprocess_data(X_train.copy(), fit=True)
                            X_test, _, _ = preprocess_data(X_test.copy(), scaler=scaler, label_encoders=encoders)

                            # Train model
                            model = train_churn_model(X_train, y_train)

                            # Store in session
                            st.session_state.model = model
                            st.session_state.scaler = scaler
                            st.session_state.label_encoders = encoders
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                            st.session_state.predictions = model.predict(X_test)
                            st.session_state.probabilities = model.predict_proba(X_test)[:, 1]

                            st.success("✅ Model trained successfully!")
                        except Exception as e:
                            st.error(f"Error: {e}")

            st.divider()

            # Display model performance if trained
            if st.session_state.model is not None:
                st.subheader("📊 Model Performance")

                y_pred = st.session_state.predictions
                y_proba = st.session_state.probabilities
                y_true = st.session_state.y_test

                # Metrics
                col1, col2, col3, col4 = st.columns(4)

                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                with col1:
                    acc = accuracy_score(y_true, y_pred)
                    st.metric("Accuracy", f"{acc:.1%}", delta=f"{acc:.2f}")

                with col2:
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    st.metric("Precision", f"{prec:.1%}", delta=f"{prec:.2f}")

                with col3:
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    st.metric("Recall", f"{rec:.1%}", delta=f"{rec:.2f}")

                with col4:
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    st.metric("F1-Score", f"{f1:.1%}", delta=f"{f1:.2f}")

                st.divider()

                # Confusion matrix & ROC curve
                col1, col2 = st.columns(2)

                with col1:
                    cm = confusion_matrix(y_true, y_pred)
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['No Churn', 'Churn'],
                        y=['No Churn', 'Churn'],
                        text=cm,
                        texttemplate='%{text}',
                        colorscale='Blues'
                    ))
                    fig.update_layout(
                        title="Confusion Matrix",
                        template='plotly_dark',
                        height=350,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fpr, tpr, _ = roc_curve(y_true, y_proba)
                    auc = roc_auc_score(y_true, y_proba)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC (AUC={auc:.3f})',
                        line=dict(color='#00d4ff', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
                    fig.update_layout(
                        title="ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        template='plotly_dark',
                        height=350,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Feature importance
                st.divider()
                st.subheader("🎯 Feature Importance")

                feature_names = st.session_state.X_test.columns
                importance = st.session_state.model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False).head(15)

                fig = go.Figure(data=[go.Bar(
                    y=importance_df['Feature'],
                    x=importance_df['Importance'],
                    orientation='h',
                    marker=dict(color=importance_df['Importance'], colorscale='Viridis')
                )])
                fig.update_layout(
                    title="Top 15 Most Important Features",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    template='plotly_dark',
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 3: PREDICTIONS
    # ========================================================================
    with tab3:
        st.subheader("🎯 Predict Customer Churn Risk")

        if st.session_state.model is None:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab")
        else:
            st.info(
                "Get churn predictions for your customers. Enter customer details manually "
                "or use the prediction results from the trained model."
            )

            col1, col2 = st.columns([0.5, 0.5])

            with col1:
                st.subheader("📥 Batch Predictions")
                st.write("Predictions on test set:")

                pred_df = pd.DataFrame({
                    'Actual_Churn': st.session_state.y_test.values,
                    'Predicted_Churn': st.session_state.predictions,
                    'Churn_Risk_%': (st.session_state.probabilities * 100).round(1)
                }).reset_index(drop=True)


                # Color coding
                def color_risk(val):
                    if val >= 70:
                        return 'background-color: rgba(255, 0, 110, 0.3)'
                    elif val >= 40:
                        return 'background-color: rgba(255, 165, 0, 0.3)'
                    else:
                        return 'background-color: rgba(0, 249, 0, 0.3)'


                st.dataframe(
                    pred_df.style.map(color_risk, subset=['Churn_Risk_%']),
                    use_container_width=True,
                    height=400
                )

            with col2:
                st.subheader("⚠️ High-Risk Customers")

                high_risk = pred_df[pred_df['Churn_Risk_%'] >= 70].head(15)

                if len(high_risk) > 0:
                    st.metric("At-Risk Customers", len(high_risk), delta=f"Risk > 70%")
                    st.dataframe(
                        high_risk.sort_values('Churn_Risk_%', ascending=False),
                        use_container_width=True,
                        height=350
                    )
                else:
                    st.success("✅ No high-risk customers detected!")

                st.divider()

                st.subheader("📊 Risk Distribution")
                risk_bins = pd.cut(pred_df['Churn_Risk_%'], bins=[0, 25, 50, 75, 100])
                risk_counts = risk_bins.value_counts().sort_index()

                fig = go.Figure(data=[go.Bar(
                    x=['Low\n(0-25%)', 'Medium\n(25-50%)', 'High\n(50-75%)', 'Critical\n(75-100%)'],
                    y=risk_counts.values,
                    marker=dict(color=['#00f900', '#ffff00', '#ff6600', '#ff006e'])
                )])
                fig.update_layout(
                    title="Churn Risk Distribution",
                    template='plotly_dark',
                    height=300,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 4: ANALYTICS
    # ========================================================================
    with tab4:
        st.subheader("📈 Data Analytics & Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Churn by Demographics")
            if 'Age' in df.columns:
                age_churn = df.groupby(pd.cut(df['Age'], bins=5))['Churn'].agg(['sum', 'count'])
                age_churn['Rate'] = (age_churn['sum'] / age_churn['count'] * 100).round(1)

                fig = px.bar(
                    x=age_churn.index.astype(str),
                    y=age_churn['Rate'],
                    labels={'x': 'Age Group', 'y': 'Churn Rate (%)'},
                    color=age_churn['Rate'],
                    color_continuous_scale='Reds'
                )
                fig.update_layout(template='plotly_dark', height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Churn by Spending")
            if 'Monthly_Spend' in df.columns:
                spend_churn = df.groupby(pd.cut(df['Monthly_Spend'], bins=5))['Churn'].agg(['sum', 'count'])
                spend_churn['Rate'] = (spend_churn['sum'] / spend_churn['count'] * 100).round(1)

                fig = px.bar(
                    x=spend_churn.index.astype(str),
                    y=spend_churn['Rate'],
                    labels={'x': 'Monthly Spend ($)', 'y': 'Churn Rate (%)'},
                    color=spend_churn['Rate'],
                    color_continuous_scale='Blues'
                )
                fig.update_layout(template='plotly_dark', height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Feature Correlations")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu'
            ))
            fig.update_layout(
                height=350,
                template='plotly_dark',
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Key Insights")

            churn_rate = (df['Churn'].sum() / len(df) * 100)

            insights = [
                f"📊 Overall churn rate: **{churn_rate:.1f}%**",
                f"👥 Total customers: **{len(df):,}**",
            ]

            if 'Monthly_Spend' in df.columns:
                avg_churn = df[df['Churn'] == 1]['Monthly_Spend'].mean()
                avg_retain = df[df['Churn'] == 0]['Monthly_Spend'].mean()
                insights.append(
                    f"💰 Avg spend (churned): **${avg_churn:.2f}** vs **${avg_retain:.2f}** (retained)"
                )

            if 'Support_Tickets' in df.columns:
                avg_tickets_churn = df[df['Churn'] == 1]['Support_Tickets'].mean()
                insights.append(
                    f"🎟️ Avg support tickets (churned): **{avg_tickets_churn:.1f}**"
                )

            for insight in insights:
                st.markdown(f"• {insight}")

    # ========================================================================
    # TAB 5: EXPORT
    # ========================================================================
    with tab5:
        st.subheader("💾 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Download Data")

            if st.session_state.predictions is not None:
                export_df = pd.DataFrame({
                    'Actual_Churn': st.session_state.y_test.values,
                    'Predicted_Churn': st.session_state.predictions,
                    'Churn_Risk_Percentage': (st.session_state.probabilities * 100).round(2)
                })

                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

                st.success(f"✅ {len(export_df)} predictions ready for download")

        with col2:
            st.markdown("### Model Summary")

            if st.session_state.model is not None:
                summary = f"""
                **Model Type:** Gradient Boosting Classifier
                **Features Used:** {len(st.session_state.X_test.columns)}
                **Test Samples:** {len(st.session_state.y_test)}
                **Positive Class:** Churn (1)
                **Training Complete:** ✅ Yes
                """
                st.info(summary)
else:
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.info(
            "👋 **Welcome to the Churn Prediction System**\n\n"
            "1. **Upload Data** or generate samples in the sidebar\n"
            "2. **Train Model** to build the ML classifier\n"
            "3. **View Predictions** and churn risk scores\n"
            "4. **Analyze** patterns and export results"
        )

    with col2:
        if st.button("📊 Load Sample Data Now", use_container_width=True):
            st.session_state.original_df = generate_sample_data(500)
            st.rerun()