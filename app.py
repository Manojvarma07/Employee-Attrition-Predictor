import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="HR Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Default dataset path
DEFAULT_DATA_PATH = r"C:\Users\Dell\OneDrive\Desktop\employeeattrtionpredictor\WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Cache functions for performance
@st.cache_data
def load_and_preprocess_data(uploaded_file=None):
    """Load and preprocess the dataset - EXACTLY matching Colab code"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            if os.path.exists(DEFAULT_DATA_PATH):
                df = pd.read_csv(DEFAULT_DATA_PATH)
            else:
                st.error(f"Default dataset not found at: {DEFAULT_DATA_PATH}")
                return None
        
        required_cols = ['OverTime', 'MonthlyIncome', 'JobSatisfaction', 
                         'EnvironmentSatisfaction', 'YearsAtCompany', 
                         'TotalWorkingYears', 'JobInvolvement', 
                         'WorkLifeBalance', 'BusinessTravel', 'Attrition']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        features = [
            'OverTime', 'MonthlyIncome', 'JobSatisfaction', 'EnvironmentSatisfaction',
            'YearsAtCompany', 'TotalWorkingYears', 'JobInvolvement', 'WorkLifeBalance',
            'BusinessTravel', 'Attrition'
        ]
        
        df_orig = df.copy()
        df = df[features].copy()
        
        # CRITICAL: Use SINGLE LabelEncoder instance (matches Colab exactly)
        le = LabelEncoder()
        le_dict = {}
        for col in ['OverTime', 'BusinessTravel', 'Attrition']:
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
        
        # Feature engineering
        df['Income_Satisfaction'] = df['MonthlyIncome'] * df['JobSatisfaction']
        df['Tenure_Balance'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
        df['Overload_Score'] = df['OverTime'] * df['WorkLifeBalance']
        
        # MATCH COLAB: No duplicates parameter
        df['Income_Bracket'] = pd.qcut(df['MonthlyIncome'], q=4, labels=False)
        
        return df, df_orig, le_dict
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_resource
def train_model(df):
    """Train the SVM model - EXACTLY matching Colab code"""
    try:
        df = df.reset_index(drop=True).copy()
        
        # Apply Yeo-Johnson Power Transformation
        num_cols = ['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears',
                    'Income_Satisfaction', 'Tenure_Balance', 'Overload_Score']
        
        pt = PowerTransformer(method='yeo-johnson')
        df[num_cols] = pt.fit_transform(df[num_cols])
        
        # Polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_feats = poly.fit_transform(df[['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears']])
        poly_feat_names = poly.get_feature_names_out(['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears'])
        df_poly = pd.DataFrame(poly_feats, columns=poly_feat_names, index=df.index)
        
        # Drop and merge
        df.drop(columns=['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears'], inplace=True)
        df = pd.concat([df, df_poly], axis=1)
        
        # Prepare data
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']
        
        feature_names = X.columns.tolist()
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Balance with SMOTE
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X_scaled, y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
        )
        
        # EXACT MATCH TO COLAB: GridSearchCV with same param_grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf']
        }
        
        # GridSearchCV (matches Colab)
        grid_search = GridSearchCV(
            SVC(probability=True),
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            verbose=0,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Get predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        return {
            'model': best_model,
            'scaler': scaler,
            'pt': pt,
            'poly': poly,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'X_bal': X_bal,
            'y_bal': y_bal,
            'feature_names': feature_names,
            'best_params': best_params,
            'best_cv_score': best_score
        }
    
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None


# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
    if os.path.exists(DEFAULT_DATA_PATH):
        st.session_state['use_default'] = True
    else:
        st.session_state['use_default'] = False

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📈 Data Analysis", "🤖 Model Prediction", "📊 Model Performance", "ℹ️ About Project"]
)

# Main content based on navigation
if page == "🏠 Home":
    st.markdown('<p class="main-header">👥 HR Employee Attrition Analytics Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the HR Attrition Prediction System
    
    This comprehensive analytics platform helps organizations:
    - 📊 **Analyze** employee data and identify attrition patterns
    - 🔮 **Predict** which employees are at risk of leaving
    - 📈 **Evaluate** model performance with advanced metrics
    - 💡 **Make** data-driven retention decisions
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**📤 Upload Data**\n\nStart by uploading your HR dataset in the sidebar")
    
    with col2:
        st.success("**🔍 Explore Insights**\n\nAnalyze patterns and trends in your employee data")
    
    with col3:
        st.warning("**🎯 Predict Attrition**\n\nGet real-time predictions for employee retention")
    
    # File uploader
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Upload")
    
    if os.path.exists(DEFAULT_DATA_PATH):
        use_default = st.sidebar.checkbox("Use Default Dataset", value=True)
        st.session_state['use_default'] = use_default
        
        if use_default:
            st.session_state['uploaded_file'] = None
            st.session_state['data_loaded'] = True
            st.success(f"✅ Using default dataset from:\n`{DEFAULT_DATA_PATH}`")
            st.info("Navigate to other pages to explore the data!")
    else:
        st.session_state['use_default'] = False
    
    if not st.session_state.get('use_default', False):
        uploaded_file = st.sidebar.file_uploader(
            "Upload HR Dataset (CSV)",
            type=['csv'],
            help="Upload the WA_Fn-UseC_-HR-Employee-Attrition.csv file"
        )
        
        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['data_loaded'] = True
            st.success("✅ Dataset uploaded successfully! Navigate to other pages to explore.")
        else:
            st.info("👆 Please upload your dataset to begin analysis")


elif page == "📈 Data Analysis":
    st.markdown('<p class="main-header">📈 Data Analysis Dashboard</p>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please upload a dataset or use default dataset on the Home page first!")
        st.stop()
    
    if st.session_state.get('use_default', False):
        data = load_and_preprocess_data(None)
    else:
        data = load_and_preprocess_data(st.session_state.get('uploaded_file'))
    
    if data is None:
        st.error("Error loading data. Please check your dataset.")
        st.stop()
    
    df, df_orig, le_dict = data
    
    # Dataset Overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(df_orig))
    with col2:
        attrition_count = df['Attrition'].sum()
        st.metric("Attrition Cases", attrition_count)
    with col3:
        attrition_rate = (attrition_count / len(df)) * 100
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    with col4:
        st.metric("Features", len(df.columns) - 1)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "🔗 Correlations", "📈 Trends", "🎯 Feature Importance"])
    
    with tab1:
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                df_orig,
                names='Attrition',
                title='Attrition Distribution',
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df_orig,
                x='MonthlyIncome',
                color='Attrition',
                title='Monthly Income Distribution by Attrition',
                nbins=30,
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig = px.histogram(
                df_orig,
                x='JobSatisfaction',
                color='Attrition',
                title='Job Satisfaction Levels',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            fig = px.histogram(
                df_orig,
                x='WorkLifeBalance',
                color='Attrition',
                title='Work-Life Balance Distribution',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        numeric_cols = df_orig.select_dtypes(include=[np.number]).columns
        corr_data = df_orig[numeric_cols].copy()
        
        if 'Attrition' in df_orig.columns:
            corr_data['Attrition'] = df['Attrition']
        
        corr_matrix = corr_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title='Feature Correlation Heatmap', height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        if 'Attrition' in corr_matrix.columns:
            st.subheader("Top Correlations with Attrition")
            attrition_corr = corr_matrix['Attrition'].sort_values(ascending=False)
            attrition_corr = attrition_corr[attrition_corr.index != 'Attrition']
            
            fig = px.bar(
                x=attrition_corr.values[:10],
                y=attrition_corr.index[:10],
                orientation='h',
                title='Top 10 Features Correlated with Attrition',
                labels={'x': 'Correlation', 'y': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Attrition Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                df_orig,
                x='Attrition',
                y='YearsAtCompany',
                title='Years at Company vs Attrition',
                color='Attrition'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df_orig,
                x='Attrition',
                y='TotalWorkingYears',
                title='Total Working Years vs Attrition',
                color='Attrition'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if 'OverTime' in df_orig.columns:
            overtime_attrition = pd.crosstab(df_orig['OverTime'], df_orig['Attrition'], normalize='index') * 100
            fig = px.bar(
                overtime_attrition,
                title='Attrition Rate by Overtime Status',
                labels={'value': 'Percentage (%)', 'OverTime': 'Overtime'},
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Engineered Features Analysis")
        
        st.markdown("""
        **Custom Features Created:**
        - **Income_Satisfaction**: MonthlyIncome × JobSatisfaction
        - **Tenure_Balance**: YearsAtCompany / (TotalWorkingYears + 1)
        - **Overload_Score**: OverTime × WorkLifeBalance
        - **Income_Bracket**: Quartile-based income categorization
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df,
                x='Income_Satisfaction',
                y='Attrition',
                title='Income-Satisfaction Score vs Attrition',
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df,
                x='Attrition',
                y='Overload_Score',
                title='Work Overload Score by Attrition'
            )
            st.plotly_chart(fig, use_container_width=True)


elif page == "🤖 Model Prediction":
    st.markdown('<p class="main-header">🤖 Employee Attrition Prediction</p>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please upload a dataset or use default dataset on the Home page first!")
        st.stop()
    
    if st.session_state.get('use_default', False):
        data = load_and_preprocess_data(None)
    else:
        data = load_and_preprocess_data(st.session_state.get('uploaded_file'))
    
    if data is None:
        st.error("Error loading data")
        st.stop()
    
    df, df_orig, le_dict = data
    
    with st.spinner("🔄 Training model with GridSearchCV... This may take a moment."):
        model_data = train_model(df.copy())
    
    if model_data is None:
        st.error("Error training model")
        st.stop()
    
    st.success(f"✅ Model trained successfully! Best params: C={model_data['best_params']['C']}, Gamma={model_data['best_params']['gamma']}")
    
    st.markdown("### 🎯 Make Individual Predictions")
    st.markdown("Enter employee details below to predict attrition risk:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    
    with col2:
        env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
    
    with col3:
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    
    if st.button("🔮 Predict Attrition", type="primary"):
        try:
            overtime_enc = 1 if overtime == "Yes" else 0
            travel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
            business_travel_enc = travel_map[business_travel]
            
            input_data = pd.DataFrame({
                'OverTime': [overtime_enc],
                'MonthlyIncome': [monthly_income],
                'JobSatisfaction': [job_satisfaction],
                'EnvironmentSatisfaction': [env_satisfaction],
                'YearsAtCompany': [years_at_company],
                'TotalWorkingYears': [total_working_years],
                'JobInvolvement': [job_involvement],
                'WorkLifeBalance': [work_life_balance],
                'BusinessTravel': [business_travel_enc]
            })
            
            input_data['Income_Satisfaction'] = input_data['MonthlyIncome'] * input_data['JobSatisfaction']
            input_data['Tenure_Balance'] = input_data['YearsAtCompany'] / (input_data['TotalWorkingYears'] + 1)
            input_data['Overload_Score'] = input_data['OverTime'] * input_data['WorkLifeBalance']
            
            # Simple binning for Income_Bracket
            if monthly_income < 3000:
                input_data['Income_Bracket'] = 0
            elif monthly_income < 5000:
                input_data['Income_Bracket'] = 1
            elif monthly_income < 7000:
                input_data['Income_Bracket'] = 2
            else:
                input_data['Income_Bracket'] = 3
            
            num_cols = ['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears',
                        'Income_Satisfaction', 'Tenure_Balance', 'Overload_Score']
            input_data[num_cols] = model_data['pt'].transform(input_data[num_cols])
            
            poly_input = input_data[['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears']].copy()
            poly_feats = model_data['poly'].transform(poly_input)
            poly_feat_names = model_data['poly'].get_feature_names_out(
                ['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears']
            )
            df_poly = pd.DataFrame(poly_feats, columns=poly_feat_names, index=input_data.index)
            
            input_data = input_data.drop(columns=['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears'])
            input_data = pd.concat([input_data.reset_index(drop=True), df_poly.reset_index(drop=True)], axis=1)
            
            input_data = input_data[model_data['feature_names']]
            
            input_scaled = model_data['scaler'].transform(input_data)
            
            prediction = model_data['model'].predict(input_scaled)[0]
            probability = model_data['model'].predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("### ⚠️ HIGH RISK")
                    st.markdown("**Employee likely to leave**")
                else:
                    st.success("### ✅ LOW RISK")
                    st.markdown("**Employee likely to stay**")
            
            with col2:
                st.metric("Attrition Probability", f"{probability[1]*100:.1f}%")
            
            with col3:
                st.metric("Retention Probability", f"{probability[0]*100:.1f}%")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1]*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Attrition Risk Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("💡 Recommendations")
            if prediction == 1:
                st.markdown("""
                **Action Items:**
                - 🎯 Schedule a one-on-one meeting to discuss career goals
                - 💰 Review compensation and benefits package
                - 📈 Discuss growth opportunities and professional development
                - 🤝 Improve work-life balance and reduce overtime if applicable
                - 🌟 Increase engagement through meaningful projects
                """)
            else:
                st.markdown("""
                **Retention Strategies:**
                - ✅ Continue current engagement practices
                - 📊 Monitor satisfaction levels regularly
                - 🎓 Provide learning and development opportunities
                - 🏆 Recognize and reward contributions
                """)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")


elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">📊 Model Performance Analysis</p>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please upload a dataset or use default dataset on the Home page first!")
        st.stop()
    
    if st.session_state.get('use_default', False):
        data = load_and_preprocess_data(None)
    else:
        data = load_and_preprocess_data(st.session_state.get('uploaded_file'))
    
    if data is None:
        st.error("Error loading data")
        st.stop()
    
    df, df_orig, le_dict = data
    
    with st.spinner("🔄 Training and evaluating model..."):
        model_data = train_model(df.copy())
    
    if model_data is None:
        st.error("Error training model")
        st.stop()
    
    # Performance metrics
    st.subheader("🎯 Model Performance Metrics")
    
    accuracy = accuracy_score(model_data['y_test'], model_data['y_pred'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Accuracy", f"{accuracy*100:.2f}%")
    with col2:
        st.metric("CV Accuracy", f"{model_data['best_cv_score']*100:.2f}%")
    with col3:
        st.metric("Training Samples", len(model_data['y_train']))
    with col4:
        st.metric("Test Samples", len(model_data['y_test']))
    
    # Display best hyperparameters
    st.info(f"🔍 **Optimal Hyperparameters:** C = {model_data['best_params']['C']}, Gamma = {model_data['best_params']['gamma']}, Kernel = {model_data['best_params']['kernel']}")
    
    # Tabs for different performance views
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Classification Report", "📊 Confusion Matrix", "📈 ROC Curve", "🔍 Dimensionality Reduction"])
    
    with tab1:
        st.subheader("Classification Report")
        
        report = classification_report(
            model_data['y_test'],
            model_data['y_pred'],
            target_names=['No Attrition', 'Attrition'],
            output_dict=True
        )
        
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']), use_container_width=True)
        
        metrics_df = report_df.iloc[:2][['precision', 'recall', 'f1-score']]
        
        fig = go.Figure()
        for metric in ['precision', 'recall', 'f1-score']:
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=metrics_df.index,
                y=metrics_df[metric],
                text=[f"{val:.3f}" for val in metrics_df[metric]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Performance Metrics by Class',
            xaxis_title='Class',
            yaxis_title='Score',
            barmode='group',
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Confusion Matrix")
        
        cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted: No Attrition', 'Predicted: Attrition'],
            y=['Actual: No Attrition', 'Actual: Attrition'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        tn, fp, fn, tp = cm.ravel()
        
        with col1:
            st.metric("True Positives", tp)
        with col2:
            st.metric("True Negatives", tn)
        with col3:
            st.metric("False Positives", fp)
        with col4:
            st.metric("False Negatives", fn)
    
    with tab3:
        st.subheader("ROC Curve")
        
        fpr, tpr, thresholds = roc_curve(model_data['y_test'], model_data['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("ROC-AUC Score", f"{roc_auc:.4f}")
        
        st.markdown(f"""
        **Interpretation:**
        - AUC = {roc_auc:.3f} indicates {'excellent' if roc_auc > 0.9 else 'good' if roc_auc > 0.8 else 'fair'} model discrimination
        - The model can distinguish between attrition and non-attrition cases effectively
        """)
    
    with tab4:
        st.subheader("Dimensionality Reduction Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**PCA Projection (2D)**")
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(model_data['X_bal'])
            
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Attrition': model_data['y_bal'].map({0: 'No', 1: 'Yes'})
            })
            
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Attrition',
                title='PCA: Principal Component Analysis',
                color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"Explained Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
        
        with col2:
            st.markdown("**LDA Projection (1D)**")
            lda = LDA(n_components=1)
            X_lda = lda.fit_transform(model_data['X_bal'], model_data['y_bal'])
            
            lda_df = pd.DataFrame({
                'LD1': X_lda[:, 0],
                'Attrition': model_data['y_bal'].map({0: 'No', 1: 'Yes'})
            })
            
            fig = px.violin(
                lda_df,
                y='LD1',
                x='Attrition',
                color='Attrition',
                box=True,
                title='LDA: Linear Discriminant Analysis',
                color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("LDA maximizes class separation in reduced dimensions")
        
        st.markdown("---")
        st.subheader("PCA Explained Variance")
        
        pca_full = PCA(random_state=42)
        pca_full.fit(model_data['X_bal'])
        
        explained_var = np.cumsum(pca_full.explained_variance_ratio_)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(explained_var) + 1)),
            y=explained_var,
            mode='lines+markers',
            name='Cumulative Explained Variance',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                      annotation_text="95% Variance")
        
        fig.update_layout(
            title='Cumulative Explained Variance by Principal Components',
            xaxis_title='Number of Components',
            yaxis_title='Cumulative Explained Variance',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        n_components_95 = np.argmax(explained_var >= 0.95) + 1
        st.info(f"💡 {n_components_95} components needed to explain 95% of variance")


elif page == "ℹ️ About Project":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This **HR Employee Attrition Prediction System** uses advanced machine learning techniques to predict
    employee turnover and help organizations make data-driven retention decisions.
    
    ### 📊 Model Performance
    
    - **Test Accuracy**: 95.75%
    - **Cross-Validation Accuracy**: 93.15%
    - **Optimal Parameters**: C=10, Gamma=1, Kernel=RBF
    
    ### 🔧 Technical Stack
    
    | Component | Technology |
    |-----------|-----------|
    | **Frontend** | Streamlit |
    | **Data Processing** | Pandas, NumPy |
    | **Visualization** | Plotly, Seaborn, Matplotlib |
    | **Machine Learning** | Scikit-learn, Imbalanced-learn |
    | **Model** | Support Vector Machine (SVM) |
    | **Hyperparameter Tuning** | GridSearchCV (5-fold CV) |
    
    ### 🧠 Machine Learning Pipeline
    
    ```
    1. Data Preprocessing
       ├── Label Encoding (Categorical Features)
       ├── Feature Engineering
       │   ├── Income_Satisfaction
       │   ├── Tenure_Balance
       │   ├── Overload_Score
       │   └── Income_Bracket
       └── Yeo-Johnson Power Transformation
    
    2. Feature Enhancement
       ├── Polynomial Features (degree=2)
       └── RobustScaler Normalization
    
    3. Class Balancing
       └── SMOTE (Synthetic Minority Over-sampling)
    
    4. Model Training
       ├── SVM with RBF Kernel
       ├── GridSearchCV (80 fits)
       │   ├── C: [0.1, 1, 10, 100]
       │   └── Gamma: [0.001, 0.01, 0.1, 1]
       └── 5-Fold Cross-Validation
    
    5. Evaluation
       ├── Classification Metrics
       ├── Confusion Matrix
       ├── ROC-AUC Score
       └── Dimensionality Reduction Viz
    ```
    
    ### 📝 Dataset Information
    
    **Default Path**: `C:\\Users\\Dell\\OneDrive\\Desktop\\employeeattrtionpredictor\\WA_Fn-UseC_-HR-Employee-Attrition.csv`
    
    **Key Features Used**:
    - OverTime
    - MonthlyIncome
    - JobSatisfaction
    - EnvironmentSatisfaction
    - YearsAtCompany
    - TotalWorkingYears
    - JobInvolvement
    - WorkLifeBalance
    - BusinessTravel
    
    ### 🚀 Getting Started
    
    1. **Use Default Dataset**: Check the box on Home page
    2. **Or Upload Custom Data**: Upload your own CSV file
    3. **Explore Insights**: Navigate to Data Analysis page
    4. **Make Predictions**: Use Model Prediction for individual cases
    5. **Evaluate Performance**: Review metrics on Model Performance page
    
    ---
    
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
        <h3>Built with ❤️ using Streamlit and Scikit-learn</h3>
        <p>Achieving 95.75% Accuracy with GridSearchCV Optimization</p>
    </div>
    """, unsafe_allow_html=True)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Quick Stats
- **Model**: SVM (RBF Kernel)
- **Accuracy**: 95.75%
- **Best C**: 10
- **Best Gamma**: 1
- **Features**: 9 base + 4 engineered
""")

st.sidebar.markdown("---")
if st.session_state.get('data_loaded', False):
    st.sidebar.success("✅ Dataset loaded!")
else:
    st.sidebar.info("💡 **Tip**: Load dataset on Home page!")
