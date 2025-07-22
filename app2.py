import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .train-metric-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading and model training
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the HR dataset"""
    try:
        df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    except FileNotFoundError:
        st.error("❌ Dataset file not found! Please ensure 'WA_Fn-UseC_-HR-Employee-Attrition.csv' is in the same directory.")
        return None, None, None, None, None, None
    
    # Data preprocessing
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Drop unnecessary columns
    columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
    
    # Define categorical features
    categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                           'JobRole', 'MaritalStatus', 'OverTime']
    categorical_features = [col for col in categorical_features if col in df.columns]
    
    # Prepare features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return df, X_train, X_test, y_train, y_test, categorical_features

@st.cache_resource
def train_model(X_train, y_train, categorical_features):
    """Train the CatBoost model"""
    catboost_classifier = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=False,
        cat_features=categorical_features
    )
    
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    catboost_classifier.fit(train_pool)
    
    return catboost_classifier

def main():
    # Header
    st.markdown('<h1 class="main-header">👔 Employee Attrition Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    data = load_and_preprocess_data()
    if data[0] is None:
        return
    
    df, X_train, X_test, y_train, y_test, categorical_features = data
    
    # Sidebar for navigation
    st.sidebar.markdown("## 🎯 Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "📊 Dashboard",
        "🔮 Predict Attrition",
        "📈 Model Performance",
        "🎯 Feature Analysis"
    ])
    
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "🔮 Predict Attrition":
        show_prediction_page(X_train, y_train, categorical_features, df)
    elif page == "📈 Model Performance":
        show_model_performance(X_train, X_test, y_train, y_test, categorical_features)
    elif page == "🎯 Feature Analysis":
        show_feature_analysis(X_train, y_train, categorical_features)

def show_dashboard(df):
    st.markdown("## 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Total Employees</h3>
            <h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        attrition_rate = (df['Attrition'].sum() / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>📤 Attrition Rate</h3>
            <h2>{attrition_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_age = df['Age'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Average Age</h3>
            <h2>{avg_age:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_income = df['MonthlyIncome'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Avg Income</h3>
            <h2>${avg_income:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Attrition by Department")
        dept_attrition = df.groupby('Department')['Attrition'].agg(['count', 'sum']).reset_index()
        dept_attrition['attrition_rate'] = (dept_attrition['sum'] / dept_attrition['count']) * 100
        
        fig = px.bar(dept_attrition, x='Department', y='attrition_rate',
                    title="Attrition Rate by Department",
                    labels={'attrition_rate': 'Attrition Rate (%)'},
                    color='attrition_rate',
                    color_continuous_scale='Reds')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💼 Job Role Distribution")
        job_counts = df['JobRole'].value_counts()
        fig = px.pie(values=job_counts.values, names=job_counts.index,
                    title="Employee Distribution by Job Role")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Age Distribution")
        fig = px.histogram(df, x='Age', color='Attrition', 
                          title="Age Distribution by Attrition Status",
                          labels={'Attrition': 'Attrition Status'},
                          color_discrete_map={0: '#4CAF50', 1: '#F44336'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Income vs Attrition")
        fig = px.box(df, x='Attrition', y='MonthlyIncome',
                    title="Monthly Income Distribution by Attrition",
                    labels={'Attrition': 'Attrition Status', 'MonthlyIncome': 'Monthly Income'})
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(X_train, y_train, categorical_features, df):
    st.markdown("## 🔮 Predict Employee Attrition")
    
    # Train model
    with st.spinner("🤖 Training model..."):
        model = train_model(X_train, y_train, categorical_features)
    
    st.markdown("### 📝 Enter Employee Information")
    
    col1, col2, col3 = st.columns(3)
    
    # Input fields
    with col1:
        age = st.slider("👤 Age", 18, 65, 30)
        gender = st.selectbox("👥 Gender", df['Gender'].unique())
        marital_status = st.selectbox("💍 Marital Status", df['MaritalStatus'].unique())
        education = st.slider("🎓 Education Level", 1, 5, 3)
        education_field = st.selectbox("📚 Education Field", df['EducationField'].unique())
    
    with col2:
        department = st.selectbox("🏢 Department", df['Department'].unique())
        job_role = st.selectbox("💼 Job Role", df['JobRole'].unique())
        job_level = st.slider("📊 Job Level", 1, 5, 2)
        monthly_income = st.slider("💰 Monthly Income", 1000, 20000, 5000)
        distance_from_home = st.slider("🏠 Distance from Home", 1, 30, 10)
    
    with col3:
        overtime = st.selectbox("⏰ Overtime", df['OverTime'].unique())
        business_travel = st.selectbox("✈️ Business Travel", df['BusinessTravel'].unique())
        work_life_balance = st.slider("⚖️ Work Life Balance", 1, 4, 3)
        job_satisfaction = st.slider("😊 Job Satisfaction", 1, 4, 3)
        environment_satisfaction = st.slider("🌍 Environment Satisfaction", 1, 4, 3)
    
    # Additional fields
    st.markdown("### 📊 Additional Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_working_years = st.slider("📅 Total Working Years", 0, 40, 10)
        years_at_company = st.slider("🏢 Years at Company", 0, 40, 5)
    
    with col2:
        years_in_current_role = st.slider("💼 Years in Current Role", 0, 20, 2)
        years_since_last_promotion = st.slider("🚀 Years Since Last Promotion", 0, 15, 1)
    
    with col3:
        years_with_curr_manager = st.slider("👨‍💼 Years with Current Manager", 0, 20, 3)
        num_companies_worked = st.slider("🏭 Number of Companies Worked", 0, 10, 2)
    
    with col4:
        training_times_last_year = st.slider("📚 Training Times Last Year", 0, 6, 2)
        percent_salary_hike = st.slider("📈 Percent Salary Hike", 10, 25, 15)
    
    # Create prediction button
    if st.button("🔮 Predict Attrition", type="primary"):
        # Prepare input data
        input_data = {
            'Age': age,
            'BusinessTravel': business_travel,
            'DailyRate': monthly_income * 12 / 365,  # Approximate daily rate
            'Department': department,
            'DistanceFromHome': distance_from_home,
            'Education': education,
            'EducationField': education_field,
            'EnvironmentSatisfaction': environment_satisfaction,
            'Gender': gender,
            'HourlyRate': monthly_income / 160,  # Approximate hourly rate
            'JobInvolvement': 3,  # Default value
            'JobLevel': job_level,
            'JobRole': job_role,
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': marital_status,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_income,
            'NumCompaniesWorked': num_companies_worked,
            'OverTime': overtime,
            'PercentSalaryHike': percent_salary_hike,
            'PerformanceRating': 3,  # Default value
            'RelationshipSatisfaction': 3,  # Default value
            'StockOptionLevel': 1,  # Default value
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times_last_year,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'YearsWithCurrManager': years_with_curr_manager
        }
        
        # Make prediction
        sample_df = pd.DataFrame([input_data])
        sample_df = sample_df.reindex(columns=X_train.columns, fill_value=0)
        
        sample_pool = Pool(sample_df, cat_features=categorical_features)
        prediction = model.predict(sample_pool)[0]
        probability = model.predict_proba(sample_pool)[:, 1][0]
        
        # Display results
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box">
                <h2>⚠️ High Risk of Attrition</h2>
                <h3>Probability: {probability:.2%}</h3>
                <p>This employee is likely to leave the company.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <h2>✅ Low Risk of Attrition</h2>
                <h3>Probability: {probability:.2%}</h3>
                <p>This employee is likely to stay with the company.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk meter
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Attrition Risk Meter"},
            delta = {'reference': 50},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_model_performance(X_train, X_test, y_train, y_test, categorical_features):
    st.markdown("## 📈 Model Performance Analysis")
    
    # Train model
    with st.spinner("🤖 Training model..."):
        model = train_model(X_train, y_train, categorical_features)
    
    # Make predictions on both training and test sets
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for both sets
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_auc_score = roc_auc_score(y_train, y_train_pred_proba)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc_score = roc_auc_score(y_test, y_test_pred_proba)
    
    # Display Training Metrics
    st.markdown("### 🎓 Training Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="train-metric-card">
            <h3>🎯 Training Accuracy</h3>
            <h2>{train_accuracy:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="train-metric-card">
            <h3>📊 Training AUC</h3>
            <h2>{train_auc_score:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Display Test Metrics
    st.markdown("### 🧪 Test Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Test Accuracy</h3>
            <h2>{test_accuracy:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Test AUC</h3>
            <h2>{test_auc_score:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        test_precision = len(y_test[(y_test == 1) & (y_test_pred == 1)]) / len(y_test[y_test_pred == 1]) if len(y_test[y_test_pred == 1]) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔍 Precision</h3>
            <h2>{test_precision:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        test_recall = len(y_test[(y_test == 1) & (y_test_pred == 1)]) / len(y_test[y_test == 1]) if len(y_test[y_test == 1]) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎪 Recall</h3>
            <h2>{test_recall:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance Comparison
    st.markdown("### 📊 Training vs Test Performance Comparison")
    comparison_data = {
        'Metric': ['Accuracy', 'AUC Score'],
        'Training': [train_accuracy, train_auc_score],
        'Test': [test_accuracy, test_auc_score]
    }
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(comparison_df, x='Metric', y=['Training', 'Test'], 
                title="Training vs Test Performance",
                barmode='group',
                color_discrete_map={'Training': '#11998e', 'Test': '#667eea'})
    fig.update_layout(yaxis_title='Score', showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Confusion Matrix (Test Set)")
        cm = confusion_matrix(y_test, y_test_pred)
        
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['No Attrition', 'Attrition'],
                       y=['No Attrition', 'Attrition'],
                       color_continuous_scale='Blues',
                       text_auto=True)
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 ROC Curves Comparison")
        # Calculate ROC curves for both training and test sets
        train_fpr, train_tpr, _ = roc_curve(y_train, y_train_pred_proba)
        test_fpr, test_tpr, _ = roc_curve(y_test, y_test_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_fpr, y=train_tpr, mode='lines', 
                                name=f'Training ROC (AUC = {train_auc_score:.3f})',
                                line=dict(color='#11998e', width=2)))
        fig.add_trace(go.Scatter(x=test_fpr, y=test_tpr, mode='lines', 
                                name=f'Test ROC (AUC = {test_auc_score:.3f})',
                                line=dict(color='#667eea', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                name='Random Classifier', 
                                line=dict(dash='dash', color='gray')))
        fig.update_layout(
            title='ROC Curves - Training vs Test',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True,
            width=500,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis(X_train, y_train, categorical_features):
    st.markdown("## 🎯 Feature Importance Analysis")
    
    # Train model
    with st.spinner("🤖 Training model..."):
        model = train_model(X_train, y_train, categorical_features)
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Display top features
    st.subheader("🏆 Top 15 Most Important Features")
    
    top_features = importance_df.head(15)
    fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                title="Feature Importance Scores",
                labels={'importance': 'Importance Score', 'feature': 'Features'},
                color='importance',
                color_continuous_scale='Viridis')
    fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    st.subheader("📊 Complete Feature Importance Table")
    st.dataframe(importance_df, use_container_width=True)
    
    # Download button for feature importance
    csv = importance_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Feature Importance CSV",
        data=csv,
        file_name="feature_importance.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()