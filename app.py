"""
Intelligent Property Price Prediction System
================================================
Streamlit-based UI for predicting property prices
using machine learning models.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.predict import predict_price, predict_batch, get_feature_importance, load_model

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Section ===
@st.cache_resource
def get_all_model_info():
    """Load the model info JSON once."""
    from src.predict import MODELS_DIR
    import json
    with open(os.path.join(MODELS_DIR, 'model_info.json'), 'r') as f:
        return json.load(f)

try:
    all_info = get_all_model_info()
    model_names = list(all_info['all_results'].keys())
    
    # Sidebar Model Selection
    with st.sidebar:
        st.markdown("### 🤖 Model Control Panel")
        selected_model_name = st.selectbox(
            "Select Model for Prediction",
            options=model_names,
            index=model_names.index(all_info['best_model'])
        )
        
        # Load the specific selected model
        model, scaler, feature_names, model_info = load_model(selected_model_name)
        model_loaded = True
        
        # Performance Metrics for selected model
        m = all_info['all_results'][selected_model_name]
        st.markdown(f"""
        <div class="metric-card" style="padding: 15px; background: rgba(16, 185, 129, 0.05);">
            <p style="font-size: 0.8rem; color: #94a3b8; margin: 0;">R² Score</p>
            <h2 style="margin: 0; color: #10b981;">{m['R2']:.4f}</h2>
            <p style="font-size: 0.8rem; color: #94a3b8; margin: 10px 0 0 0;">MAE</p>
            <h2 style="margin: 0; color: #f8fafc;">₹{m['MAE']:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📚 What do these mean?"):
            st.markdown("""
            **R² Score**: How accurate the model is (1.0 is perfect).
            
            **MAE (Mean Absolute Error)**: Average amount the prediction is "off" by.
            
            **RMSE**: Similar to MAE but punishes large errors more heavily.
            """)
        
        st.markdown("---")

except Exception as e:
    model_loaded = False
    st.error(f"Model loading failed! Error: {e}")

# ─── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <h1>Property Price Predictor</h1>
    <p>AI-powered real estate price prediction using machine learning</p>
</div>
""", unsafe_allow_html=True)

if model_loaded:
    # ─── Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Predict Price", "Model Insights", "Batch Upload"])

    # ─────────────────────────────────────────────────────────────
    # TAB 1: Single Property Prediction
    # ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<p class="section-header">Enter Property Details</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
            bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
            bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4], index=1)
            stories = st.selectbox("Stories", options=[1, 2, 3, 4], index=1)

        with col2:
            mainroad = st.selectbox("Main Road Access", options=['yes', 'no'], index=0)
            guestroom = st.selectbox("Guest Room", options=['yes', 'no'], index=1)
            basement = st.selectbox("Basement", options=['yes', 'no'], index=1)
            hotwaterheating = st.selectbox("Hot Water Heating", options=['yes', 'no'], index=1)

        with col3:
            airconditioning = st.selectbox("Air Conditioning", options=['yes', 'no'], index=0)
            parking = st.selectbox("Parking Spots", options=[0, 1, 2, 3], index=1)
            prefarea = st.selectbox("Preferred Area", options=['yes', 'no'], index=0)
            furnishingstatus = st.selectbox("Furnishing Status",
                                            options=['furnished', 'semi-furnished', 'unfurnished'], index=1)

        st.markdown("")
        predict_btn = st.button("Predict Price", use_container_width=True)

        if predict_btn:
            input_data = {
                'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
                'stories': stories, 'mainroad': mainroad, 'guestroom': guestroom,
                'basement': basement, 'hotwaterheating': hotwaterheating,
                'airconditioning': airconditioning, 'parking': parking,
                'prefarea': prefarea, 'furnishingstatus': furnishingstatus
            }

            with st.spinner(f"Analyzing property using {selected_model_name}..."):
                # Use the selected model from sidebar
                result = predict_price(
                    input_data, 
                    model=model, 
                    scaler=scaler, 
                    feature_names=feature_names,
                    model_name=selected_model_name
                )

            predicted = result['predicted_price']

            # Prediction Card
            st.markdown(f"""
            <div class="prediction-card">
                <p>Estimated Property Price ({selected_model_name})</p>
                <h1>₹{predicted:,.0f}</h1>
                <p>Confidence (R²): {result['model_r2']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

            # Price context
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Price per Sq Ft</h3>
                    <h2>₹{predicted/area:,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Monthly EMI (20yr @ 8.5%)</h3>
                    <h2>₹{(predicted * 0.085/12 * (1+0.085/12)**240) / ((1+0.085/12)**240 - 1):,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_c:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Model Error (MAE)</h3>
                    <h2>₹{result['model_mae']:,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # TAB 2: Model Insights
    # ─────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<p class="section-header"> Model Performance</p>', unsafe_allow_html=True)

        # Model metrics comparison
        metrics_df = pd.DataFrame(all_info['all_results']).T
        metrics_df.index.name = 'Model'
        metrics_df = metrics_df.reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig_r2 = px.bar(
                metrics_df, x='Model', y='R2',
                title='Model Comparison — R² Score',
                color='R2',
                color_continuous_scale='Teal',
                text='R2'
            )
            fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_r2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                title_font=dict(size=16, color='#f8fafc'),
                xaxis=dict(title='', tickfont=dict(color='#94a3b8')),
                yaxis=dict(title='R² Score', gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#94a3b8')),
                coloraxis_showscale=False,
                height=400
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        with col2:
            fig_mae = px.bar(
                metrics_df, x='Model', y='MAE',
                title='Model Comparison — MAE',
                color='MAE',
                color_continuous_scale='Emrld',
                text='MAE'
            )
            fig_mae.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
            fig_mae.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                title_font=dict(size=16, color='#f8fafc'),
                xaxis=dict(title='', tickfont=dict(color='#94a3b8')),
                yaxis=dict(title='Mean Absolute Error', gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#94a3b8')),
                coloraxis_showscale=False,
                height=400
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        # Feature Importance for SELECTED model
        from src.predict import MODELS_DIR
        st.markdown(f'<p class="section-header"> Feature Importance ({selected_model_name})</p>', unsafe_allow_html=True)
        
        safe_name = selected_model_name.replace(' ', '_').lower()
        importance_path = os.path.join(MODELS_DIR, f'importance_{safe_name}.csv')
        
        if os.path.exists(importance_path):
            importance_df = pd.read_csv(importance_path)
            fig_imp = px.bar(
                importance_df.head(12),
                x='importance', y='feature',
                orientation='h',
                title=f'Key Drivers for {selected_model_name}',
                color='importance',
                color_continuous_scale='Teal'
            )
            fig_imp.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                title_font=dict(size=16, color='#f8fafc'),
                xaxis=dict(title='Importance', gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#94a3b8')),
                yaxis=dict(title='', categoryorder='total ascending', tickfont=dict(color='#94a3b8')),
                coloraxis_showscale=False,
                height=450
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info(f"Feature importance not available for {selected_model_name}.")

    # ─────────────────────────────────────────────────────────────
    # TAB 3: Batch Upload
    # ─────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<p class="section-header">📁 Batch Prediction from CSV</p>', unsafe_allow_html=True)
        st.info(f"Processing will use the selected model: **{selected_model_name}**")

        st.markdown("")
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

        if uploaded_file:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f" Loaded {len(df_upload)} properties")

                if st.button(" Predict All Prices", use_container_width=True):
                    with st.spinner(f" Processing batch predictions using {selected_model_name}..."):
                        result_df = predict_batch(
                            df_upload, 
                            model=model, 
                            scaler=scaler, 
                            feature_names=feature_names
                        )

                    st.markdown('<p class="section-header"> Results</p>', unsafe_allow_html=True)
                    st.dataframe(result_df, use_container_width=True)

                    # Download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label=" Download Predictions CSV",
                        data=csv,
                        file_name=f"predictions_{safe_name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f" Error processing file: {e}")

    # ─── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This app uses **machine learning** to predict
        property prices based on various features like
        area, rooms, amenities, and location preferences.
        """)

        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown(f"""
        - **Model:** {model_info['best_model']}
        - **R² Score:** {model_info['metrics']['R2']:.4f}
        - **MAE:** ₹{model_info['metrics']['MAE']:,.0f}
        - **RMSE:** ₹{model_info['metrics']['RMSE']:,.0f}
        """)

        st.markdown("---")
        st.markdown("### Tech Stack")
        st.markdown("""
        - **ML:** scikit-learn
        - **UI:** Streamlit
        - **Viz:** Plotly
        - **Data:** pandas, NumPy
        """)

        st.markdown("---")
