"""
🏠 Intelligent Property Price Prediction System
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
    page_title="🏠 Property Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "static", "style.css")
with open(css_path) as f:
    css_content = f.read()
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────
@st.cache_resource
def cached_load_model():
    return load_model()

try:
    model, scaler, feature_names, model_info = cached_load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model not found! Please run `python src/train.py` first.\n\nError: {e}")

# ─── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <h1>🏠 Property Price Predictor</h1>
    <p>AI-powered real estate price prediction using machine learning</p>
</div>
""", unsafe_allow_html=True)

if model_loaded:
    # ─── Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🏡 Predict Price", "📊 Model Insights", "📁 Batch Upload"])

    # ─────────────────────────────────────────────────────────────
    # TAB 1: Single Property Prediction
    # ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<p class="section-header">📝 Enter Property Details</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            area = st.number_input("🏗️ Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
            bedrooms = st.selectbox("🛏️ Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
            bathrooms = st.selectbox("🚿 Bathrooms", options=[1, 2, 3, 4], index=1)
            stories = st.selectbox("🏢 Stories", options=[1, 2, 3, 4], index=1)

        with col2:
            mainroad = st.selectbox("🛣️ Main Road Access", options=['yes', 'no'], index=0)
            guestroom = st.selectbox("🛋️ Guest Room", options=['yes', 'no'], index=1)
            basement = st.selectbox("🏠 Basement", options=['yes', 'no'], index=1)
            hotwaterheating = st.selectbox("🔥 Hot Water Heating", options=['yes', 'no'], index=1)

        with col3:
            airconditioning = st.selectbox("❄️ Air Conditioning", options=['yes', 'no'], index=0)
            parking = st.selectbox("🚗 Parking Spots", options=[0, 1, 2, 3], index=1)
            prefarea = st.selectbox("⭐ Preferred Area", options=['yes', 'no'], index=0)
            furnishingstatus = st.selectbox("🪑 Furnishing Status",
                                            options=['furnished', 'semi-furnished', 'unfurnished'], index=1)

        st.markdown("")
        predict_btn = st.button("🔮 Predict Price", use_container_width=True)

        if predict_btn:
            input_data = {
                'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
                'stories': stories, 'mainroad': mainroad, 'guestroom': guestroom,
                'basement': basement, 'hotwaterheating': hotwaterheating,
                'airconditioning': airconditioning, 'parking': parking,
                'prefarea': prefarea, 'furnishingstatus': furnishingstatus
            }

            with st.spinner("🔄 Analyzing property..."):
                result = predict_price(input_data, model=model, scaler=scaler, feature_names=feature_names)

            predicted = result['predicted_price']

            # Prediction Card
            st.markdown(f"""
            <div class="prediction-card">
                <p>Estimated Property Price</p>
                <h1>₹{predicted:,.0f}</h1>
                <p>Model: {result['model_name']} | R² Score: {result['model_r2']}</p>
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
                    <h3>Model Accuracy</h3>
                    <h2>{result['model_r2'] * 100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # TAB 2: Model Insights
    # ─────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<p class="section-header">📊 Model Performance</p>', unsafe_allow_html=True)

        # Model metrics comparison
        all_results = model_info.get('all_results', {})
        if all_results:
            metrics_df = pd.DataFrame(all_results).T
            metrics_df.index.name = 'Model'
            metrics_df = metrics_df.reset_index()

            col1, col2 = st.columns(2)

            with col1:
                # R² comparison chart
                fig_r2 = px.bar(
                    metrics_df, x='Model', y='R2',
                    title='Model Comparison — R² Score',
                    color='R2',
                    color_continuous_scale='Viridis',
                    text='R2'
                )
                fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_r2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e7ff'),
                    title_font=dict(size=16, color='#e0e7ff'),
                    xaxis=dict(title='', tickfont=dict(color='#c7d2fe')),
                    yaxis=dict(title='R² Score', gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#c7d2fe')),
                    coloraxis_showscale=False,
                    height=400
                )
                st.plotly_chart(fig_r2, use_container_width=True)

            with col2:
                # MAE comparison
                fig_mae = px.bar(
                    metrics_df, x='Model', y='MAE',
                    title='Model Comparison — MAE',
                    color='MAE',
                    color_continuous_scale='Reds_r',
                    text='MAE'
                )
                fig_mae.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
                fig_mae.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e7ff'),
                    title_font=dict(size=16, color='#e0e7ff'),
                    xaxis=dict(title='', tickfont=dict(color='#c7d2fe')),
                    yaxis=dict(title='Mean Absolute Error', gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#c7d2fe')),
                    coloraxis_showscale=False,
                    height=400
                )
                st.plotly_chart(fig_mae, use_container_width=True)

            # Metrics table
            st.markdown('<p class="section-header">📋 Detailed Metrics</p>', unsafe_allow_html=True)
            styled_df = metrics_df.copy()
            styled_df['MAE'] = styled_df['MAE'].apply(lambda x: f"₹{x:,.2f}")
            styled_df['RMSE'] = styled_df['RMSE'].apply(lambda x: f"₹{x:,.2f}")
            styled_df['R2'] = styled_df['R2'].apply(lambda x: f"{x:.4f}")

            # Highlight best model
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            st.info(f"🏆 **Best Model:** {model_info['best_model']} with R² = {model_info['metrics']['R2']:.4f}")

        # Feature Importance
        st.markdown('<p class="section-header">🎯 Feature Importance</p>', unsafe_allow_html=True)
        importance_df = get_feature_importance()
        if importance_df is not None:
            fig_imp = px.bar(
                importance_df.head(12),
                x='importance', y='feature',
                orientation='h',
                title='Top Price-Driving Factors',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e7ff'),
                title_font=dict(size=16, color='#e0e7ff'),
                xaxis=dict(title='Importance', gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#c7d2fe')),
                yaxis=dict(title='', categoryorder='total ascending', tickfont=dict(color='#c7d2fe')),
                coloraxis_showscale=False,
                height=450
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("""
            <div class="metric-card" style="text-align: left; padding: 20px;">
                <h3 style="margin-bottom: 12px;">💡 Key Insights</h3>
                <p style="color: #c7d2fe; line-height: 1.8;">
                    • <strong>Area</strong> is the most influential factor in determining property prices<br>
                    • <strong>Bathrooms</strong> and <strong>stories</strong> significantly impact pricing<br>
                    • <strong>Air conditioning</strong> and <strong>main road access</strong> add premium value<br>
                    • <strong>Furnished</strong> properties command higher prices than unfurnished ones
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # TAB 3: Batch Upload
    # ─────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<p class="section-header">📁 Batch Prediction from CSV</p>', unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="text-align: left;">
            <h3>📋 Required CSV Columns</h3>
            <p style="color: #c7d2fe;">
                area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                hotwaterheating, airconditioning, parking, prefarea, furnishingstatus
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

        if uploaded_file:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_upload)} properties")

                # Show preview
                st.markdown('<p class="section-header">📋 Data Preview</p>', unsafe_allow_html=True)
                st.dataframe(df_upload.head(10), use_container_width=True)

                if st.button("🔮 Predict All Prices", use_container_width=True):
                    with st.spinner("🔄 Processing batch predictions..."):
                        result_df = predict_batch(df_upload, model=model, scaler=scaler, feature_names=feature_names)

                    st.markdown('<p class="section-header">📊 Predictions</p>', unsafe_allow_html=True)
                    st.dataframe(result_df, use_container_width=True)

                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Price", f"₹{result_df['predicted_price'].mean():,.0f}")
                    with col2:
                        st.metric("Min Price", f"₹{result_df['predicted_price'].min():,.0f}")
                    with col3:
                        st.metric("Max Price", f"₹{result_df['predicted_price'].max():,.0f}")
                    with col4:
                        st.metric("Total Properties", len(result_df))

                    # Price distribution of predictions
                    fig_dist = px.histogram(
                        result_df, x='predicted_price', nbins=20,
                        title='Predicted Price Distribution',
                        color_discrete_sequence=['#764ba2']
                    )
                    fig_dist.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e7ff'),
                        title_font=dict(size=16, color='#e0e7ff'),
                        xaxis=dict(title='Predicted Price', gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(title='Count', gridcolor='rgba(255,255,255,0.1)'),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    # Download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="property_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    # ─── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏠 About")
        st.markdown("""
        This app uses **machine learning** to predict
        property prices based on various features like
        area, rooms, amenities, and location preferences.
        """)

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.markdown(f"""
        - **Model:** {model_info['best_model']}
        - **R² Score:** {model_info['metrics']['R2']:.4f}
        - **MAE:** ₹{model_info['metrics']['MAE']:,.0f}
        - **RMSE:** ₹{model_info['metrics']['RMSE']:,.0f}
        """)

        st.markdown("---")
        st.markdown("### 🔧 Tech Stack")
        st.markdown("""
        - **ML:** scikit-learn
        - **UI:** Streamlit
        - **Viz:** Plotly
        - **Data:** pandas, NumPy
        """)

        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: #6b7280; font-size: 0.8rem;'>"
            "Built with ❤️ for AI Real Estate Analytics</p>",
            unsafe_allow_html=True
        )
