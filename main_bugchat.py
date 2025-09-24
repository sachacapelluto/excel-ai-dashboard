# main.py - Excel AI Dashboard avec CSV et génération directe
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
import requests
import json
import re
from typing import Dict, List, Tuple, Optional
import sqlite3
import statistics
warnings.filterwarnings('ignore')

# Configuration page
st.set_page_config(
    page_title="Excel AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Initialisation cleaning
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'cleaning_applied' not in st.session_state:
    st.session_state.cleaning_applied = False
# Initialisation session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'upload_count' not in st.session_state:
    st.session_state.upload_count = 0
if 'data_source' not in st.session_state:
    st.session_state.data_source = None

# Header
st.title("📊 Excel AI Dashboard")
st.markdown("**Transform your data into AI-powered insights in seconds**")
st.markdown("---")

# Sidebar info
with st.sidebar:
    st.markdown("### 🚀 Features")
    st.markdown("✅ Excel & CSV Upload")
    st.markdown("✅ Data Cleaning System") 
    st.markdown("✅ Sample Data Generator") 
    st.markdown("✅ Auto Data Analysis") 
    st.markdown("✅ Custom Chart Builder")
    st.markdown("✅ Smart Visualizations")
    st.markdown("✅ AI Chat")
    st.markdown("🧠 Memory System (Coming Soon)")
    
    st.markdown("---")
    st.markdown("### 🤖 AI Assistant Status")
    
    if st.session_state.processed_data is not None:
        st.metric("Files Processed", st.session_state.upload_count)
        st.info(f"Source: {st.session_state.data_source}")
        st.metric("Rows", len(st.session_state.processed_data))
        st.metric("Columns", len(st.session_state.processed_data.columns))
        
        # NOUVEAU - Statut nettoyage
        if st.session_state.get('cleaning_applied', False):
            st.markdown("### 🧹 Cleaning Status")
            st.success("✅ Data Cleaned")
    else:
        st.metric("Files Processed", st.session_state.upload_count)




def generate_ecommerce_data():
    """Génère données e-commerce"""
    dates = [datetime.now() - timedelta(days=x) for x in range(90)]
    data = {
        'date': dates,
        'sales': [random.randint(1000, 5000) for _ in range(90)],
        'visitors': [random.randint(500, 2000) for _ in range(90)], 
        'conversion_rate': [round(random.uniform(0.02, 0.08), 4) for _ in range(90)],
        'product_category': [random.choice(['Electronics', 'Clothing', 'Books', 'Home']) for _ in range(90)],
        'marketing_spend': [random.randint(200, 800) for _ in range(90)],
        'orders': [random.randint(50, 200) for _ in range(90)],
        'avg_order_value': [round(random.uniform(25.0, 150.0), 2) for _ in range(90)]
    }
    return pd.DataFrame(data)

def generate_healthcare_data():
    """Génère données healthcare"""
    dates = [datetime.now() - timedelta(days=x) for x in range(60)]
    data = {
        'date': dates,
        'patients_seen': [random.randint(20, 80) for _ in range(60)],
        'avg_wait_time': [random.randint(15, 45) for _ in range(60)],
        'satisfaction_score': [round(random.uniform(3.5, 5.0), 2) for _ in range(60)],
        'department': [random.choice(['Emergency', 'Cardiology', 'Pediatrics', 'Surgery']) for _ in range(60)],
        'staff_on_duty': [random.randint(5, 15) for _ in range(60)],
        'revenue': [random.randint(5000, 25000) for _ in range(60)],
        'readmission_rate': [round(random.uniform(0.05, 0.20), 3) for _ in range(60)]
    }
    return pd.DataFrame(data)



def generate_dirty_dataset():
    """Génère un dataset intentionnellement 'sale' pour tester le nettoyage"""
    import random
    from datetime import datetime, timedelta
    import numpy as np
    
    # Base de données avec 100 lignes
    dates = []
    sales = []
    products = []
    regions = []
    prices = []
    customers = []
    categories = []
    revenues = []
    
    base_date = datetime.now() - timedelta(days=100)
    
    for i in range(100):
        # Dates avec formats inconsistants (PROBLÈME 1)
        if i % 4 == 0:
            dates.append((base_date + timedelta(days=i)).strftime('%Y-%m-%d'))  # Format ISO
        elif i % 4 == 1:
            dates.append((base_date + timedelta(days=i)).strftime('%d/%m/%Y'))  # Format EU
        elif i % 4 == 2:
            dates.append((base_date + timedelta(days=i)).strftime('%m-%d-%Y'))  # Format US
        else:
            dates.append((base_date + timedelta(days=i)).strftime('%B %d, %Y'))  # Format texte
        
        # Sales avec valeurs manquantes (PROBLÈME 2)
        if i % 7 == 0:  # ~14% de valeurs manquantes
            sales.append(np.nan)
        else:
            sales.append(random.randint(500, 3000))
        
        # Products avec valeurs vides (PROBLÈME 3)
        if i % 8 == 0:
            products.append("")  # Chaîne vide
        elif i % 8 == 1:
            products.append(None)  # None
        else:
            products.append(random.choice(['iPhone', 'Samsung', 'iPad', 'MacBook', 'Dell Laptop']))
        
        # Regions avec inconsistances (PROBLÈME 4)
        region_variants = {
            'North': ['North', 'NORTH', ' North ', 'north', 'N'],
            'South': ['South', 'SOUTH', ' South', 'south', 'S'],
            'East': ['East', 'EAST', 'east', ' East ', 'E'],
            'West': ['West', 'WEST', 'west', ' West', 'W']
        }
        base_region = random.choice(list(region_variants.keys()))
        regions.append(random.choice(region_variants[base_region]))
        
        # Prices avec formats incohérents (PROBLÈME 5)
        base_price = random.randint(100, 2000)
        if i % 5 == 0:
            prices.append(f"${base_price:,}")  # Format avec $
        elif i % 5 == 1:
            prices.append(f"{base_price}.00")  # Format décimal
        elif i % 5 == 2:
            prices.append(f"{base_price}€")  # Format euro
        elif i % 5 == 3:
            prices.append(str(base_price))  # String simple
        else:
            prices.append(base_price)  # Numeric correct
        
        # Customers avec doublons intentionnels (PROBLÈME 6)
        if i < 20:  # Premiers 20 = clients uniques
            customers.append(f"Customer_{i+1:03d}")
        else:  # Après, répétitions pour créer doublons
            customers.append(f"Customer_{random.randint(1, 15):03d}")
        
        # Categories avec valeurs manquantes (PROBLÈME 7)
        if i % 6 == 0:
            categories.append(np.nan)
        else:
            categories.append(random.choice(['Electronics', 'Accessories', 'Software', 'Hardware']))
        
        # Revenue avec outliers extrêmes (PROBLÈME 8)
        if i % 15 == 0:  # Outliers intentionnels
            revenues.append(random.randint(50000, 100000))  # Très élevé
        elif i % 17 == 0:
            revenues.append(random.randint(1, 10))  # Très bas
        else:
            revenues.append(random.randint(1000, 5000))  # Normal
    
    # Construction DataFrame avec colonnes supplémentaires problématiques
    data = {
        'transaction_date': dates,  # Formats de dates mixtes
        'sales_amount': sales,  # Valeurs manquantes
        'product_name': products,  # Chaînes vides
        'region': regions,  # Inconsistances de casse
        'price_str': prices,  # Types mixtes (string/numeric)
        'customer_id': customers,  # Doublons
        'category': categories,  # Valeurs manquantes
        'total_revenue': revenues,  # Outliers
        
        # Colonnes supplémentaires avec problèmes spécifiques
        'empty_column': [np.nan] * 100,  # PROBLÈME 9: Colonne complètement vide
        'mostly_empty': [1, 2, np.nan, np.nan, np.nan] * 20,  # PROBLÈME 10: 90% vide
        'debug_flag': [True, False] * 50,  # PROBLÈME 11: Colonne technique inutile
        'temp_id': [f"temp_{i}" for i in range(100)],  # PROBLÈME 12: Colonne temporaire
        
        # Données numériques en format text (PROBLÈME 13)
        'quantity_text': [str(random.randint(1, 10)) for _ in range(100)],
        'discount_text': [f"{random.randint(0, 30)}%" for _ in range(100)],
        
        # Espaces et caractères indésirables (PROBLÈME 14)
        'notes': [
            f" Note {i} " if i % 3 == 0 else
            f"Note{i}\t" if i % 3 == 1 else
            f"  Note  {i}  \n" 
            for i in range(100)
        ]
    }
    
    # Création DataFrame
    df = pd.DataFrame(data)
    
    # Ajout de lignes dupliquées exactes (PROBLÈME 15)
    duplicate_indices = [5, 15, 25, 35, 45]  # 5 doublons exacts
    duplicate_rows = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    return df

def generate_moderately_dirty_dataset():
    """Génère un dataset avec quelques problèmes seulement (test intermédiaire)"""
    import random
    from datetime import datetime, timedelta
    
    # Dataset plus petit avec moins de problèmes
    dates = [(datetime.now() - timedelta(days=x)) for x in range(50)]
    
    data = {
        'date': dates,
        'sales': [random.randint(800, 2500) if i % 10 != 0 else np.nan for i in range(50)],  # 10% manquantes
        'product': [random.choice(['Phone', 'Laptop', 'Tablet']) for _ in range(50)],
        'region': [random.choice(['North', 'South', 'East', 'West']) for _ in range(50)],
        'revenue': [
            random.randint(15000, 25000) if i % 20 == 0 else  # Quelques outliers
            random.randint(1000, 4000)
            for i in range(50)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Ajout de 2-3 doublons
    duplicate_rows = df.iloc[[5, 15, 25]].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    return df


def display_sample_data_generation():
    """Interface améliorée avec datasets de test nettoyage"""
    st.markdown("**Generate sample data for testing:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🛍️ Generate E-commerce Data", use_container_width=True):
            with st.spinner("Generating clean e-commerce data..."):
                df = generate_ecommerce_data()
                df = process_data(df, "Generated E-commerce Dataset (90 days)")
            st.success("✅ Clean e-commerce data generated!")
    
    with col2:
        if st.button("🏥 Generate Healthcare Data", use_container_width=True):
            with st.spinner("Generating clean healthcare data..."):
                df = generate_healthcare_data()
                df = process_data(df, "Generated Healthcare Dataset (60 days)")
            st.success("✅ Clean healthcare data generated!")
    
    with col3:
        if st.button("🧪 Generate DIRTY Dataset", use_container_width=True, help="Perfect for testing data cleaning features!"):
            with st.spinner("Generating intentionally messy data..."):
                df = generate_dirty_dataset()
                df = process_data(df, "🧪 DIRTY Test Dataset (15+ problems)")
            st.success("⚠️ Dirty dataset generated - perfect for cleaning tests!")
    
    # Bouton pour dataset moyennement sale
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔧 Generate Moderately Dirty Dataset", use_container_width=True):
            with st.spinner("Generating dataset with some issues..."):
                df = generate_moderately_dirty_dataset()
                df = process_data(df, "🔧 Moderately Dirty Dataset (few problems)")
            st.success("✅ Moderately dirty dataset generated!")
    
    with col2:
        st.info("💡 **Tip:** Use the Dirty Dataset to test all cleaning features!")
    
    st.markdown("---")
    st.markdown("### 🎯 Dataset Descriptions:")
    
    with st.expander("🧪 What's in the DIRTY Dataset?"):
        st.markdown("""
        **This dataset contains 15+ intentional problems:**
        
        **Missing Values:**
        - 🚫 14% missing sales data
        - 🚫 17% missing categories
        
        **Data Type Issues:**
        - 📊 Numbers stored as text with $ and % symbols
        - 📅 Dates in 4 different formats (ISO, EU, US, text)
        
        **Inconsistent Data:**
        - 🔤 Region names with mixed case and spaces
        - 🗑️ Empty product names
        
        **Duplicates & Outliers:**
        - 🔄 5 exact duplicate transactions
        - 🎯 Extreme revenue outliers (1-10 vs 50000-100000)
        
        **Useless Columns:**
        - 📂 Completely empty columns
        - 🔧 Debug flags and temp IDs
        - 📝 Text with extra spaces and tabs
        
        **Perfect for testing all cleaning features!** 🧹
        """)


def create_custom_chart(df, chart_type, x_col, y_col, color_col=None, size_col=None, title="", 
                       chart_color="#1f77b4", x_label=None, y_label=None, axis_color="#000000"):
    """Version avec personnalisation axes et couleurs"""
    try:
        # Couleur par défaut
        color_sequence = [chart_color]
        
        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title or f"{y_col} over {x_col}")
            if not color_col:
                fig.update_traces(line_color=chart_color)
                
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title or f"{y_col} by {x_col}")
            if not color_col:
                fig.update_traces(marker_color=chart_color)
                
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, 
                           title=title or f"{y_col} vs {x_col}")
            if not color_col:
                fig.update_traces(marker_color=chart_color)
                
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_col, title=title or f"Distribution of {x_col}")
            if not color_col:
                fig.update_traces(marker_color=chart_color)
                
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title or f"{y_col} distribution by {x_col}")
            if not color_col:
                fig.update_traces(marker_color=chart_color)
                
        elif chart_type == "Pie Chart":
            if y_col:
                pie_data = df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.pie(pie_data, values=y_col, names=x_col, title=title or f"{y_col} by {x_col}")
            else:
                value_counts = df[x_col].value_counts().reset_index()
                fig = px.pie(value_counts, values='count', names=x_col, title=title or f"Distribution of {x_col}")
                
        elif chart_type == "Heatmap":
            if df[x_col].dtype in ['object'] and df[y_col].dtype in ['object']:
                heatmap_data = pd.crosstab(df[x_col], df[y_col])
            else:
                heatmap_data = df.pivot_table(values=color_col if color_col else df.columns[0], 
                                            index=x_col, columns=y_col, aggfunc='mean')
            
            fig = px.imshow(heatmap_data, title=title or f"Heatmap: {x_col} vs {y_col}")
            
        elif chart_type == "Area Chart":
            fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title or f"{y_col} area over {x_col}")
            if not color_col:
                fig.update_traces(fill='tonexty', fillcolor=chart_color, line_color=chart_color)
                
        elif chart_type == "Violin Plot":
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, title=title or f"{y_col} distribution by {x_col}")
            if not color_col:
                fig.update_traces(marker_color=chart_color)
                
        else:
            return None
        
        # PERSONNALISATION AXES ET COULEURS
        fig.update_layout(
            height=500, 
            showlegend=True,
            # Labels personnalisés
            xaxis_title=x_label or x_col,
            yaxis_title=y_label or y_col,
            # Couleurs axes
            xaxis=dict(
                title_font_color=axis_color,
                tickfont_color=axis_color
            ),
            yaxis=dict(
                title_font_color=axis_color,
                tickfont_color=axis_color
            ),
            # Couleur titre
            title_font_color=axis_color
        )
        
        # Cas spécial pour Pie Chart (pas d'axes)
        if chart_type == "Pie Chart":
            fig.update_layout(
                title_font_color=axis_color,
                font_color=axis_color
            )
            
        return fig
        
    except Exception as e:
        st.error(f"❌ Erreur création graphique: {str(e)}")
        return None

def display_chart_builder(df):
    """Interface avec personnalisation axes et couleurs"""
    st.subheader("🎨 Custom Chart Builder")
    st.markdown("**Create your personalized visualizations:**")
    
    # Séparer colonnes par type
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
            datetime_cols.append(col)
    
    all_cols = df.columns.tolist()
    
    # Interface en 2 colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Chart Configuration**")
        chart_type = st.selectbox(
            "Chart Type",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", 
             "Box Plot", "Pie Chart", "Heatmap", "Area Chart", "Violin Plot"],
            help="Choose the type of visualization"
        )
        
        # Sélection intelligente selon le type
        if chart_type in ["Line Chart", "Area Chart"]:
            x_col = st.selectbox("X-Axis Data", datetime_cols + categorical_cols + numeric_cols, 
                               help="Usually time/dates for trends")
        elif chart_type == "Histogram":
            x_col = st.selectbox("Column to Analyze", numeric_cols + categorical_cols,
                               help="Column to show distribution")
        elif chart_type == "Pie Chart":
            x_col = st.selectbox("Categories", categorical_cols,
                               help="Categories for pie slices")
        else:
            x_col = st.selectbox("X-Axis Data", all_cols)
    
    with col2:
        st.markdown("**📊 Data Configuration**")
        # Y-axis selon type de graphique
        if chart_type in ["Histogram", "Pie Chart"]:
            y_col = st.selectbox("Values (optional)", [None] + numeric_cols,
                               help="Leave empty to count occurrences")
        elif chart_type in ["Box Plot", "Violin Plot"]:
            y_col = st.selectbox("Y-Axis Values", numeric_cols,
                               help="Numeric values to analyze")
        elif chart_type == "Heatmap":
            y_col = st.selectbox("Y-Axis Data", all_cols,
                               help="Second dimension")
        else:
            y_col = st.selectbox("Y-Axis Data", numeric_cols + [None])
    
    # SECTION AVANCÉE AMÉLIORÉE
    with st.expander("🎨 Advanced Styling & Data Options"):
        
        # DATA OPTIONS
        st.markdown("**📊 Data Visualization**")
        col1, col2 = st.columns(2)
        
        with col1:
            color_col = st.selectbox(
                "Color By", 
                [None] + categorical_cols + numeric_cols,
                help="🎯 Color data points by category/value. Example: Color sales bars by 'product_category' to see which category performs best"
            )
            
            # Explication Color By
            if color_col:
                unique_values = df[color_col].nunique()
                st.info(f"ℹ️ Will create {unique_values} different colors for each unique value in '{color_col}'")
        
        with col2:
            if chart_type == "Scatter Plot":
                size_col = st.selectbox(
                    "Size By", 
                    [None] + numeric_cols,
                    help="Size points by values (bubble chart effect)"
                )
            else:
                size_col = None
        
        st.markdown("---")
        
        # STYLE OPTIONS
        st.markdown("**🎨 Visual Styling**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chart_color = st.color_picker(
                "Chart Color", 
                value="#1f77b4", 
                help="Main color for your chart (only applies when 'Color By' is empty)"
            )
        
        with col2:
            axis_color = st.color_picker(
                "Text & Axes Color", 
                value="#000000", 
                help="Color for titles, labels, and axis text"
            )
        
        with col3:
            st.markdown("**Color Preview**")
            st.markdown(f'<div style="background: linear-gradient(45deg, {chart_color}, {axis_color}); height: 30px; border-radius: 5px; border: 1px solid #ddd;"></div>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # LABELS OPTIONS
        st.markdown("**📝 Custom Labels**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_label = st.text_input(
                "X-Axis Label", 
                value="", 
                placeholder=f"Default: {x_col}" if x_col else "Auto",
                help="Custom name for X-axis"
            )
        
        with col2:
            if chart_type not in ["Pie Chart", "Histogram"]:
                y_label = st.text_input(
                    "Y-Axis Label", 
                    value="", 
                    placeholder=f"Default: {y_col}" if y_col else "Auto",
                    help="Custom name for Y-axis"
                )
            else:
                y_label = None
        
        with col3:
            custom_title = st.text_input(
                "Chart Title", 
                value="", 
                placeholder="Auto-generated title",
                help="Custom title for your chart"
            )
    
    # SECTION AIDE COLOR BY
    if not color_col:
        st.info("💡 **Tip:** Use 'Color By' to add a data dimension! Example: Color sales by product category to see which products perform best.")
    
    # Bouton création avec style
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        create_button = st.button(
            "🎨 Create Custom Chart", 
            type="primary", 
            use_container_width=True
        )
    
    if create_button:
        if x_col and (y_col or chart_type in ["Histogram", "Pie Chart"]):
            with st.spinner("🎨 Creating your personalized chart..."):
                fig = create_custom_chart(
                    df, chart_type, x_col, y_col, color_col, size_col, 
                    custom_title, chart_color, x_label, y_label, axis_color
                )
                
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"chart_builder_{len(st.session_state.get('custom_charts', []))}")
                
                # Sauvegarder avec nouvelles options
                if 'custom_charts' not in st.session_state:
                    st.session_state.custom_charts = []
                
                chart_config = {
                    'type': chart_type,
                    'x_col': x_col,
                    'y_col': y_col,
                    'color_col': color_col,
                    'size_col': size_col,
                    'title': custom_title or f"{chart_type}: {y_col or x_col} by {x_col}",
                    'chart_color': chart_color,
                    'axis_color': axis_color,
                    'x_label': x_label,
                    'y_label': y_label,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.custom_charts.append(chart_config)
                
                # Message personnalisé
                color_info = f" with {color_col} coloring" if color_col else f" in {chart_color}"
                st.success(f"✅ {chart_type} created{color_info}!")
        else:
            st.warning("⚠️ Please select required columns for this chart type")


def display_chart_gallery(df):
    """Gallery avec support des nouvelles options"""
    if 'custom_charts' in st.session_state and st.session_state.custom_charts:
        st.subheader("🖼️ Your Chart Gallery")
        st.markdown(f"**{len(st.session_state.custom_charts)} custom charts created**")
        
        # Options de galerie
        col1, col2 = st.columns([3, 1])
        with col1:
            view_mode = st.radio("View Mode", ["Grid View", "List View"], horizontal=True)
        with col2:
            if st.button("🗑️ Clear Gallery"):
                st.session_state.custom_charts = []
                st.rerun()
        
        if view_mode == "Grid View":
            # Affichage en grille
            cols = st.columns(2)
            for i, chart_config in enumerate(st.session_state.custom_charts[-6:]):
                with cols[i % 2]:
                    with st.container():
                        st.markdown(f"**{chart_config['title']}**")
                        
                        # Info styling
                        style_info = []
                        if chart_config.get('color_col'):
                            style_info.append(f"Colored by {chart_config['color_col']}")
                        else:
                            style_info.append(f"Color: {chart_config.get('chart_color', '#1f77b4')}")
                        
                        st.caption(" | ".join(style_info) + f" | {chart_config['timestamp']}")
                        
                        # Recréer avec toutes les options
                        fig = create_custom_chart(
                            df, chart_config['type'], chart_config['x_col'], 
                            chart_config['y_col'], chart_config['color_col'], 
                            chart_config['size_col'], chart_config['title'],
                            chart_config.get('chart_color', '#1f77b4'),
                            chart_config.get('x_label'), chart_config.get('y_label'),
                            chart_config.get('axis_color', '#000000')
                        )
                        if fig:
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True, key=f"gallery_grid_{i}_{chart_config['timestamp']}")
        
        else:
            # Affichage en liste détaillée
            for i, chart_config in enumerate(reversed(st.session_state.custom_charts)):
                with st.expander(f"📊 {chart_config['title']} - {chart_config['timestamp']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Détails configuration
                        st.markdown(f"**Type:** {chart_config['type']}")
                        st.markdown(f"**X-Axis:** {chart_config['x_col']} → *{chart_config.get('x_label', 'Auto label')}*")
                        if chart_config['y_col']:
                            st.markdown(f"**Y-Axis:** {chart_config['y_col']} → *{chart_config.get('y_label', 'Auto label')}*")
                        
                        # Info couleurs
                        if chart_config['color_col']:
                            st.markdown(f"**🎨 Color By:** {chart_config['color_col']}")
                        else:
                            chart_color = chart_config.get('chart_color', '#1f77b4')
                            st.markdown(f"**🎨 Chart Color:** {chart_color}")
                            st.markdown(f'<div style="background-color: {chart_color}; height: 20px; width: 100px; border-radius: 3px; border: 1px solid #ddd; display: inline-block;"></div>', 
                                       unsafe_allow_html=True)
                        
                        axis_color = chart_config.get('axis_color', '#000000')
                        st.markdown(f"**📝 Text Color:** {axis_color}")
                    
                    with col2:
                        if st.button("🔄 Recreate", key=f"recreate_list_{i}_{chart_config['timestamp']}"):
                            fig = create_custom_chart(
                                df, chart_config['type'], chart_config['x_col'], 
                                chart_config['y_col'], chart_config['color_col'], 
                                chart_config['size_col'], chart_config['title'],
                                chart_config.get('chart_color', '#1f77b4'),
                                chart_config.get('x_label'), chart_config.get('y_label'),
                                chart_config.get('axis_color', '#000000')
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"recreated_{i}_{chart_config['timestamp']}")
    else:
        st.info("🎨 Create some custom charts to see them in your gallery!")

def suggest_chart_ideas(df):
    """Suggère des idées de graphiques selon les données"""
    st.subheader("💡 Chart Suggestions")
    st.markdown("**Based on your data, try these visualizations:**")
    
    suggestions = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Détection colonnes temporelles
    datetime_cols = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
            datetime_cols.append(col)
    
    # Suggestions basées sur structure des données
    if datetime_cols and numeric_cols:
        suggestions.append({
            'icon': '📈',
            'title': 'Time Series Analysis',
            'description': f'Show {numeric_cols[0]} trends over {datetime_cols[0]}',
            'type': 'Line Chart',
            'x': datetime_cols[0],
            'y': numeric_cols[0]
        })
    
    if len(numeric_cols) >= 2:
        suggestions.append({
            'icon': '🔍',
            'title': 'Correlation Analysis', 
            'description': f'Explore relationship between {numeric_cols[0]} and {numeric_cols[1]}',
            'type': 'Scatter Plot',
            'x': numeric_cols[0],
            'y': numeric_cols[1]
        })
    
    if categorical_cols and numeric_cols:
        suggestions.append({
            'icon': '📊',
            'title': 'Category Performance',
            'description': f'Compare {numeric_cols[0]} across {categorical_cols[0]}',
            'type': 'Bar Chart',
            'x': categorical_cols[0],
            'y': numeric_cols[0]
        })
    
    if categorical_cols:
        suggestions.append({
            'icon': '🥧',
            'title': 'Distribution Analysis',
            'description': f'See composition of {categorical_cols[0]}',
            'type': 'Pie Chart',
            'x': categorical_cols[0],
            'y': None
        })
    
    if len(numeric_cols) >= 3:
        suggestions.append({
            'icon': '🌡️',
            'title': 'Correlation Heatmap',
            'description': 'Visualize correlations between all numeric columns',
            'type': 'Heatmap',
            'x': numeric_cols[0],
            'y': numeric_cols[1]
        })
    
    # Affichage suggestions
    cols = st.columns(min(3, len(suggestions)))
    for i, suggestion in enumerate(suggestions[:3]):  # Top 3 suggestions
        with cols[i]:
            st.markdown(f"### {suggestion['icon']} {suggestion['title']}")
            st.markdown(suggestion['description'])
            
            if st.button(f"Create {suggestion['type']}", key=f"suggest_{i}"):
                fig = create_custom_chart(
                    df, suggestion['type'], suggestion['x'], suggestion['y'], 
                    title=suggestion['title']
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Auto-save to gallery
                    if 'custom_charts' not in st.session_state:
                        st.session_state.custom_charts = []
                    
                    chart_config = {
                        'type': suggestion['type'],
                        'x_col': suggestion['x'],
                        'y_col': suggestion['y'],
                        'color_col': None,
                        'size_col': None,
                        'title': suggestion['title'],
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state.custom_charts.append(chart_config)



def analyze_data_quality(df):
    """Analyse complète de la qualité des données"""
    report = {
        'missing_values': {},
        'duplicates': 0,
        'outliers': {},
        'data_types': {},
        'empty_columns': [],
        'quality_score': 100,
        'total_issues': 0
    }
    
    # 1. Analyse valeurs manquantes
    missing_data = df.isnull().sum()
    for col in missing_data[missing_data > 0].index:
        missing_count = missing_data[col]
        missing_percent = (missing_count / len(df)) * 100
        report['missing_values'][col] = {
            'count': int(missing_count),
            'percentage': round(missing_percent, 2)
        }
    
    # 2. Détection doublons
    report['duplicates'] = df.duplicated().sum()
    
    # 3. Détection outliers (colonnes numériques)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            report['outliers'][col] = {
                'count': len(outliers),
                'percentage': round((len(outliers) / len(df)) * 100, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2)
            }
    
    # 4. Problèmes types de données
    for col in df.columns:
        if df[col].dtype == 'object':
            # Vérifier si colonne numérique cachée
            try:
                # Test conversion numérique
                pd.to_numeric(df[col].dropna().head(10))
                report['data_types'][col] = 'numeric_as_text'
            except:
                # Test dates
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    report['data_types'][col] = 'date_as_text'
                except:
                    pass
    
    # 5. Colonnes vides ou quasi-vides
    for col in df.columns:
        non_null_percent = (df[col].count() / len(df)) * 100
        if non_null_percent < 10:  # Moins de 10% de données
            report['empty_columns'].append({
                'column': col,
                'fill_rate': round(non_null_percent, 2)
            })
    
    # 6. Calcul score qualité
    issues_found = (
        len(report['missing_values']) + 
        (1 if report['duplicates'] > 0 else 0) +
        len(report['outliers']) +
        len(report['data_types']) +
        len(report['empty_columns'])
    )
    
    report['total_issues'] = issues_found
    report['quality_score'] = max(0, 100 - (issues_found * 15))  # -15 points par type de problème
    
    return report

def display_quality_report(df, report):
    """Affiche le rapport de qualité avec style"""
    st.subheader("📊 Data Quality Report")
    
    # Score global avec couleur
    score = report['quality_score']
    if score >= 80:
        score_color = "🟢"
        score_status = "Excellent"
    elif score >= 60:
        score_color = "🟡"  
        score_status = "Good"
    elif score >= 40:
        score_color = "🟠"
        score_status = "Needs Improvement"
    else:
        score_color = "🔴"
        score_status = "Poor"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Quality Score", f"{score}/100", delta=score_status)
        st.markdown(f"{score_color} **{score_status}**")
    
    with col2:
        st.metric("Total Issues", report['total_issues'])
    
    with col3:
        st.metric("Dataset Size", f"{len(df):,} rows")
    
    with col4:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    
    if report['total_issues'] == 0:
        st.success("🎉 **Perfect!** Your data is clean and ready for analysis!")
        return
    
    st.markdown("---")
    
    # Détails des problèmes
    if report['missing_values']:
        st.markdown("### 🚫 Missing Values Detected")
        missing_df = pd.DataFrame([
            {'Column': col, 'Missing Count': info['count'], 'Missing %': f"{info['percentage']}%"}
            for col, info in report['missing_values'].items()
        ])
        st.dataframe(missing_df, hide_index=True)
    
    if report['duplicates'] > 0:
        st.markdown(f"### 🔄 Duplicate Rows: **{report['duplicates']}** found")
    
    if report['outliers']:
        st.markdown("### 🎯 Outliers Detected")
        outlier_df = pd.DataFrame([
            {'Column': col, 'Outliers': info['count'], 'Outliers %': f"{info['percentage']}%"}
            for col, info in report['outliers'].items()
        ])
        st.dataframe(outlier_df, hide_index=True)
    
    if report['data_types']:
        st.markdown("### 🔄 Data Type Issues")
        for col, issue in report['data_types'].items():
            if issue == 'numeric_as_text':
                st.info(f"📊 Column **{col}** contains numbers stored as text")
            elif issue == 'date_as_text':
                st.info(f"📅 Column **{col}** contains dates stored as text")
    
    if report['empty_columns']:
        st.markdown("### 🗑️ Nearly Empty Columns")
        empty_df = pd.DataFrame(report['empty_columns'])
        st.dataframe(empty_df, hide_index=True)

def display_cleaning_actions(df, report):
    """Interface d'actions de nettoyage avec boutons"""
    st.subheader("🛠️ Data Cleaning Actions")
    
    if report['total_issues'] == 0:
        st.info("✨ No cleaning actions needed - your data is already clean!")
        return df
    
    st.markdown("**Select which cleaning actions to apply:**")
    
    # Actions sélectionnables
    actions_to_apply = []
    cleaned_df = df.copy()
    
    # Section 1: Valeurs manquantes
    if report['missing_values']:
        st.markdown("#### 🚫 Missing Values Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numeric Columns:**")
            numeric_missing = [col for col in report['missing_values'].keys() 
                             if df[col].dtype in ['int64', 'float64']]
            
            for col in numeric_missing:
                missing_info = report['missing_values'][col]
                with st.expander(f"📊 {col} ({missing_info['count']} missing values)"):
                    
                    action = st.radio(
                        f"How to handle missing values in {col}?",
                        ["Keep as is", "Fill with Mean", "Fill with Median", "Remove rows"],
                        key=f"missing_{col}",
                        help=f"{missing_info['percentage']}% of data is missing"
                    )
                    
                    if action != "Keep as is":
                        actions_to_apply.append({
                            'type': 'missing_numeric',
                            'column': col,
                            'action': action,
                            'affected_rows': missing_info['count']
                        })
        
        with col2:
            st.markdown("**Text Columns:**")
            text_missing = [col for col in report['missing_values'].keys() 
                           if df[col].dtype == 'object']
            
            for col in text_missing:
                missing_info = report['missing_values'][col]
                with st.expander(f"📝 {col} ({missing_info['count']} missing values)"):
                    
                    action = st.radio(
                        f"How to handle missing values in {col}?",
                        ["Keep as is", "Fill with Most Common", "Fill with 'Unknown'", "Remove rows"],
                        key=f"missing_text_{col}",
                        help=f"{missing_info['percentage']}% of data is missing"
                    )
                    
                    if action != "Keep as is":
                        actions_to_apply.append({
                            'type': 'missing_text',
                            'column': col,
                            'action': action,
                            'affected_rows': missing_info['count']
                        })
    
    # Section 2: Doublons
    if report['duplicates'] > 0:
        st.markdown("#### 🔄 Duplicate Rows Actions")
        with st.expander(f"🔄 Remove {report['duplicates']} duplicate rows"):
            duplicate_action = st.radio(
                "How to handle duplicate rows?",
                ["Keep as is", "Remove duplicates (keep first)", "Remove duplicates (keep last)"],
                key="duplicates_action"
            )
            
            if duplicate_action != "Keep as is":
                actions_to_apply.append({
                    'type': 'duplicates',
                    'action': duplicate_action,
                    'affected_rows': report['duplicates']
                })
    
    # Section 3: Outliers
    if report['outliers']:
        st.markdown("#### 🎯 Outliers Actions")
        
        for col, outlier_info in report['outliers'].items():
            with st.expander(f"🎯 {col} ({outlier_info['count']} outliers)"):
                st.info(f"Normal range: {outlier_info['lower_bound']} to {outlier_info['upper_bound']}")
                
                outlier_action = st.radio(
                    f"How to handle outliers in {col}?",
                    ["Keep as is", "Remove outlier rows", "Cap to normal range"],
                    key=f"outlier_{col}",
                    help=f"{outlier_info['percentage']}% of data are outliers"
                )
                
                if outlier_action != "Keep as is":
                    actions_to_apply.append({
                        'type': 'outliers',
                        'column': col,
                        'action': outlier_action,
                        'affected_rows': outlier_info['count'],
                        'bounds': (outlier_info['lower_bound'], outlier_info['upper_bound'])
                    })
    
    # Section 4: Types de données
    if report['data_types']:
        st.markdown("#### 🔄 Data Type Conversions")
        
        for col, issue_type in report['data_types'].items():
            with st.expander(f"🔄 Convert {col} data type"):
                if issue_type == 'numeric_as_text':
                    convert_numeric = st.checkbox(
                        f"Convert {col} to numeric",
                        key=f"convert_numeric_{col}",
                        help="This will convert text numbers to actual numbers for calculations"
                    )
                    if convert_numeric:
                        actions_to_apply.append({
                            'type': 'convert_numeric',
                            'column': col
                        })
                
                elif issue_type == 'date_as_text':
                    convert_date = st.checkbox(
                        f"Convert {col} to date format",
                        key=f"convert_date_{col}",
                        help="This will convert text dates to proper date format"
                    )
                    if convert_date:
                        actions_to_apply.append({
                            'type': 'convert_date',
                            'column': col
                        })
    
    # Section 5: Colonnes vides
    if report['empty_columns']:
        st.markdown("#### 🗑️ Empty Columns Actions")
        
        for empty_col in report['empty_columns']:
            col_name = empty_col['column']
            fill_rate = empty_col['fill_rate']
            
            with st.expander(f"🗑️ {col_name} ({fill_rate}% filled)"):
                drop_column = st.checkbox(
                    f"Remove column {col_name}",
                    key=f"drop_{col_name}",
                    help=f"This column has very little data ({fill_rate}% filled)"
                )
                if drop_column:
                    actions_to_apply.append({
                        'type': 'drop_column',
                        'column': col_name
                    })
    
    # Bouton d'application avec preview
    if actions_to_apply:
        st.markdown("---")
        st.markdown("### 📋 Actions Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Actions to apply:**")
            for i, action in enumerate(actions_to_apply, 1):
                if action['type'] == 'missing_numeric':
                    st.markdown(f"{i}. Fill missing {action['column']} with {action['action'].lower()}")
                elif action['type'] == 'missing_text':
                    st.markdown(f"{i}. Handle missing {action['column']}: {action['action']}")
                elif action['type'] == 'duplicates':
                    st.markdown(f"{i}. Remove {action['affected_rows']} duplicate rows")
                elif action['type'] == 'outliers':
                    st.markdown(f"{i}. {action['action']} in {action['column']}")
                elif action['type'] == 'convert_numeric':
                    st.markdown(f"{i}. Convert {action['column']} to numeric")
                elif action['type'] == 'convert_date':
                    st.markdown(f"{i}. Convert {action['column']} to date")
                elif action['type'] == 'drop_column':
                    st.markdown(f"{i}. Remove column {action['column']}")
        
        with col2:
            st.markdown("**Impact:**")
            total_affected = sum(action.get('affected_rows', 0) for action in actions_to_apply)
            st.info(f"📊 Approximately {total_affected} data points will be affected")
            
            original_shape = df.shape
            st.markdown(f"**Original:** {original_shape[0]} rows × {original_shape[1]} columns")
        
        # Boutons d'action
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Preview Changes", type="secondary", use_container_width=True):
                st.session_state.preview_cleaning = True
        
        with col2:
            if st.button("🧹 Apply All Changes", type="primary", use_container_width=True):
                cleaned_df = apply_cleaning_actions(df, actions_to_apply)
                st.session_state.cleaned_data = cleaned_df
                st.session_state.original_data = df
                st.session_state.cleaning_applied = True
                st.success("✅ Data cleaning applied successfully!")
                st.rerun()
        
        with col3:
            if st.button("↩️ Reset All", use_container_width=True):
                st.session_state.cleaned_data = None
                st.session_state.cleaning_applied = False
                st.rerun()
    
    return cleaned_df

def apply_cleaning_actions(df, actions):
    """Applique les actions de nettoyage sélectionnées"""
    cleaned_df = df.copy()
    
    for action in actions:
        try:
            if action['type'] == 'missing_numeric':
                if action['action'] == "Fill with Mean":
                    cleaned_df[action['column']] = cleaned_df[action['column']].fillna(
                        cleaned_df[action['column']].mean()
                    )
                elif action['action'] == "Fill with Median":
                    cleaned_df[action['column']] = cleaned_df[action['column']].fillna(
                        cleaned_df[action['column']].median()
                    )
                elif action['action'] == "Remove rows":
                    cleaned_df = cleaned_df.dropna(subset=[action['column']])
            
            elif action['type'] == 'missing_text':
                if action['action'] == "Fill with Most Common":
                    mode_value = cleaned_df[action['column']].mode()[0] if len(cleaned_df[action['column']].mode()) > 0 else 'Unknown'
                    cleaned_df[action['column']] = cleaned_df[action['column']].fillna(mode_value)
                elif action['action'] == "Fill with 'Unknown'":
                    cleaned_df[action['column']] = cleaned_df[action['column']].fillna('Unknown')
                elif action['action'] == "Remove rows":
                    cleaned_df = cleaned_df.dropna(subset=[action['column']])
            
            elif action['type'] == 'duplicates':
                if "keep first" in action['action']:
                    cleaned_df = cleaned_df.drop_duplicates(keep='first')
                elif "keep last" in action['action']:
                    cleaned_df = cleaned_df.drop_duplicates(keep='last')
            
            elif action['type'] == 'outliers':
                col = action['column']
                lower_bound, upper_bound = action['bounds']
                
                if action['action'] == "Remove outlier rows":
                    cleaned_df = cleaned_df[
                        (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
                    ]
                elif action['action'] == "Cap to normal range":
                    cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif action['type'] == 'convert_numeric':
                cleaned_df[action['column']] = pd.to_numeric(
                    cleaned_df[action['column']], errors='coerce'
                )
            
            elif action['type'] == 'convert_date':
                cleaned_df[action['column']] = pd.to_datetime(
                    cleaned_df[action['column']], errors='coerce'
                )
            
            elif action['type'] == 'drop_column':
                cleaned_df = cleaned_df.drop(columns=[action['column']])
        
        except Exception as e:
            st.error(f"❌ Error applying {action['type']} to {action.get('column', 'data')}: {str(e)}")
    
    return cleaned_df

def display_before_after_comparison(original_df, cleaned_df):
    """Affiche la comparaison avant/après nettoyage"""
    st.subheader("📈 Before vs After Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Original Data")
        st.info(f"**Shape:** {original_df.shape[0]} rows × {original_df.shape[1]} columns")
        
        original_missing = original_df.isnull().sum().sum()
        original_duplicates = original_df.duplicated().sum()
        
        st.markdown(f"**Missing values:** {original_missing}")
        st.markdown(f"**Duplicate rows:** {original_duplicates}")
        st.markdown(f"**Memory usage:** {original_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    with col2:
        st.markdown("#### ✨ Cleaned Data")
        st.success(f"**Shape:** {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
        
        cleaned_missing = cleaned_df.isnull().sum().sum()
        cleaned_duplicates = cleaned_df.duplicated().sum()
        
        st.markdown(f"**Missing values:** {cleaned_missing}")
        st.markdown(f"**Duplicate rows:** {cleaned_duplicates}")
        st.markdown(f"**Memory usage:** {cleaned_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Changements
    st.markdown("#### 📋 Summary of Changes")
    rows_removed = original_df.shape[0] - cleaned_df.shape[0]
    cols_removed = original_df.shape[1] - cleaned_df.shape[1]
    missing_fixed = original_missing - cleaned_missing
    
    change_metrics = []
    if rows_removed > 0:
        change_metrics.append(f"🗑️ Removed {rows_removed} rows")
    if cols_removed > 0:
        change_metrics.append(f"🗑️ Removed {cols_removed} columns")
    if missing_fixed > 0:
        change_metrics.append(f"✅ Fixed {missing_fixed} missing values")
    if original_duplicates > cleaned_duplicates:
        change_metrics.append(f"✅ Removed {original_duplicates - cleaned_duplicates} duplicates")
    
    if change_metrics:
        for metric in change_metrics:
            st.markdown(f"- {metric}")
    else:
        st.info("No structural changes made to the dataset")
    
    # Boutons d'export
    col1, col2 = st.columns(2)
    
    with col1:
        csv_cleaned = cleaned_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Data (CSV)",
            data=csv_cleaned,
            file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Use Cleaned Data for Analysis", type="primary", use_container_width=True):
            st.session_state.processed_data = cleaned_df
            st.session_state.data_source = f"Cleaned Data ({original_df.shape[0]}→{cleaned_df.shape[0]} rows)"
            st.success("✅ Switched to cleaned data for analysis!")
            st.rerun()

def load_file_data(uploaded_file):
    """Charge données depuis fichier uploadé"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Tentatives multiples pour CSV avec différents séparateurs et encodings
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except Exception:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
                except Exception:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep='\t', encoding='utf-8')
                    except Exception:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
            
            return df, f"CSV Upload: {uploaded_file.name}"
            
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
            return df, f"Excel Upload: {uploaded_file.name}"
        else:
            st.error("❌ Format non supporté")
            return None, None
            
    except Exception as e:
        st.error(f"❌ Erreur lecture fichier: {str(e)}")
        return None, None

def process_data(df, source_name):
    """Traite et analyse les données"""
    st.session_state.processed_data = df
    st.session_state.data_source = source_name
    st.session_state.upload_count += 1
    
    return df


# SYSTÈME LLM LOCAL INTÉGRÉ - À AJOUTER dans main.py

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Installation automatique des dépendances LLM
def install_llm_dependencies():
    """Installe les dépendances LLM automatiquement"""
    try:
        import subprocess
        import sys
        
        # Installation Hugging Face Transformers (plus fiable que GPT4All)
        packages = [
            'transformers>=4.35.0',
            'torch>=2.0.0',
            'accelerate>=0.24.0',
            'sentencepiece>=0.1.99'
        ]
        
        for package in packages:
            try:
                __import__(package.split('>=')[0])
            except ImportError:
                st.info(f"📦 Installation de {package.split('>=')[0]}... (première utilisation)")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        return True
    except Exception as e:
        st.error(f"Erreur installation: {e}")
        return False

class LocalLLMChat:
    def __init__(self, df=None):
        self.df = df
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.model_loaded = False
        self.init_chat_memory()
        
    def init_chat_memory(self):
        """Initialise base de données mémoire"""
        self.chat_conn = sqlite3.connect('llm_chat_memory.db', check_same_thread=False)
        cursor = self.chat_conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                user_message TEXT,
                llm_response TEXT,
                data_context TEXT,
                reasoning_used BOOLEAN
            )
        ''')
        self.chat_conn.commit()
    
    def load_model(self, model_choice="microsoft/DialoGPT-medium"):
        """Charge le modèle LLM local"""
        if self.model_loaded:
            return True
        
        try:
            # Installation des dépendances si nécessaire
            if not install_llm_dependencies():
                return False
            
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            with st.spinner("🤖 Chargement du modèle IA... (peut prendre 1-2 minutes)"):
                
                # Modèles disponibles par ordre de préférence
                models = {
                    "microsoft/DialoGPT-medium": "Conversationnel équilibré (500MB)",
                    "microsoft/DialoGPT-small": "Léger et rapide (100MB)", 
                    "distilbert/distilgpt2": "Ultra-léger (80MB)",
                    "gpt2": "Classique OpenAI GPT-2 (500MB)"
                }
                
                try:
                    # Chargement du modèle sélectionné
                    self.tokenizer = AutoTokenizer.from_pretrained(model_choice, padding_side='left')
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_choice,
                        pad_token_id=self.tokenizer.eos_token_id,
                        torch_dtype="auto"
                    )
                    
                    # Configuration pour la génération
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.model_loaded = True
                    st.success(f"✅ Modèle {model_choice} chargé avec succès!")
                    return True
                    
                except Exception as model_error:
                    st.warning(f"⚠️ Erreur chargement {model_choice}: {model_error}")
                    # Fallback vers modèle plus simple
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
                        self.model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        self.model_loaded = True
                        st.success("✅ Modèle fallback chargé!")
                        return True
                    except:
                        return False
        
        except Exception as e:
            st.error(f"❌ Impossible de charger le modèle: {e}")
            return False
    
    def analyze_data_for_context(self) -> str:
        """Analyse les données pour créer contexte intelligent"""
        if self.df is None:
            return "Aucune donnée disponible."
        
        context = []
        
        # Informations de base
        context.append(f"Dataset: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        
        # Colonnes et types
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or self.df[col].dtype == 'datetime64[ns]']
        
        if numeric_cols:
            context.append(f"Colonnes numériques: {', '.join(numeric_cols[:5])}")
            # Statistiques clés pour colonnes principales
            for col in numeric_cols[:3]:
                mean_val = self.df[col].mean()
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                context.append(f"{col}: moyenne {mean_val:.1f}, range {min_val:.1f}-{max_val:.1f}")
        
        if categorical_cols:
            context.append(f"Colonnes catégorielles: {', '.join(categorical_cols[:3])}")
            # Valeurs uniques pour catégories principales
            for col in categorical_cols[:2]:
                unique_vals = self.df[col].nunique()
                top_values = self.df[col].value_counts().head(3).index.tolist()
                context.append(f"{col}: {unique_vals} valeurs uniques, top: {', '.join(map(str, top_values))}")
        
        if date_cols:
            context.append(f"Période temporelle: {date_cols[0]} disponible")
        
        # Qualité des données
        missing_total = self.df.isnull().sum().sum()
        if missing_total > 0:
            context.append(f"Données manquantes: {missing_total} valeurs")
        
        return " | ".join(context)
    
    def calculate_specific_insight(self, question: str) -> str:
        """Calcule des insights spécifiques selon la question"""
        if self.df is None:
            return ""
        
        question_lower = question.lower()
        calculations = []
        
        try:
            # Détection mots-clés et calculs correspondants
            if any(word in question_lower for word in ['moyenne', 'mean', 'average']):
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    for col in numeric_cols[:2]:
                        avg = self.df[col].mean()
                        calculations.append(f"Moyenne {col}: {avg:.2f}")
            
            if any(word in question_lower for word in ['corrélation', 'correlation', 'relation']):
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    corr_val = self.df[numeric_cols[0]].corr(self.df[numeric_cols[1]])
                    calculations.append(f"Corrélation {numeric_cols[0]}-{numeric_cols[1]}: {corr_val:.2f}")
            
            if any(word in question_lower for word in ['tendance', 'trend', 'évolution']):
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    first_half = self.df[col][:len(self.df)//2].mean()
                    second_half = self.df[col][len(self.df)//2:].mean()
                    trend = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
                    direction = "hausse" if trend > 0 else "baisse"
                    calculations.append(f"Tendance {col}: {direction} de {abs(trend):.1f}%")
            
            if any(word in question_lower for word in ['maximum', 'max', 'minimum', 'min']):
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    max_val = self.df[col].max()
                    min_val = self.df[col].min()
                    calculations.append(f"Range {col}: min {min_val:.1f}, max {max_val:.1f}")
        
        except Exception as e:
            calculations.append(f"Erreur calcul: {e}")
        
        return " | ".join(calculations) if calculations else ""
    
    def generate_response(self, user_message: str) -> str:
        """Génère réponse avec le LLM local + calculs"""
        if not self.model_loaded:
            return "❌ Modèle IA non chargé. Utilisez 'Charger Modèle IA' d'abord."
        
        try:
            # Contexte des données
            data_context = self.analyze_data_for_context()
            specific_calculations = self.calculate_specific_insight(user_message)
            
            # Construction du prompt enrichi
            system_prompt = f"""Tu es un assistant IA expert en analyse de données. Tu réponds en français de manière claire et précise.

CONTEXTE DES DONNÉES:
{data_context}

CALCULS SPÉCIFIQUES:
{specific_calculations}

QUESTION UTILISATEUR: {user_message}

Réponds de manière conversationnelle en utilisant les informations ci-dessus. Si tu mentionnes des chiffres, utilise ceux fournis dans le contexte. Sois concis mais informatif."""

            # Tokenisation
            inputs = self.tokenizer.encode(system_prompt, return_tensors='pt', max_length=512, truncation=True)
            
            # Génération avec paramètres optimisés
            with st.spinner("🤖 Génération de la réponse..."):
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=150,  # Réponses concises
                    num_return_sequences=1,
                    temperature=0.7,     # Créativité modérée
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # Décodage et nettoyage
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraction de la réponse (après le prompt)
            response_part = response[len(system_prompt):].strip()
            
            # Nettoyage et amélioration
            if len(response_part) < 10:  # Réponse trop courte
                response_part = self.create_fallback_response(user_message, specific_calculations)
            
            # Sauvegarde en mémoire
            self.save_conversation(user_message, response_part, data_context)
            
            return response_part
            
        except Exception as e:
            return f"❌ Erreur génération: {e}\n\n🔧 Réponse basique: {self.create_fallback_response(user_message, specific_calculations)}"
    
    def create_fallback_response(self, question: str, calculations: str) -> str:
        """Crée réponse de fallback intelligente"""
        if calculations:
            return f"📊 Basé sur vos données: {calculations}\n\n💡 Ces résultats peuvent vous aider à répondre à votre question sur {question}."
        else:
            return f"🤖 Je comprends votre question sur '{question}'. Avec les données disponibles ({self.analyze_data_for_context()}), je peux vous aider à analyser ces informations plus spécifiquement."
    
    def save_conversation(self, user_message: str, llm_response: str, data_context: str):
        """Sauvegarde conversation"""
        cursor = self.chat_conn.cursor()
        cursor.execute('''
            INSERT INTO llm_conversations (timestamp, user_message, llm_response, data_context, reasoning_used)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), user_message, llm_response, data_context, True))
        self.chat_conn.commit()
    
    def get_conversation_history(self, limit: int = 5) -> List[Dict]:
        """Récupère historique"""
        cursor = self.chat_conn.cursor()
        cursor.execute('''
            SELECT timestamp, user_message, llm_response
            FROM llm_conversations 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        return [{
            'timestamp': row[0],
            'user': row[1],
            'llm': row[2]
        } for row in results]

def display_llm_chat_interface(df):
    """Interface chat avec LLM local intégré"""
    st.subheader("🤖 Local AI Chat - Reasoning & Analysis")
    st.markdown("**Chat with a local AI model that can reason about your data**")
    
    # Initialisation système LLM
    if 'llm_chat_system' not in st.session_state:
        st.session_state.llm_chat_system = LocalLLMChat(df)
    else:
        # Mise à jour DataFrame
        st.session_state.llm_chat_system.df = df
    
    llm_chat = st.session_state.llm_chat_system
    
    # Section de contrôle du modèle
    with st.expander("🔧 AI Model Controls", expanded=not llm_chat.model_loaded):
        col1, col2 = st.columns(2)
        
        with col1:
            if not llm_chat.model_loaded:
                st.markdown("**🚀 Load AI Model**")
                model_choice = st.selectbox(
                    "Choose Model Size:",
                    [
                        "microsoft/DialoGPT-small",
                        "microsoft/DialoGPT-medium", 
                        "distilbert/distilgpt2"
                    ],
                    help="Small = Fast, Medium = Better quality"
                )
                
                if st.button("📥 Load AI Model", type="primary"):
                    success = llm_chat.load_model(model_choice)
                    if success:
                        st.rerun()
            else:
                st.success("✅ **AI Model Loaded & Ready**")
                st.info("🤖 You can now have intelligent conversations about your data!")
                
                if st.button("🔄 Reload Model"):
                    llm_chat.model_loaded = False
                    llm_chat.model = None
                    llm_chat.tokenizer = None
                    st.rerun()
        
        with col2:
            st.markdown("**📊 Current Data Context**")
            if df is not None:
                context_preview = llm_chat.analyze_data_for_context()
                st.text_area("Data Context:", context_preview, height=100, disabled=True)
            else:
                st.warning("No data loaded")
    
    # Interface de chat principale
    if llm_chat.model_loaded:
        st.markdown("---")
        
        # Historique des conversations
        if 'llm_chat_history' not in st.session_state:
            st.session_state.llm_chat_history = []
        
        # Affichage historique
        if st.session_state.llm_chat_history:
            st.markdown("### 💬 Conversation")
            
            for chat in st.session_state.llm_chat_history[-3:]:  # 3 dernières
                # Message utilisateur
                st.markdown(f"**🙋‍♂️ You:** {chat['user']}")
                
                # Réponse IA
                st.markdown(f"**🤖 AI:** {chat['llm']}")
                
                st.markdown("---")
        
        # Zone input
        user_input = st.text_area(
            "💬 Chat with AI about your data:",
            placeholder="Ask me anything about your data... e.g., 'What patterns do you see in the sales data?' or 'Can you reason about the correlation between price and quantity?'",
            height=80,
            key="llm_user_input"
        )
        
        # Suggestions de questions intelligentes
        if df is not None:
            st.markdown("**💡 Suggested Questions:**")
            suggestions = [
                "What insights can you find in this data?",
                "Explain the relationship between the main variables",
                "What trends or patterns do you notice?",
                "Can you reason about potential business implications?",
                "What recommendations would you make based on this data?"
            ]
            
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                with cols[i]:
                    if st.button(f"💡 {suggestion[:20]}...", key=f"suggestion_llm_{i}"):
                        st.session_state.suggested_llm_question = suggestion
                        st.rerun()
        
        # Traitement de la question
        if st.button("🚀 Send to AI", type="primary") and user_input:
            if df is None:
                st.error("❌ Please load data first (upload file or generate sample data)")
            else:
                with st.spinner("🤖 AI is thinking and reasoning..."):
                    # Génération réponse IA
                    ai_response = llm_chat.generate_response(user_input)
                
                # Ajout à l'historique
                chat_entry = {
                    'user': user_input,
                    'llm': ai_response,
                    'timestamp': datetime.now()
                }
                
                st.session_state.llm_chat_history.append(chat_entry)
                
                # Effacer input
                st.rerun()
        
        # Bouton suggestion automatique
        if st.session_state.get('suggested_llm_question'):
            st.text_area("Selected Question:", st.session_state.suggested_llm_question, height=60, key="selected_q")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Send This Question", type="primary"):
                    ai_response = llm_chat.generate_response(st.session_state.suggested_llm_question)
                    chat_entry = {
                        'user': st.session_state.suggested_llm_question,
                        'llm': ai_response,
                        'timestamp': datetime.now()
                    }
                    st.session_state.llm_chat_history.append(chat_entry)
                    del st.session_state.suggested_llm_question
                    st.rerun()
            with col2:
                if st.button("❌ Clear"):
                    del st.session_state.suggested_llm_question
                    st.rerun()
    
    else:
        st.info("👆 **Load an AI model above to start intelligent conversations about your data**")
        
        st.markdown("### 🎯 What you can do with Local AI:")
        st.markdown("""
        - **🧠 Deep reasoning** about your data patterns
        - **💡 Business insights** and recommendations  
        - **🔍 Complex analysis** with natural language
        - **🗨️ Natural conversations** about statistics and trends
        - **🔒 Complete privacy** - everything runs locally
        - **💰 Zero API costs** - no external dependencies
        """)
    
    # Status dans sidebar
    if 'sidebar_llm_status' not in st.session_state:
        st.session_state.sidebar_llm_status = llm_chat.model_loaded


def display_data_analysis(df):
    """Affiche l'analyse complète des données avec chat IA"""
    
    # Success message avec info
    st.success(f"✅ Data processed: **{len(df)} rows** and **{len(df.columns)} columns**")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("Numeric Columns", len(numeric_cols))
    with col4:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values)
    
    st.markdown("---")
    
    # Data preview
    st.subheader("2. 👀 Data Preview")
    st.markdown("**First 10 rows:**")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column information
    st.subheader("3. 📋 Column Analysis")
    
    col_info = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        non_null = df[col].count()
        unique_vals = df[col].nunique()
        
        col_info.append({
            'Column': col,
            'Type': col_type,
            'Non-Null': non_null,
            'Unique Values': unique_vals,
            'Missing': len(df) - non_null
        })
    
    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df, use_container_width=True)
    
    # TABS AVEC CHAT IA AJOUTÉ
    st.markdown("---")
    st.subheader("4. 📊 Data Analysis & AI Assistant")
    
    # Tabs avec Chat IA ajouté (6 onglets maintenant)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧹 Data Cleaning", 
        "🤖 Auto Charts", 
        "🎨 Custom Builder", 
        "🖼️ Gallery", 
        "💡 Suggestions", 
        "🤖 AI Chat"  # NOUVEAU TAB
    ])
    
    with tab1:
        # Section Data Cleaning (code existant)
        st.markdown("**Clean and optimize your data for better analysis**")
        
        if st.session_state.get('cleaning_applied', False) and st.session_state.get('cleaned_data') is not None:
            display_before_after_comparison(st.session_state.get('original_data', df), st.session_state.get('cleaned_data'))
            
            if st.button("🔄 Start New Cleaning Process"):
                st.session_state.cleaning_applied = False
                st.session_state.cleaned_data = None
                st.rerun()
        
        else:
            # Processus de nettoyage normal
            with st.spinner("🔍 Analyzing data quality..."):
                quality_report = analyze_data_quality(df)
            
            display_quality_report(df, quality_report)
            
            if quality_report['total_issues'] > 0:
                st.markdown("---")
                cleaned_df = display_cleaning_actions(df, quality_report)
            else:
                st.markdown("---")
                st.success("🎉 **Great!** Your data is already clean and ready for analysis.")
                
                if st.button("🔍 Show Detailed Analysis Anyway"):
                    st.info("Even clean data can benefit from optimization. Here are some advanced options:")
                    display_cleaning_actions(df, quality_report)
    
    with tab2:
        st.markdown("**Automatic visualizations based on your data:**")
        
        # Utiliser données nettoyées si disponibles
        analysis_df = st.session_state.get('cleaned_data', df) if st.session_state.get('cleaning_applied', False) else df
        
        # Graphiques automatiques existants (code existant)
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            first_numeric = numeric_cols[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Distribution: {first_numeric}**")
                fig1 = px.histogram(
                    analysis_df, 
                    x=first_numeric,
                    title=f"Distribution of {first_numeric}",
                    nbins=20
                )
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True, key="auto_hist")
            
            with col2:
                st.markdown(f"**Box Plot: {first_numeric}**")
                fig2 = px.box(
                    analysis_df,
                    y=first_numeric,
                    title=f"Box Plot of {first_numeric}"
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True, key="auto_box")
            
            # Matrice de corrélation
            if len(numeric_cols) > 1:
                st.markdown("**Correlation Matrix**")
                
                corr_matrix = analysis_df[numeric_cols].corr()
                
                fig3 = px.imshow(
                    corr_matrix,
                    title="Correlation Between Numeric Columns",
                    aspect="auto",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig3, use_container_width=True, key="auto_corr")
                
                # Corrélations fortes
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corr.append({
                                'Column 1': corr_matrix.columns[i],
                                'Column 2': corr_matrix.columns[j],
                                'Correlation': round(corr_val, 3)
                            })
                
                if strong_corr:
                    st.markdown("**🔍 Strong Correlations Found:**")
                    st.dataframe(pd.DataFrame(strong_corr))
                else:
                    st.info("No strong correlations (>0.7) found.")
            
            # Analyse temporelle
            date_columns = []
            for col in analysis_df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or analysis_df[col].dtype == 'datetime64[ns]':
                    try:
                        if analysis_df[col].dtype != 'datetime64[ns]':
                            pd.to_datetime(analysis_df[col])
                        date_columns.append(col)
                    except:
                        pass
            
            if date_columns and len(numeric_cols) > 0:
                st.markdown("**📅 Time Series Analysis**")
                date_col = date_columns[0]
                
                analysis_df_time = analysis_df.copy()
                if analysis_df_time[date_col].dtype != 'datetime64[ns]':
                    analysis_df_time[date_col] = pd.to_datetime(analysis_df_time[date_col])
                
                # Agrégation par date
                time_series = analysis_df_time.groupby(analysis_df_time[date_col].dt.date)[first_numeric].sum().reset_index()
                
                fig4 = px.line(
                    time_series,
                    x=date_col,
                    y=first_numeric,
                    title=f"Time Series: {first_numeric} over {date_col}"
                )
                st.plotly_chart(fig4, use_container_width=True, key="auto_time")
        else:
            st.info("No numeric columns found for automatic visualizations.")
    
    with tab3:
        # Custom Builder avec données nettoyées
        analysis_df = st.session_state.get('cleaned_data', df) if st.session_state.get('cleaning_applied', False) else df
        display_chart_builder(analysis_df)
    
    with tab4:
        # Gallery avec données nettoyées  
        analysis_df = st.session_state.get('cleaned_data', df) if st.session_state.get('cleaning_applied', False) else df
        display_chart_gallery(analysis_df)
    
    with tab5:
        # Suggestions avec données nettoyées
        analysis_df = st.session_state.get('cleaned_data', df) if st.session_state.get('cleaning_applied', False) else df
        suggest_chart_ideas(analysis_df)
    
    with tab6:
         st.markdown("**Intelligent AI reasoning about your data**")
        display_llm_chat_interface(chat_df)

    
    # Insights intelligents (reste identique mais avec mention chat)
    st.markdown("---")
    st.subheader("5. 🧠 Smart Insights")
    
    # Utiliser données nettoyées pour insights si disponibles
    insight_df = st.session_state.get('cleaned_data', df) if st.session_state.get('cleaning_applied', False) else df
    numeric_cols = insight_df.select_dtypes(include=[np.number]).columns
    
    insights = []
    
    # Complétude des données
    completeness = (1 - insight_df.isnull().sum().sum() / (len(insight_df) * len(insight_df.columns))) * 100
    insights.append(f"📊 Data is **{completeness:.1f}% complete** - {'Excellent!' if completeness > 95 else 'Good!' if completeness > 80 else 'Consider cleaning missing values'}")
    
    # Message si données nettoyées utilisées
    if st.session_state.get('cleaning_applied', False):
        insights.append("✨ **Using cleaned data** - Insights based on optimized dataset")
    
    # Nouvelle suggestion pour chat IA
    insights.append("🤖 **Try the AI Chat** - Ask questions like 'What's the average sales?' or 'Show me correlations'")
    
    # Analyse colonnes numériques
    if len(numeric_cols) > 0:
        high_variance_cols = []
        for col in numeric_cols:
            cv = insight_df[col].std() / insight_df[col].mean() if insight_df[col].mean() != 0 else 0
            if cv > 1:
                high_variance_cols.append(col)
        
        if high_variance_cols:
            insights.append(f"📈 High variability in: **{', '.join(high_variance_cols)}** - Great for analysis!")
    
    # Analyse catégorielle
    categorical_cols = insight_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        high_cardinality = []
        for col in categorical_cols:
            if insight_df[col].nunique() > len(insight_df) * 0.8:
                high_cardinality.append(col)
        
        if high_cardinality:
            insights.append(f"🏷️ High cardinality in: **{', '.join(high_cardinality)}** - Consider grouping")
    
    # Potentiel temporel
    date_columns = []
    for col in insight_df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or insight_df[col].dtype == 'datetime64[ns]':
            date_columns.append(col)
    
    if date_columns:
        insights.append(f"📅 Time series potential with **{date_columns[0]}** - Perfect for trend analysis!")
    
    # Affichage insights
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Next steps mis à jour avec chat
    st.markdown("---")
    st.subheader("6. 🚀 Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🤖 AI Chat**")
        st.success("✅ Available now!")
        if st.button("💬 Go to Chat", key="goto_chat_btn"):
            st.info("👆 Click on the '🤖 AI Chat' tab above to start chatting!")
    
    with col2:
        st.markdown("**📊 Create Charts**") 
        st.success("✅ Available now!")
        if st.button("🎨 Go to Builder", key="goto_builder_btn"):
            st.info("👆 Use the '🎨 Custom Builder' tab above!")
    
    with col3:
        st.markdown("**📄 Export Report**")
        st.info("Coming in next version")
        st.button("Soon!", disabled=True, key="export_btn")
    
    # Note sur données nettoyées et chat IA
    if st.session_state.get('cleaning_applied', False):
        st.success("✅ **Note:** All analysis, visualizations and AI Chat are using your cleaned data for maximum accuracy!")
    else:
        st.info("💡 **Tip:** Try the Data Cleaning tab first, then use AI Chat to explore your optimized data!")

# MODIFICATION SIDEBAR pour inclure Chat IA
# Trouvez votre section sidebar et mettez à jour la liste des features

def update_sidebar_with_chat():
    """Met à jour la sidebar avec chat IA"""
    with st.sidebar:
        st.markdown("### 🚀 Features")
        st.markdown("✅ Excel & CSV Upload")
        st.markdown("✅ Data Cleaning System")
        st.markdown("✅ Sample Data Generator") 
        st.markdown("✅ Auto Data Analysis") 
        st.markdown("✅ Custom Chart Builder")
        st.markdown("✅ Smart Visualizations")
        st.markdown("✅ **AI Chat Assistant**")  # NOUVEAU - mis en évidence
        st.markdown("🔄 Memory System (Coming Soon)")
        
        st.markdown("---")
        st.markdown("### 🤖 AI Assistant Status")
        
        # Détection Ollama pour info sidebar
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=1)
            ollama_status = response.status_code == 200
        except:
            ollama_status = False
            
        if ollama_status:
            st.success("🚀 Conversational Mode Available")
            st.markdown("*Ollama detected - Full AI chat enabled*")
        else:
            st.info("🧠 Smart Mode Active") 
            st.markdown("*Install Ollama for conversational mode*")
        
        st.markdown("---")
        st.markdown("### 📊 Current Data Status")
        
        if st.session_state.processed_data is not None:
            st.metric("Files Processed", st.session_state.upload_count)
            st.info(f"Source: {st.session_state.data_source}")
            st.metric("Rows", len(st.session_state.processed_data))
            st.metric("Columns", len(st.session_state.processed_data.columns))
            
            # Statut nettoyage
            if st.session_state.get('cleaning_applied', False):
                st.markdown("### 🧹 Cleaning Status")
                st.success("✅ Data Cleaned")
                
            # Statut chat
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                st.markdown("### 💬 Chat Activity")
                st.metric("Questions Asked", len(st.session_state.chat_history))
                st.caption(f"Last: {st.session_state.chat_history[-1]['timestamp'].strftime('%H:%M')}")
        else:
            st.metric("Files Processed", st.session_state.upload_count)
            
# FONCTION D'INITIALISATION POUR LE CHAT
# Ajoutez ceci dans votre fonction main() au début

def initialize_chat_session_state():
    """Initialise les variables session pour le chat"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_system' not in st.session_state:
        st.session_state.chat_system = None
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = "🔄 Hybrid (Best of Both)"

def update_sidebar_info():
    """Met à jour les infos sidebar avec statut nettoyage"""
    with st.sidebar:
        st.markdown("### 🚀 Features")
        st.markdown("✅ Excel & CSV Upload")
        st.markdown("✅ Data Cleaning System") 
        st.markdown("✅ Sample Data Generator") 
        st.markdown("✅ Auto Data Analysis") 
        st.markdown("✅ Custom Chart Builder")
        st.markdown("✅ Smart Visualizations")
        st.markdown("🔄 AI Chat (Coming Soon)")
        st.markdown("🧠 Memory System (Coming Soon)")
        
        st.markdown("---")
        st.markdown("### 📊 Current Data Status")
        
        if st.session_state.processed_data is not None:
            # Infos données principales
            st.metric("Files Processed", st.session_state.upload_count)
            st.info(f"Source: {st.session_state.data_source}")
            st.metric("Rows", len(st.session_state.processed_data))
            st.metric("Columns", len(st.session_state.processed_data.columns))
            
            # Statut nettoyage
            if st.session_state.get('cleaning_applied', False):
                cleaned_data = st.session_state.get('cleaned_data')
                original_data = st.session_state.get('original_data')
                
                if cleaned_data is not None and original_data is not None:
                    st.markdown("### 🧹 Cleaning Status")
                    st.success("✅ Data Cleaned")
                    
                    rows_change = len(original_data) - len(cleaned_data)
                    cols_change = len(original_data.columns) - len(cleaned_data.columns)
                    
                    if rows_change > 0:
                        st.markdown(f"📉 Removed {rows_change} rows")
                    if cols_change > 0:
                        st.markdown(f"📉 Removed {cols_change} columns")
                    
                    missing_before = original_data.isnull().sum().sum()
                    missing_after = cleaned_data.isnull().sum().sum()
                    if missing_before > missing_after:
                        st.markdown(f"✅ Fixed {missing_before - missing_after} missing values")
        else:
            st.metric("Files Processed", st.session_state.upload_count)


def initialize_cleaning_session_state():
    """Initialise les variables session pour le nettoyage"""
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'cleaning_applied' not in st.session_state:
        st.session_state.cleaning_applied = False
    if 'preview_cleaning' not in st.session_state:
        st.session_state.preview_cleaning = False


class HybridChatSystem:
    def __init__(self, df=None):
        self.df = df
        self.conversation_history = []
        self.init_chat_memory()
        
    def init_chat_memory(self):
        """Initialise base de données pour mémoire chat"""
        self.chat_conn = sqlite3.connect('chat_memory.db', check_same_thread=False)
        cursor = self.chat_conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                user_question TEXT,
                bot_response TEXT,
                response_type TEXT,
                data_context TEXT
            )
        ''')
        self.chat_conn.commit()
    
    def detect_ollama_availability(self) -> bool:
        """Vérifie si Ollama est disponible localement"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def analyze_question_pattern(self, question: str) -> Dict:
        """Analyse le pattern de la question pour déterminer le type de réponse"""
        question_lower = question.lower()
        
        patterns = {
            'statistics': ['moyenne', 'mean', 'average', 'total', 'sum', 'max', 'min', 'médiane', 'median'],
            'trends': ['tendance', 'trend', 'évolution', 'evolution', 'croissance', 'growth', 'baisse', 'decrease'],
            'correlation': ['corrélation', 'correlation', 'relation', 'lien', 'link'],
            'distribution': ['distribution', 'répartition', 'spread', 'outlier', 'aberrant'],
            'comparison': ['compare', 'comparer', 'différence', 'difference', 'versus', 'vs'],
            'visualization': ['graphique', 'chart', 'plot', 'visualisation', 'graph'],
            'insights': ['insight', 'analyse', 'analysis', 'pattern', 'découverte']
        }
        
        detected_patterns = []
        for pattern_type, keywords in patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_patterns.append(pattern_type)
        
        # Détection des colonnes mentionnées
        mentioned_columns = []
        if self.df is not None:
            for col in self.df.columns:
                if col.lower() in question_lower:
                    mentioned_columns.append(col)
        
        return {
            'patterns': detected_patterns,
            'columns': mentioned_columns,
            'complexity': 'simple' if len(detected_patterns) <= 1 else 'complex',
            'requires_calculation': any(p in detected_patterns for p in ['statistics', 'trends', 'correlation'])
        }
    
    def intelligent_response(self, question: str) -> Dict:
        """Génère réponse intelligente basée sur analyse des données"""
        if self.df is None:
            return {
                'response': "❌ Aucune donnée chargée. Veuillez d'abord uploader un fichier ou générer des données d'exemple.",
                'type': 'error',
                'suggestions': ['Upload Excel/CSV', 'Generate sample data']
            }
        
        analysis = self.analyze_question_pattern(question)
        
        # Réponses pour statistiques de base
        if 'statistics' in analysis['patterns']:
            return self.generate_statistics_response(question, analysis)
        
        # Réponses pour tendances
        elif 'trends' in analysis['patterns']:
            return self.generate_trends_response(question, analysis)
        
        # Réponses pour corrélations
        elif 'correlation' in analysis['patterns']:
            return self.generate_correlation_response(question, analysis)
        
        # Réponses pour visualisations
        elif 'visualization' in analysis['patterns']:
            return self.generate_visualization_response(question, analysis)
        
        # Réponse générale pour questions complexes
        else:
            return self.generate_general_response(question, analysis)
    
    def generate_statistics_response(self, question: str, analysis: Dict) -> Dict:
        """Génère réponse pour questions statistiques"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if not analysis['columns'] and len(numeric_cols) > 0:
            # Utilise première colonne numérique si pas spécifié
            target_col = numeric_cols[0]
        elif analysis['columns']:
            target_col = analysis['columns'][0]
        else:
            return {'response': "❌ Aucune colonne numérique trouvée pour calculer les statistiques.", 'type': 'error'}
        
        try:
            if target_col in numeric_cols:
                mean_val = self.df[target_col].mean()
                median_val = self.df[target_col].median()
                max_val = self.df[target_col].max()
                min_val = self.df[target_col].min()
                std_val = self.df[target_col].std()
                
                response = f"""📊 **Statistiques pour {target_col}:**

• **Moyenne:** {mean_val:.2f}
• **Médiane:** {median_val:.2f}
• **Maximum:** {max_val:.2f}
• **Minimum:** {min_val:.2f}
• **Écart-type:** {std_val:.2f}

📈 **Interprétation:** {"Données très variables" if std_val/mean_val > 0.3 else "Données relativement stables"}"""
                
                suggestions = [
                    f"Créer histogramme de {target_col}",
                    f"Analyser outliers dans {target_col}",
                    "Voir corrélations avec autres colonnes"
                ]
                
                return {'response': response, 'type': 'statistics', 'suggestions': suggestions}
        except Exception as e:
            return {'response': f"❌ Erreur calcul statistiques: {str(e)}", 'type': 'error'}
    
    def generate_trends_response(self, question: str, analysis: Dict) -> Dict:
        """Génère réponse pour questions de tendance"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or self.df[col].dtype == 'datetime64[ns]']
        
        if not date_cols:
            return {'response': "❌ Aucune colonne date trouvée pour analyser les tendances.", 'type': 'error'}
        
        if not analysis['columns'] and len(numeric_cols) > 0:
            target_col = numeric_cols[0]
        elif analysis['columns']:
            target_col = analysis['columns'][0]
        else:
            return {'response': "❌ Veuillez spécifier une colonne numérique pour l'analyse de tendance.", 'type': 'error'}
        
        try:
            date_col = date_cols[0]
            
            # Conversion en datetime si nécessaire
            if self.df[date_col].dtype != 'datetime64[ns]':
                df_temp = self.df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            else:
                df_temp = self.df
            
            # Calcul tendance (régression simple)
            df_sorted = df_temp.sort_values(date_col)
            values = df_sorted[target_col].dropna()
            
            if len(values) < 2:
                return {'response': "❌ Pas assez de données pour calculer une tendance.", 'type': 'error'}
            
            # Calcul pente approximative
            first_half = values[:len(values)//2].mean()
            second_half = values[len(values)//2:].mean()
            trend_direction = "📈 CROISSANTE" if second_half > first_half else "📉 DÉCROISSANTE"
            trend_percentage = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
            
            response = f"""📊 **Analyse de tendance - {target_col}:**

🎯 **Direction:** {trend_direction}
📈 **Variation:** {abs(trend_percentage):.1f}% {"d'augmentation" if trend_percentage > 0 else "de diminution"}
📅 **Période:** {df_sorted[date_col].min().strftime('%Y-%m-%d')} → {df_sorted[date_col].max().strftime('%Y-%m-%d')}
📊 **Valeur actuelle:** {values.iloc[-1]:.2f}

💡 **Conseil:** {"Tendance positive, maintenez le cap!" if trend_percentage > 0 else "Tendance négative, analysez les causes."}"""
            
            suggestions = [
                f"Créer graphique tendance {target_col}",
                "Analyser saisonnalité",
                "Identifier facteurs d'influence"
            ]
            
            return {'response': response, 'type': 'trends', 'suggestions': suggestions}
            
        except Exception as e:
            return {'response': f"❌ Erreur analyse tendance: {str(e)}", 'type': 'error'}
    
    def generate_correlation_response(self, question: str, analysis: Dict) -> Dict:
        """Génère réponse pour questions de corrélation"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            return {'response': "❌ Au moins 2 colonnes numériques nécessaires pour calculer les corrélations.", 'type': 'error'}
        
        try:
            corr_matrix = self.df[numeric_cols].corr()
            
            # Trouve les corrélations les plus fortes
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Seuil corrélation significative
                        strong_correlations.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            if strong_correlations:
                response = "🔍 **Corrélations significatives trouvées:**\n\n"
                
                for corr in sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)[:3]:
                    strength = "très forte" if abs(corr['correlation']) > 0.8 else "forte" if abs(corr['correlation']) > 0.6 else "modérée"
                    direction = "positive" if corr['correlation'] > 0 else "négative"
                    
                    response += f"• **{corr['col1']}** ↔ **{corr['col2']}**: {corr['correlation']:.2f} ({strength}, {direction})\n"
                
                response += f"\n💡 **Interprétation:** Les variables les plus corrélées évoluent ensemble."
                
                suggestions = [
                    "Créer matrice corrélation visuelle",
                    "Analyser relation cause-effet",
                    "Créer scatter plots"
                ]
            else:
                response = "📊 **Aucune corrélation forte détectée.**\n\nVos variables semblent relativement indépendantes les unes des autres."
                suggestions = ["Voir corrélations faibles", "Analyser par segments"]
            
            return {'response': response, 'type': 'correlation', 'suggestions': suggestions}
            
        except Exception as e:
            return {'response': f"❌ Erreur analyse corrélation: {str(e)}", 'type': 'error'}
    
    def generate_visualization_response(self, question: str, analysis: Dict) -> Dict:
        """Génère réponse pour demandes de visualisation"""
        suggestions = []
        
        # Suggestions basées sur types de colonnes
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        
        if len(numeric_cols) > 0:
            suggestions.extend([
                f"Histogramme de {numeric_cols[0]}",
                f"Box plot de {numeric_cols[0]}"
            ])
        
        if len(categorical_cols) > 0:
            suggestions.extend([
                f"Graphique en barres par {categorical_cols[0]}",
                f"Pie chart de {categorical_cols[0]}"
            ])
        
        if date_cols and len(numeric_cols) > 0:
            suggestions.append(f"Tendance temporelle {numeric_cols[0]}")
        
        response = """🎨 **Créons une visualisation !**

Utilisez l'onglet **"🎨 Custom Builder"** ci-dessus pour créer des graphiques personnalisés.

📊 **Suggestions basées sur vos données:**"""
        
        for i, suggestion in enumerate(suggestions[:5], 1):
            response += f"\n{i}. {suggestion}"
        
        return {'response': response, 'type': 'visualization', 'suggestions': suggestions}
    
    def generate_general_response(self, question: str, analysis: Dict) -> Dict:
        """Génère réponse générale pour questions complexes"""
        # Insights automatiques sur les données
        insights = []
        
        # Aperçu dataset
        insights.append(f"📊 Votre dataset contient **{len(self.df)} lignes** et **{len(self.df.columns)} colonnes**")
        
        # Colonnes par type
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            insights.append(f"📈 **{len(numeric_cols)} colonnes numériques:** {', '.join(numeric_cols[:3])}")
        
        if len(categorical_cols) > 0:
            insights.append(f"🏷️ **{len(categorical_cols)} colonnes catégorielles:** {', '.join(categorical_cols[:3])}")
        
        # Données manquantes
        missing_data = self.df.isnull().sum().sum()
        if missing_data > 0:
            insights.append(f"⚠️ **{missing_data} valeurs manquantes** détectées")
        else:
            insights.append("✅ **Aucune donnée manquante**")
        
        response = f"""🤖 **À propos de vos données:**

{chr(10).join(f"• {insight}" for insight in insights)}

❓ **Questions que vous pouvez poser:**
• "Quelle est la moyenne de [colonne] ?"
• "Montre-moi la tendance des [données] ?"
• "Y a-t-il des corrélations ?"
• "Crée un graphique de [colonnes]"

💡 Essayez d'être spécifique avec les noms de colonnes !"""
        
        return {'response': response, 'type': 'general', 'suggestions': [
            "Voir statistiques générales",
            "Analyser qualité des données", 
            "Explorer corrélations"
        ]}
    
    def chat_with_ollama(self, question: str, context: str) -> str:
        """Utilise Ollama pour réponse conversationnelle"""
        try:
            prompt = f"""Tu es un assistant IA spécialisé dans l'analyse de données. 
            
Contexte des données: {context}
Question utilisateur: {question}

Réponds de manière claire et concise, en français, en te basant uniquement sur les données fournies. 
Si tu ne peux pas répondre avec certitude, dis-le clairement."""
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama2',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.1}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'Erreur génération réponse')
            else:
                return "Erreur communication avec Ollama"
                
        except Exception as e:
            return f"Erreur Ollama: {str(e)}"
    
    def prepare_data_context(self) -> str:
        """Prépare contexte des données pour LLM"""
        if self.df is None:
            return "Aucune donnée disponible"
        
        context = f"Dataset: {len(self.df)} lignes, {len(self.df.columns)} colonnes\n"
        context += f"Colonnes: {', '.join(self.df.columns)}\n"
        
        # Statistiques de base pour colonnes numériques
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:3]:  # Limité à 3 colonnes pour éviter contexte trop long
            context += f"{col}: moyenne={self.df[col].mean():.2f}, min={self.df[col].min()}, max={self.df[col].max()}\n"
        
        return context
    
    def save_conversation(self, question: str, response: str, response_type: str):
        """Sauvegarde conversation en base"""
        cursor = self.chat_conn.cursor()
        cursor.execute('''
            INSERT INTO chat_conversations (timestamp, user_question, bot_response, response_type, data_context)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), question, response, response_type, str(len(self.df)) if self.df is not None else "None"))
        self.chat_conn.commit()
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Récupère historique conversations"""
        cursor = self.chat_conn.cursor()
        cursor.execute('''
            SELECT timestamp, user_question, bot_response, response_type
            FROM chat_conversations 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        return [{
            'timestamp': row[0],
            'question': row[1],
            'response': row[2],
            'type': row[3]
        } for row in results]

def display_chat_interface(df):
    """Interface chat complète avec choix du mode"""
    st.subheader("🤖 AI Data Assistant")
    
    # Initialisation système chat
    if 'chat_system' not in st.session_state:
        st.session_state.chat_system = HybridChatSystem(df)
    else:
        # Mise à jour DataFrame si changé
        st.session_state.chat_system.df = df
    
    chat_system = st.session_state.chat_system
    
    # Interface de choix du mode
    with st.expander("⚙️ Chat Mode Settings", expanded=False):
        st.markdown("**Choose your AI experience:**")
        
        # Détection Ollama
        ollama_available = chat_system.detect_ollama_availability()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧠 Smart Mode (Recommended)")
            st.info("""
            **✅ Always available**
            • Instant responses (< 1 second)
            • Precise calculations on your data
            • Works everywhere (no setup needed)
            • Based on statistical analysis
            """)
        
        with col2:
            st.markdown("#### 🚀 Conversational Mode")
            if ollama_available:
                st.success("""
                **✅ Ollama detected!**
                • Natural language responses
                • Creative explanations
                • More flexible conversations
                • Requires local Ollama installation
                """)
            else:
                st.warning("""
                **❌ Ollama not available**
                • Install Ollama locally for this mode
                • More natural conversations
                • Falls back to Smart Mode automatically
                """)
        
        # Sélection mode
        if ollama_available:
            chat_mode = st.radio(
                "Select Mode:",
                ["🧠 Smart Mode", "🚀 Conversational Mode", "🔄 Hybrid (Best of Both)"],
                index=2,
                horizontal=True
            )
        else:
            chat_mode = st.radio(
                "Select Mode:",
                ["🧠 Smart Mode", "🔄 Hybrid (Fallback to Smart)"],
                index=1,
                horizontal=True
            )
        
        st.session_state.chat_mode = chat_mode
    
    # Zone de chat principale
    st.markdown("---")
    
    # Historique des conversations
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Affichage historique
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # 5 dernières conversations
            with st.container():
                st.markdown(f"**🙋‍♂️ You:** {chat['question']}")
                st.markdown(f"**🤖 AI:** {chat['response']}")
                
                if chat.get('suggestions'):
                    cols = st.columns(len(chat['suggestions']))
                    for j, suggestion in enumerate(chat['suggestions'][:3]):
                        with cols[j]:
                            if st.button(f"💡 {suggestion}", key=f"suggestion_{i}_{j}"):
                                st.session_state.suggested_question = suggestion
                                st.rerun()
                st.markdown("---")
    
    # Input utilisateur
    question_input = st.text_input(
        "💬 Ask about your data:",
        placeholder="e.g., What's the average sales? Show me trends. Any correlations?",
        value=st.session_state.get('suggested_question', ''),
        key="user_question"
    )
    
    # Suggestions de questions
    if df is not None:
        st.markdown("**💡 Quick Questions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Dataset Overview"):
                st.session_state.suggested_question = "Tell me about this dataset"
                st.rerun()
        
        with col2:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                if st.button(f"📈 {numeric_cols[0]} Stats"):
                    st.session_state.suggested_question = f"What are the statistics for {numeric_cols[0]}?"
                    st.rerun()
        
        with col3:
            if len(numeric_cols) > 1:
                if st.button("🔍 Find Correlations"):
                    st.session_state.suggested_question = "Show me correlations between columns"
                    st.rerun()
        
        with col4:
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols and len(numeric_cols) > 0:
                if st.button("📈 Show Trends"):
                    st.session_state.suggested_question = f"What's the trend for {numeric_cols[0]}?"
                    st.rerun()
    
    # Traitement question
    if question_input and st.button("💬 Send", type="primary"):
        with st.spinner("🤖 Thinking..."):
            # Choix du mode de réponse
            mode = st.session_state.get('chat_mode', '🧠 Smart Mode')
            
            if mode == "🚀 Conversational Mode" and ollama_available:
                # Mode conversationnel avec Ollama
                context = chat_system.prepare_data_context()
                ai_response = chat_system.chat_with_ollama(question_input, context)
                response_type = "ollama"
                suggestions = ["Ask another question", "Create visualization", "Analyze further"]
                
            elif mode == "🔄 Hybrid (Best of Both)" or mode == "🔄 Hybrid (Fallback to Smart)":
                # Mode hybride
                intelligent_result = chat_system.intelligent_response(question_input)
                
                if intelligent_result['type'] in ['statistics', 'trends', 'correlation'] and ollama_available:
                    # Questions simples → réponse intelligente rapide
                    ai_response = intelligent_result['response']
                    suggestions = intelligent_result.get('suggestions', [])
                    response_type = "intelligent"
                elif ollama_available:
                    # Questions complexes → Ollama si disponible
                    context = chat_system.prepare_data_context()
                    ai_response = chat_system.chat_with_ollama(question_input, context)
                    suggestions = ["Ask another question", "Get statistics", "Show visualization"]
                    response_type = "ollama"
                else:
                    # Fallback → réponse intelligente
                    ai_response = intelligent_result['response']
                    suggestions = intelligent_result.get('suggestions', [])
                    response_type = "intelligent"
            
            else:
                # Mode smart uniquement
                intelligent_result = chat_system.intelligent_response(question_input)
                ai_response = intelligent_result['response']
                suggestions = intelligent_result.get('suggestions', [])
                response_type = "intelligent"
        
        # Ajout à l'historique
        chat_entry = {
            'question': question_input,
            'response': ai_response,
            'type': response_type,
            'suggestions': suggestions,
            'timestamp': datetime.now()
        }
        
        st.session_state.chat_history.append(chat_entry)
        chat_system.save_conversation(question_input, ai_response, response_type)
        
        # Clear suggested question
        if 'suggested_question' in st.session_state:
            del st.session_state.suggested_question
        
        st.rerun()
    
    # Message si pas de données
    if df is None:
        st.info("💡 Upload data or generate sample data to start chatting with your dataset!")


# Interface principale
st.subheader("1. 📁 Get Your Data")

# Tabs pour organiser les options
tab1, tab2 = st.tabs(["📁 Upload File", "🎯 Generate Sample Data"])

with tab1:
    st.markdown("**Upload your Excel or CSV file:**")
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=['xlsx', 'xls', 'csv'],
        help="Supported formats: Excel (.xlsx, .xls) and CSV (.csv)"
    )
    
    if uploaded_file is not None:
        with st.spinner("📊 Processing your file..."):
            df, source_name = load_file_data(uploaded_file)
            
        if df is not None:
            df = process_data(df, source_name)

with tab2:
    display_sample_data_generation()

# Affichage de l'analyse si données disponibles
if st.session_state.processed_data is not None:
    st.markdown("---")
    display_data_analysis(st.session_state.processed_data)
    
    # Option de téléchargement des données générées
    if "Generated" in st.session_state.data_source:
        st.markdown("---")
        st.subheader("📥 Download Generated Data")
        
        # CSV download
        csv = st.session_state.processed_data.to_csv(index=False)
        filename = "sample_ecommerce_data.csv" if "E-commerce" in st.session_state.data_source else "sample_healthcare_data.csv"
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=filename,
            mime='text/csv',
            use_container_width=True
        )

else:
    # Instructions quand pas de données
    st.info("👆 Upload a file or generate sample data to get started!")
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide:")
    st.markdown("1. **Option A:** Upload your own Excel/CSV file using the 'Upload File' tab")
    st.markdown("2. **Option B:** Generate sample data using the 'Generate Sample Data' tab")
    st.markdown("3. **Explore:** View automatic analysis, insights, and visualizations")
    st.markdown("4. **Next:** Tomorrow we add AI chat to query your data!")

# Clear data option
if st.session_state.processed_data is not None:
    st.markdown("---")
    if st.button("🗑️ Clear Current Data"):
        st.session_state.processed_data = None
        st.session_state.data_source = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Excel AI Dashboard v1.1** - Transform your data into insights ⚡")
st.markdown("*Support: Excel (.xlsx, .xls), CSV (.csv) | Sample data generators included*")