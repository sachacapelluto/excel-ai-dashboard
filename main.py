# main.py - Excel AI Dashboard avec CSV et génération directe
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration page
st.set_page_config(
    page_title="Excel AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.markdown("✅ Sample Data Generator") 
    st.markdown("✅ Auto Data Analysis") 
    st.markdown("✅ Smart Visualizations")
    st.markdown("🔄 AI Chat (Coming Soon)")
    st.markdown("🧠 Memory System (Coming Soon)")
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    st.metric("Files Processed", st.session_state.upload_count)
    
    if st.session_state.processed_data is not None:
        st.markdown("### 📁 Current Data")
        st.info(f"Source: {st.session_state.data_source}")
        st.metric("Rows", len(st.session_state.processed_data))
        st.metric("Columns", len(st.session_state.processed_data.columns))

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



def create_custom_chart(df, chart_type, x_col, y_col, color_col=None, size_col=None, title=""):
    """Crée un graphique personnalisé selon les paramètres"""
    try:
        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title or f"{y_col} over {x_col}")
            
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title or f"{y_col} by {x_col}")
            
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, 
                           title=title or f"{y_col} vs {x_col}")
            
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_col, title=title or f"Distribution of {x_col}")
            
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title or f"{y_col} distribution by {x_col}")
            
        elif chart_type == "Pie Chart":
            # Agrégation pour pie chart
            if y_col:
                pie_data = df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.pie(pie_data, values=y_col, names=x_col, title=title or f"{y_col} by {x_col}")
            else:
                value_counts = df[x_col].value_counts().reset_index()
                fig = px.pie(value_counts, values='count', names=x_col, title=title or f"Distribution of {x_col}")
                
        elif chart_type == "Heatmap":
            if df[x_col].dtype in ['object'] and df[y_col].dtype in ['object']:
                # Cross-tabulation pour variables catégorielles
                heatmap_data = pd.crosstab(df[x_col], df[y_col])
            else:
                # Corrélation ou pivot
                heatmap_data = df.pivot_table(values=color_col if color_col else df.columns[0], 
                                            index=x_col, columns=y_col, aggfunc='mean')
            
            fig = px.imshow(heatmap_data, title=title or f"Heatmap: {x_col} vs {y_col}")
            
        elif chart_type == "Area Chart":
            fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title or f"{y_col} area over {x_col}")
            
        elif chart_type == "Violin Plot":
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, title=title or f"{y_col} distribution by {x_col}")
            
        else:
            return None
            
        # Personnalisation commune
        fig.update_layout(height=500, showlegend=True)
        return fig
        
    except Exception as e:
        st.error(f"❌ Erreur création graphique: {str(e)}")
        return None

def display_chart_builder(df):
    """Interface de construction de graphiques personnalisés"""
    st.subheader("🎨 Custom Chart Builder")
    st.markdown("**Create your own visualizations:**")
    
    # Séparer colonnes par type
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
            datetime_cols.append(col)
    
    all_cols = df.columns.tolist()
    
    # Interface de sélection
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "📊 Chart Type",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", 
             "Box Plot", "Pie Chart", "Heatmap", "Area Chart", "Violin Plot"],
            help="Choose the type of visualization"
        )
        
        # Sélection intelligente selon le type de graphique
        if chart_type in ["Line Chart", "Area Chart"]:
            x_col = st.selectbox("X-Axis", datetime_cols + categorical_cols + numeric_cols, 
                               help="Usually dates or categories for line/area charts")
        elif chart_type == "Histogram":
            x_col = st.selectbox("Column", numeric_cols + categorical_cols,
                               help="Column to show distribution")
        elif chart_type == "Pie Chart":
            x_col = st.selectbox("Categories", categorical_cols,
                               help="Categorical column for pie slices")
        else:
            x_col = st.selectbox("X-Axis", all_cols)
    
    with col2:
        # Y-axis selon type de graphique
        if chart_type in ["Histogram", "Pie Chart"]:
            y_col = st.selectbox("Values (optional)", [None] + numeric_cols,
                               help="Leave empty to count occurrences")
        elif chart_type in ["Box Plot", "Violin Plot"]:
            y_col = st.selectbox("Y-Axis (Values)", numeric_cols,
                               help="Numeric column to analyze distribution")
        elif chart_type == "Heatmap":
            y_col = st.selectbox("Y-Axis", all_cols,
                               help="Second dimension for heatmap")
        else:
            y_col = st.selectbox("Y-Axis", numeric_cols + [None])
    
    # Options avancées
    with st.expander("🔧 Advanced Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color_col = st.selectbox("Color By", [None] + categorical_cols + numeric_cols,
                                   help="Add color dimension")
        
        with col2:
            if chart_type == "Scatter Plot":
                size_col = st.selectbox("Size By", [None] + numeric_cols,
                                      help="Size points by values")
            else:
                size_col = None
        
        with col3:
            custom_title = st.text_input("Custom Title", 
                                       placeholder="Leave empty for auto title")
    
    # Bouton création graphique
    if st.button("🎨 Create Chart", type="primary"):
        if x_col and (y_col or chart_type in ["Histogram", "Pie Chart"]):
            with st.spinner("Creating your custom chart..."):
                fig = create_custom_chart(df, chart_type, x_col, y_col, color_col, size_col, custom_title)
                
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Sauvegarder dans session state
                if 'custom_charts' not in st.session_state:
                    st.session_state.custom_charts = []
                
                chart_config = {
                    'type': chart_type,
                    'x_col': x_col,
                    'y_col': y_col,
                    'color_col': color_col,
                    'size_col': size_col,
                    'title': custom_title or f"{chart_type}: {y_col or x_col} by {x_col}",
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.custom_charts.append(chart_config)
                
                st.success(f"✅ {chart_type} created successfully!")
        else:
            st.warning("⚠️ Please select required columns for this chart type")

def display_chart_gallery(df):
    """Affiche galerie des graphiques créés"""
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
            for i, chart_config in enumerate(st.session_state.custom_charts[-6:]):  # 6 derniers
                with cols[i % 2]:
                    with st.container():
                        st.markdown(f"**{chart_config['title']}**")
                        st.caption(f"Created at {chart_config['timestamp']}")
                        
                        # Recréer le graphique
                        fig = create_custom_chart(
                            df, chart_config['type'], chart_config['x_col'], 
                            chart_config['y_col'], chart_config['color_col'], 
                            chart_config['size_col'], chart_config['title']
                        )
                        if fig:
                            fig.update_layout(height=300)  # Plus petit pour galerie
                            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Affichage en liste avec sélection
            for i, chart_config in enumerate(reversed(st.session_state.custom_charts)):
                with st.expander(f"📊 {chart_config['title']} - {chart_config['timestamp']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Détails du graphique
                        st.markdown(f"**Type:** {chart_config['type']}")
                        st.markdown(f"**X-Axis:** {chart_config['x_col']}")
                        st.markdown(f"**Y-Axis:** {chart_config['y_col'] or 'None'}")
                        if chart_config['color_col']:
                            st.markdown(f"**Color:** {chart_config['color_col']}")
                    
                    with col2:
                        if st.button("🔄 Recreate", key=f"recreate_{i}"):
                            fig = create_custom_chart(
                                df, chart_config['type'], chart_config['x_col'], 
                                chart_config['y_col'], chart_config['color_col'], 
                                chart_config['size_col'], chart_config['title']
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

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



def display_data_analysis(df):
    """Affiche l'analyse complète des données avec graphiques personnalisés"""
    
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
    
    # NOUVELLE SECTION - Tabs pour visualisations
    st.markdown("---")
    st.subheader("4. 📊 Data Visualizations")
    
    # Tabs pour différents types de visualisations
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Auto Charts", "🎨 Custom Builder", "🖼️ Gallery", "💡 Suggestions"])
    
    with tab1:
        st.markdown("**Automatic visualizations based on your data:**")
        
        # Graphiques automatiques existants
        if len(numeric_cols) > 0:
            first_numeric = numeric_cols[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Distribution: {first_numeric}**")
                fig1 = px.histogram(
                    df, 
                    x=first_numeric,
                    title=f"Distribution of {first_numeric}",
                    nbins=20
                )
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown(f"**Box Plot: {first_numeric}**")
                fig2 = px.box(
                    df,
                    y=first_numeric,
                    title=f"Box Plot of {first_numeric}"
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Matrice de corrélation
            if len(numeric_cols) > 1:
                st.markdown("**Correlation Matrix**")
                
                corr_matrix = df[numeric_cols].corr()
                
                fig3 = px.imshow(
                    corr_matrix,
                    title="Correlation Between Numeric Columns",
                    aspect="auto",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig3, use_container_width=True)
                
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
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or df[col].dtype == 'datetime64[ns]':
                    try:
                        if df[col].dtype != 'datetime64[ns]':
                            pd.to_datetime(df[col])
                        date_columns.append(col)
                    except:
                        pass
            
            if date_columns and len(numeric_cols) > 0:
                st.markdown("**📅 Time Series Analysis**")
                date_col = date_columns[0]
                
                df_time = df.copy()
                if df_time[date_col].dtype != 'datetime64[ns]':
                    df_time[date_col] = pd.to_datetime(df_time[date_col])
                
                # Agrégation par date
                time_series = df_time.groupby(df_time[date_col].dt.date)[first_numeric].sum().reset_index()
                
                fig4 = px.line(
                    time_series,
                    x=date_col,
                    y=first_numeric,
                    title=f"Time Series: {first_numeric} over {date_col}"
                )
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No numeric columns found for automatic visualizations.")
    
    with tab2:
        # Interface de construction de graphiques personnalisés
        display_chart_builder(df)
    
    with tab3:
        # Galerie des graphiques créés
        display_chart_gallery(df)
    
    with tab4:
        # Suggestions de graphiques
        suggest_chart_ideas(df)
    
    # Insights intelligents (reste identique)
    st.markdown("---")
    st.subheader("5. 🧠 Smart Insights")
    
    insights = []
    
    # Complétude des données
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    insights.append(f"📊 Data is **{completeness:.1f}% complete** - {'Excellent!' if completeness > 95 else 'Good!' if completeness > 80 else 'Consider cleaning missing values'}")
    
    # Analyse colonnes numériques
    if len(numeric_cols) > 0:
        high_variance_cols = []
        for col in numeric_cols:
            cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
            if cv > 1:
                high_variance_cols.append(col)
        
        if high_variance_cols:
            insights.append(f"📈 High variability in: **{', '.join(high_variance_cols)}** - Great for analysis!")
    
    # Analyse catégorielle
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        high_cardinality = []
        for col in categorical_cols:
            if df[col].nunique() > len(df) * 0.8:
                high_cardinality.append(col)
        
        if high_cardinality:
            insights.append(f"🏷️ High cardinality in: **{', '.join(high_cardinality)}** - Consider grouping")
    
    # Potentiel temporel
    date_columns = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or df[col].dtype == 'datetime64[ns]':
            date_columns.append(col)
    
    if date_columns:
        insights.append(f"📅 Time series potential with **{date_columns[0]}** - Perfect for trend analysis!")
    
    # Affichage insights
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Next steps (reste identique)
    st.markdown("---")
    st.subheader("6. 🚀 Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🤖 AI Chat**")
        st.info("Ask questions about your data")
        st.button("Coming Next!", disabled=True, key="chat_btn")
    
    with col2:
        st.markdown("**🧠 Memory System**") 
        st.info("System remembers analyses")
        st.button("Coming Next!", disabled=True, key="memory_btn")
    
    with col3:
        st.markdown("**📄 Export Report**")
        st.info("Generate PDF reports")
        st.button("Coming Next!", disabled=True, key="export_btn")


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
    st.markdown("**Generate sample data for testing:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🛍️ Generate E-commerce Data", use_container_width=True):
            with st.spinner("Generating e-commerce data..."):
                df = generate_ecommerce_data()
                df = process_data(df, "Generated E-commerce Dataset (90 days)")
            st.success("✅ E-commerce data generated and loaded!")
    
    with col2:
        if st.button("🏥 Generate Healthcare Data", use_container_width=True):
            with st.spinner("Generating healthcare data..."):
                df = generate_healthcare_data()
                df = process_data(df, "Generated Healthcare Dataset (60 days)")
            st.success("✅ Healthcare data generated and loaded!")
    
    st.markdown("---")
    st.markdown("💡 **Tip:** Click either button above to instantly generate and analyze sample data!")

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