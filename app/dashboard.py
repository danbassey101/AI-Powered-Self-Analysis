import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import json

# Add parent dir to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection import GitHubFetcher
from src.llm_analysis import OllamaAnalyzer
from src.traditional_ds import TraditionalAnalyzer

st.set_page_config(page_title="AI-GitHub Dashboard", layout="wide")

# --- Localization Setup ---
def load_translations(lang_code):
    """Loads the JSON translation file for the specified language code."""
    locale_path = os.path.join(os.path.dirname(__file__), '..', 'locales', f'{lang_code}.json')
    try:
        with open(locale_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to English if the requested language file doesn't exist
        fallback_path = os.path.join(os.path.dirname(__file__), '..', 'locales', 'en.json')
        try:
             with open(fallback_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {} # Return empty if everything fails

# Language Selector
# For now, we manually list supported languages. Lingo.dev will generate the actual JSONs.
LANGUAGES = {
    "en": "English",
    "es": "Español",
    "fr": "Français", 
    "de": "Deutsch"
}

st.sidebar.header("Language")
selected_lang_code = st.sidebar.selectbox("Select Language", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])

translations = load_translations(selected_lang_code)

def t(key, **kwargs):
    """
    Retrieves a translation for the given key.
    Supports simple string formatting via kwargs.
    Example: t("welcome_message", name="John")
    """
    text = translations.get(key, key) # Default to key if not found
    if text is None:
        return str(key)
        
    try:
        return str(text).format(**kwargs)
    except KeyError:
        return str(text) # Return unformatted text if kwargs are missing

# --- End Localization Setup ---


st.title(t("main_title"))

# Sidebar for Configuration
st.sidebar.header(t("sidebar_header"))
username = st.sidebar.text_input(t("username_label"), value=os.getenv("GITHUB_USERNAME", ""))
token = st.sidebar.text_input(t("token_label"), value=os.getenv("GITHUB_TOKEN", ""), type="password")

if st.sidebar.button(t("fetch_data_button")):
    with st.spinner(t("fetching_spinner")):
        fetcher = GitHubFetcher(username=username, token=token)
        data = fetcher.fetch_all_data()
        if data:
            fetcher.save_data(data)
            st.sidebar.success(t("fetch_success"))
        else:
            st.sidebar.error(t("fetch_error"))

if not username:
    st.info(t("enter_username_info"))
    st.stop()

try:
    # Load Data
    analyzer = TraditionalAnalyzer()
    data_loaded = analyzer.load_data()
    
    # Check if data corresponds to the current user
    if data_loaded:
        loaded_user = analyzer.profile_data.get('login', '').lower()
        if loaded_user != username.lower():
            # Pass variables to translation string
            st.warning(t("cached_data_warning", loaded_user=loaded_user, username=username))
            st.stop()
    
    if not data_loaded:
        st.warning(t("no_data_warning"))
        st.stop()

    stats = analyzer.get_basic_stats()

    # Overview Section
    st.header(t("overview_header"))
    c1, c2, c3 = st.columns(3)
    c1.metric(t("metric_total_repos"), stats.get("total_repos", 0))
    c2.metric(t("metric_total_stars"), stats.get("total_stars", 0))
    c3.metric(t("metric_commits_tracked"), stats.get("total_commits_tracked", 0))

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        t("tab_repos"), 
        t("tab_languages"), 
        t("tab_llm"), 
        t("tab_forecasting"), 
        t("tab_model_comparison")
    ])

    with tab1:
        st.subheader(t("subheader_clustering"))
        if analyzer.repos_df is not None and not analyzer.repos_df.empty:
            clustered_df = analyzer.perform_clustering()
            if clustered_df is not None and not clustered_df.empty:
                fig = px.scatter(clustered_df, x="stars", y="forks", color="cluster", hover_data=["name"], title=t("clusters_title"))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(clustered_df)
            else:
                st.info(t("clustering_info_not_enough"))
        else:
            st.info(t("clustering_info_no_data"))

    with tab2:
        st.subheader(t("subheader_language"))
        langs = stats.get("top_languages", {})
        if langs:
            fig = px.pie(values=list(langs.values()), names=list(langs.keys()), title=t("language_pie_title"))
            st.plotly_chart(fig)
        else:
            st.info(t("language_info_no_data"))

    with tab3:
        st.subheader(t("subheader_llm"))
        ollama_model = st.selectbox(t("select_model_label"), ["llama3.1", "mistral"])
        llm = OllamaAnalyzer(model_name=ollama_model)

        if st.button(t("analyze_sentiment_button")):
            if analyzer.commits_df is not None and not analyzer.commits_df.empty:
                sample_commit = analyzer.commits_df.iloc[0]['message']
                st.write(f"**Sample Commit:** {sample_commit}")
                with st.spinner(t("sentiment_spinner")):
                    sentiment = llm.analyze_sentiment(sample_commit)
                    st.write(f"**Sentiment:** {sentiment}")
            else:
                st.info(t("no_commits_info"))

        st.markdown(t("skill_extraction_header"))
        if analyzer.repos_df is not None and not analyzer.repos_df.empty:
            repo_names = analyzer.repos_df['name'].tolist()
            selected_repo = st.selectbox(t("select_repo_label"), repo_names)
            
            if st.button(t("extract_skills_button")):
                repo_data = analyzer.repos_df[analyzer.repos_df['name'] == selected_repo].iloc[0]
                readme_text = repo_data.get('readme_content', "")
                
                if readme_text:
                    with st.spinner(t("extracting_spinner", repo=selected_repo)):
                        skills = llm.extract_skills(readme_text)
                        st.success(t("skills_extracted_success"))
                        st.write(skills)
                else:
                    st.warning(t("no_readme_warning"))
        else:
            st.info(t("clustering_info_no_data"))

    with tab4:
        st.subheader(t("subheader_forecasting"))
        if st.button(t("generate_forecast_button")):
            with st.spinner(t("forecasting_spinner")):
                try:
                    forecast = analyzer.forecast_activity()
                    if forecast is not None:
                        fig = px.line(forecast, x='ds', y='yhat', title=t("forecast_title"))
                        # Add confidence intervals
                        fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), showlegend=False)
                        fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line=dict(width=0), showlegend=False)
                        st.plotly_chart(fig)
                    else:
                        st.warning(t("forecast_warning_not_enough"))
                except Exception as e:
                    st.error(t("forecast_error", error=str(e)))

    with tab5:
        st.subheader(t("subheader_comparison"))
        prompt = st.text_area(t("test_prompt_label"), "Summarize the coding style based on these commits...")
        if st.button(t("compare_button")):
            with st.spinner(t("comparison_spinner")):
                results = llm.compare_models(prompt)
                for model_name, metrics in results.items():
                    st.write(f"### {model_name}")
                    if "error" in metrics:
                       st.error(metrics["error"])
                    else:
                       st.write(f"**Time:** {metrics['time']:.2f}s")
                       st.write(f"**Response:** {metrics['response']}")
                       st.divider()

except Exception as e:
    st.error(t("error_occurred", error=str(e)))
    import traceback
    st.text(traceback.format_exc())

