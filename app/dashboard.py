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
    """Loads the JSON translation file for the specified language code, falling back to English for missing keys."""
    # Load English first as base
    base_path = os.path.join(os.path.dirname(__file__), '..', 'locales', 'en.json')
    translations = {}
    try:
        with open(base_path, 'r', encoding='utf-8') as f:
            translations = json.load(f)
    except Exception as e:
        print(f"Error loading base translations: {e}")

    if lang_code == 'en':
        return translations

    # Load selected language and update
    locale_path = os.path.join(os.path.dirname(__file__), '..', 'locales', f'{lang_code}.json')
    try:
        with open(locale_path, 'r', encoding='utf-8') as f:
            lang_data = json.load(f)
            translations.update(lang_data)
    except FileNotFoundError:
        pass # Just return English if file not found
    except Exception as e:
        print(f"Error loading {lang_code} translations: {e}")
        
    return translations

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

st.sidebar.caption("ℹ️ **Note:** Using a token ensures 100% accurate data and prevents missing repositories due to rate limits.")

with st.sidebar.expander("❓ How to get a GitHub Token?"):
    st.markdown("""
    1. Log in to [GitHub](https://github.com).
    2. Go to **Settings** > **Developer settings** > **Personal access tokens** > **Tokens (classic)**.
    3. Click **Generate new token (classic)**.
    4. Give it a name (e.g., "Dashboard").
    5. Select the **`repo`** and **`user`** scopes.
    6. Click **Generate token**.
    7. Copy the token and paste it here!
    """)

if st.sidebar.button(t("fetch_data_button")):
    if not token:
        st.sidebar.warning("⚠️ No token provided. Rate limit is 60 requests/hour. You may encounter errors.")
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    def update_progress(count, total, current_repo):
        progress = count / total
        progress_bar.progress(progress)
        status_text.text(f"Fetching {count}/{total}: {current_repo}")

    with st.spinner(t("fetching_spinner")):
        fetcher = GitHubFetcher(username=username, token=token)
        data = fetcher.fetch_all_data(progress_callback=update_progress)
        
        # Clear progress bar on completion
        progress_bar.empty()
        status_text.empty()

        if data:
            fetcher.save_data(data)
            st.sidebar.success(t("fetch_success"))
        else:
            st.sidebar.error(t("fetch_error") + " (Check terminal for details, likely rate limit)")

if not username:
    st.info(t("enter_username_info"))
    if not token:
        st.info("💡 **Tip:** Add a GitHub Token in the sidebar to increase your rate limit from 60 to 5000 requests/hour.")
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        t("tab_repos"), 
        t("tab_languages"), 
        t("tab_llm"), 
        t("tab_forecasting"), 
        t("tab_model_comparison"),
        "🎉 GitHub Replay"
    ])

    with tab1:
        st.subheader(t("subheader_clustering"))
        
        # --- Repo Health & Tech Stack ---
        if analyzer.repos_df is not None and not analyzer.repos_df.empty:
            st.markdown("### 🏆 Repository Health & Tech Stack")
            
            # Use columns for a grid layout of repo cards
            # We'll show top 10 most recent or starred to avoid clutter, or a list
            # Let's show a list of cards
            
            # Need to get the full objects to pass to helper functions
            # Since repos_df has 'files' now (we updated load_data), we can construct it
            repos = analyzer.repos_df.to_dict('records')
            
            for repo in repos:
                 # Reconstruct repo_data structure expected by functions
                 # Structure: {"details": {"files": [...]}}
                 repo_data_struct = {"details": {"files": repo.get("files", [])}}
                 
                 health = analyzer.calculate_health_score(repo_data_struct)
                 stack = analyzer.detect_tech_stack(repo_data_struct)
                 
                 # Grade Color
                 grade_color = "#2ea043" if health['grade'] == 'A' else "#e3b341" if health['grade'] == 'B' else "#da3633"
                 
                 with st.container():
                     c1, c2, c3 = st.columns([2, 1, 1])
                     with c1:
                         st.markdown(f"**{repo['name']}**")
                         st.caption(f"{repo.get('description', '')}")
                     with c2:
                         st.markdown(f"Health: <span style='color:{grade_color}; font-weight:bold; border:1px solid {grade_color}; padding:2px 6px; border-radius:4px;'>{health['grade']}</span>", unsafe_allow_html=True)
                         if health['missing']:
                             st.caption(f"Missing: {', '.join(health['missing'][:2])}")
                     with c3:
                         if stack:
                            st.write(" ".join([f"`{s}`" for s in stack]))
                         else:
                            st.caption("No stack detected")
                     st.divider()

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
            selected_repo = st.selectbox(t("select_repo_label"), repo_names, key="skill_repo")
            
            if st.button(t("extract_skills_button")):
                repo_data = analyzer.repos_df[analyzer.repos_df['name'] == selected_repo].iloc[0]
                readme_text = repo_data.get('readme_content', "")
                
                if readme_text:
                    with st.spinner(t("extracting_spinner", repo=selected_repo)):
                        skills = llm.extract_skills(readme_text)
                        if skills and skills.startswith("Error:"):
                            st.error(skills)
                        elif skills:
                            st.success(t("skills_extracted_success"))
                            st.markdown(f"### 🛠️ Detected Skills\n{skills}")
                        else:
                            st.warning(t("skills_extraction_failed"))

                else:
                    st.warning(t("no_readme_warning"))
            
            st.divider()
            st.subheader("🧠 AI README Improver")
            st.markdown("Select a repository above to analyze its README for improvements.")
            
            if st.button("🚀 Improve My README"):
                repo_data = analyzer.repos_df[analyzer.repos_df['name'] == selected_repo].iloc[0]
                readme_text = repo_data.get('readme_content', "")
                if readme_text:
                    with st.spinner(f"Analyzing README for {selected_repo}..."):
                        tips = llm.analyze_readme_quality(readme_text)
                        if tips.startswith("Error:"):
                             st.error(tips)
                        else:
                             st.markdown("### 📝 Improvement Checklist")
                             st.markdown(tips)
                else:
                    st.warning("No README found to improve.")

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

    with tab6:
        st.header("💻 GitHub Replay 2025")
        
        user_stats = analyzer.get_user_stats()
        
        # Generator Title logic
        if "user_title" not in st.session_state:
            with st.spinner("Generating your AI Developer Persona..."):
                llm_replay = OllamaAnalyzer(model_name=ollama_model if 'ollama_model' in locals() else "llama3.1")
                st.session_state["user_title"] = llm_replay.generate_user_title(user_stats)

        # --- Custom CSS for Cards ---
        st.markdown("""
        <style>
        .replay-card {
            background-color: #0d1117;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #58a6ff;
        }
        .metric-label {
            color: #8b949e;
            font-size: 1em;
        }
        .persona-title {
            font-size: 2.5em;
            background: -webkit-linear-gradient(45deg, #FF0080, #7928CA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

        # --- Persona Section ---
        st.markdown(f"""
        <div class="replay-card">
            <div class="metric-label">Your AI Developer Persona</div>
            <div class="persona-title">{st.session_state['user_title']}</div>
            <div style="margin-top: 10px; font-size: 1.2em;">{user_stats.get('chronotype', 'Day Walker')}</div>
        </div>
        """, unsafe_allow_html=True)

        # --- Metrics Grid ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="replay-card">
                <div class="metric-value">{user_stats.get('total_commits', 0)}</div>
                <div class="metric-label">Total Commits</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="replay-card">
                <div class="metric-value">{user_stats.get('longest_streak', 0)} Days</div>
                <div class="metric-label">Longest Streak</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="replay-card">
                <div class="metric-value">{user_stats.get('top_language', 'Unknown')}</div>
                <div class="metric-label">Top Language</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="replay-card">
                <div class="metric-value">{user_stats.get('most_active_month', 'Unknown')[:3]}</div>
                <div class="metric-label">Peak Month</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # --- Visualizations ---
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
             st.subheader("🕑 Daily Activity Pattern")
             if analyzer.commits_df is not None and not analyzer.commits_df.empty:
                 hourly_counts = analyzer.commits_df['date'].dt.hour.value_counts().sort_index()
                 st.bar_chart(hourly_counts)
             else:
                 st.info("No commit data available.")

        with col_viz2:
            st.subheader("💻 Top Languages")
            langs = user_stats.get("top_languages", {}) # Note: get_user_stats currently doesn't return full dict, need to fix or use analyzer.get_basic_stats
            # Actually get_basic_stats has it.
            basic_stats = analyzer.get_basic_stats()
            top_langs = basic_stats.get("top_languages", {})
            if top_langs:
                st.bar_chart(pd.Series(top_langs).head(5))
            else:
                st.info("No language data.")
        
        if st.button("Generate Replay"):
             st.balloons()

        st.divider()
        st.subheader("🚀 My GitHub Journey")
        
        timeline_events = analyzer.get_timeline_events()
        
        if timeline_events:
            # Custom HTML for Timeline
            timeline_html = """
            <style>
            .timeline {
                position: relative;
                max-width: 1200px;
                margin: 0 auto;
            }
            .timeline::after {
                content: '';
                position: absolute;
                width: 6px;
                background-color: #30363d;
                top: 0;
                bottom: 0;
                left: 31px;
                margin-left: -3px;
            }
            .container {
                padding: 10px 40px;
                position: relative;
                background-color: inherit;
                width: 100%;
            }
            .container::after {
                content: '';
                position: absolute;
                width: 25px;
                height: 25px;
                right: -17px;
                background-color: #58a6ff;
                border: 4px solid #0d1117;
                top: 15px;
                border-radius: 50%;
                z-index: 1;
                left: 18px;
            }
            .content {
                padding: 20px 30px;
                background-color: #161b22;
                position: relative;
                border-radius: 6px;
                border: 1px solid #30363d;
            }
            .date {
                font-size: 0.85em;
                color: #8b949e;
                margin-bottom: 5px;
            }
            .title {
                font-size: 1.1em;
                font-weight: bold;
                color: #c9d1d9;
            }
            </style>
            <div class="timeline">
            """
            
            for event in timeline_events:
                timeline_html += f"""
                <div class="container">
                    <div class="content">
                        <div class="date">{event['date']}</div>
                        <div class="title">{event['icon']} {event['title']}</div>
                    </div>
                </div>
                """
            
            timeline_html += "</div>"
            st.markdown(timeline_html, unsafe_allow_html=True)
        else:
            st.info("No timeline events found.")

except Exception as e:
    st.error(t("error_occurred", error=str(e)))
    import traceback
    st.text(traceback.format_exc())

