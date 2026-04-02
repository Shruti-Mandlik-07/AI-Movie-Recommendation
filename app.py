import streamlit as st
import pandas as pd
import json
import ast
import requests
import heapq
import urllib.parse
from datetime import datetime

# Set page config
st.set_page_config(page_title="CineMatch AI", layout="wide", page_icon="🍿")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Attractive Pastel Combo - Soft Periwinkle to Silver */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e4efe9 0%, #93a5ce 100%);
        background-attachment: fixed;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent;
    }

    .stApp {
        color: #1e293b;
    }
    
    /* Beautiful Gradient Title */
    h1 {
        text-align: center;
        background: linear-gradient(to right, #ff416c, #ff4b2b, #f53844, #42378f);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 4.5rem !important;
        margin-bottom: 0px;
        animation: gradientShift 5s ease infinite;
        text-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    p.subtitle {
        text-align: center;
        color: #334155;
        font-size: 1.3rem;
        margin-bottom: 3.5rem;
        font-weight: 400;
        letter-spacing: 1px;
    }
    
    /* Input Overrides */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 65, 108, 0.3);
        border-radius: 12px;
        padding: 5px;
        font-size: 1.15rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #ff416c;
        box-shadow: 0 0 15px rgba(255, 65, 108, 0.3);
    }
    
    /* Striking Button with Click Animation */
    @keyframes clickPulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 75, 43, 0.7); }
        70% { box-shadow: 0 0 0 20px rgba(255, 75, 43, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 75, 43, 0); }
    }

    div.stButton {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 10px;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 15px rgba(255, 65, 108, 0.4);
        width: 100%;
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }
    div.stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 15px 25px rgba(255, 65, 108, 0.6);
        background: linear-gradient(135deg, #ff4b2b 0%, #ff416c 100%);
    }
    div.stButton > button:active {
        transform: translateY(2px) scale(0.96);
        animation: clickPulse 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    /* Snappy Movie Card Animations */
    @keyframes fadeInSlideUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .movie-poster-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        transition: all 0.25s ease;
        position: relative;
        background-color: #ffffff;
    }

    .movie-poster-container::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(to top, rgba(255,255,255,0.4), transparent);
        opacity: 0;
        transition: opacity 0.25s ease;
        pointer-events: none;
    }

    .movie-poster-container:hover {
        transform: scale(1.03) translateY(-4px);
        box-shadow: 0 15px 30px rgba(255, 65, 108, 0.3);
        border: 2px solid #ff416c;
    }
    
    .movie-poster-container:hover::after {
        opacity: 1;
    }

    .movie-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-genres {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 20px;
    }

    /* Hide some default Streamlit elements for cleaner UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1>🍿 CineMatch AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover your next favorite movie</p>', unsafe_allow_html=True)

@st.cache_data
def fetch_poster_url(movie_id, title):
    # Fetch from TMDB using a standard tutorial API key
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        pass
        
    clean_title = urllib.parse.quote(title)
    return f"https://ui-avatars.com/api/?name={clean_title}&background=334155&color=fff&size=500"


@st.cache_data
def load_data():
    df = pd.read_csv('data/movies.csv')
    movies_data = []

    for index, row in df.iterrows():
        if pd.isna(row['title']): continue
        
        # Get year from release_date safely
        year = ""
        rd = str(row.get('release_date', ''))
        if len(rd) >= 4 and rd[:4].isdigit():
            year = rd[:4]
            
        try: genres = [g['name'] for g in json.loads(row.get('genres', '[]'))]
        except:
            try: genres = [g['name'] for g in ast.literal_eval(row.get('genres', '[]'))]
            except: genres = []
                
        try: keywords = [k['name'] for k in json.loads(row.get('keywords', '[]'))]
        except:
            try: keywords = [k['name'] for k in ast.literal_eval(row.get('keywords', '[]'))]
            except: keywords = []
                
        movies_data.append({
            'id': row['id'],
            'title': str(row['title']).strip(),
            'genres': genres,
            'keywords': keywords,
            'year': year
        })
    
    graph = {}
    movies_dict = {m['title']: m for m in movies_data}
    
    for m in movies_dict.keys():
        graph[m] = []
        
    genre_to_movies = {}
    keyword_to_movies = {}

    for m in movies_data:
        for g in m['genres']: genre_to_movies.setdefault(g, []).append(m['title'])
        for k in m['keywords']: keyword_to_movies.setdefault(k, []).append(m['title'])

    for m in movies_data:
        title = m['title']
        connected_counts = {}
        
        for k in m['keywords']:
            for other in keyword_to_movies[k]:
                if other != title:
                    connected_counts[other] = connected_counts.get(other, 0) + 2
                    
        for g in m['genres']:
            for other in genre_to_movies[g]:
                if other != title:
                    connected_counts[other] = connected_counts.get(other, 0) + 1
        
        for other, weight in connected_counts.items():
            if weight >= 2:
                cost = max(1, 10 - weight)
                graph[title].append({'node': other, 'cost': cost})
                
    return movies_dict, list(movies_dict.keys()), graph

movies_dict, movie_names, graph = load_data()

# Algorithms
def bfs_recommendation(start_node, limit=12):
    if start_node not in graph: return []
    visited = set([start_node])
    queue = [start_node]
    recommendations = []
    while queue and len(recommendations) < limit:
        curr = queue.pop(0)
        for edge in graph[curr]:
            neighbor = edge['node']
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if neighbor != start_node:
                    recommendations.append(neighbor)
                if len(recommendations) >= limit: break
    return recommendations

def dfs_recommendation(start_node, limit=12):
    if start_node not in graph: return []
    visited = set()
    recommendations = []
    def dfs(node):
        if len(recommendations) >= limit: return
        visited.add(node)
        if node != start_node:
            recommendations.append(node)
        for edge in graph[node]:
            neighbor = edge['node']
            if neighbor not in visited: dfs(neighbor)
    dfs(start_node)
    return recommendations

def ucs_recommendation(start_node, limit=12):
    if start_node not in graph: return []
    frontier = []
    heapq.heappush(frontier, (0, start_node))
    came_from = {start_node: None}
    cost_so_far = {start_node: 0}
    recommendations = []
    visited = set()
    
    while frontier and len(recommendations) < limit:
        current_cost, current_node = heapq.heappop(frontier)
        if current_node in visited: continue
        visited.add(current_node)
        
        if current_node != start_node:
            recommendations.append(current_node)
            
        for edge in graph[current_node]:
            neighbor = edge['node']
            new_cost = cost_so_far[current_node] + edge['cost']
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
    return recommendations


# UI layout
# Moving the button right below the search bar for a professional centralized look
selected_movie = st.selectbox("Select a Movie you like:", movie_names[:2000])

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

find_button = st.button("Find Recommendations 🚀", use_container_width=True)

if find_button:
    selected_movie_data = movies_dict[selected_movie]
    sel_genres = ", ".join(selected_movie_data['genres'])

    with st.spinner(f"Finding the best recommendations..."):
        # Utilizing Uniform Cost Search (UCS) under the hood for most accurate results
        recs = ucs_recommendation(selected_movie)
        
    if not recs:
        st.warning("No connections found for this movie. Try another one!")
    else:
        st.markdown(f"## ✨ Recommendations for **{selected_movie}**")
        
        # Display engaging genres and messages
        if sel_genres:
            st.info(f"🎭 **Genres you're interested in:** {sel_genres}")
        
        st.success("🍿 **Grab your popcorn!** We hope you discover your next favorite movie from these hand-picked recommendations. Enjoy the show! 🎬")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display as grid 4x3
        cols = st.columns(4)
        for idx, rec in enumerate(recs):
            m_data = movies_dict[rec]
            poster_url = fetch_poster_url(m_data['id'], m_data['title'])
            
            # Fast, snappy staggered animation
            delay = (idx % 4) * 0.05 + (idx // 4) * 0.05
            
            with cols[idx % 4]:
                st.markdown(f'''
                    <div style="animation: fadeInSlideUp 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards; animation-delay: {delay}s; opacity: 0; transform: translateY(20px);">
                        <div class="movie-poster-container" style="width: 100%; aspect-ratio: 2/3;">
                            <img src="{poster_url}" style="width: 100%; height: 100%; object-fit: cover;" alt="{m_data['title']} Poster"/>
                        </div>
                        <div class="movie-title">{m_data['title']}</div>
                        <div class="movie-genres">{", ".join(m_data['genres'][:3])}</div>
                    </div>
                ''', unsafe_allow_html=True)
