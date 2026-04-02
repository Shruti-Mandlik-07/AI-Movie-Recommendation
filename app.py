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
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0rem;
    }
    p.subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    /* Hide some default Streamlit elements for cleaner UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1>CineMatch AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover your next favorite movie</p>', unsafe_allow_html=True)

@st.cache_data
def fetch_poster_url(title, year):
    clean_title = urllib.parse.quote(title)
    
    # Try with year first as it is more accurate
    titles_to_try = []
    if year:
        titles_to_try.append(f"{clean_title}%20({year}%20film)")
    titles_to_try.append(f"{clean_title}%20(film)")
    titles_to_try.append(clean_title)

    for t in titles_to_try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={t}&prop=pageimages&format=json&pithumbsize=500"
        try:
            res = requests.get(url, timeout=5).json()
            pages = res.get('query', {}).get('pages', {})
            page_id = list(pages.keys())[0]
            if page_id != '-1' and 'thumbnail' in pages[page_id]:
                return pages[page_id]['thumbnail']['source']
        except:
            pass
            
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
col1, col2 = st.columns([3, 1])

with col1:
    selected_movie = st.selectbox("Select a Movie you like:", movie_names[:2000])

with col2:
    st.write("")
    st.write("")
    find_button = st.button("Find Recommendations 🚀", use_container_width=True)

if find_button:
    with st.spinner(f"Finding the best recommendations..."):
        # Utilizing Uniform Cost Search (UCS) under the hood for most accurate results
        recs = ucs_recommendation(selected_movie)
        
    if not recs:
        st.warning("No connections found for this movie. Try another one!")
    else:
        st.markdown("### Top Recommendations:")
        
        # Display as grid 4x3
        cols = st.columns(4)
        for idx, rec in enumerate(recs):
            m_data = movies_dict[rec]
            poster_url = fetch_poster_url(m_data['title'], m_data['year'])
            
            with cols[idx % 4]:
                st.markdown(f'''
                    <div style="width: 100%; aspect-ratio: 2/3; overflow: hidden; border-radius: 12px; margin-bottom: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
                        <img src="{poster_url}" style="width: 100%; height: 100%; object-fit: cover;" alt="{m_data['title']} Poster"/>
                    </div>
                    <div style="font-weight: 700; font-size: 1.05rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{m_data['title']}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 25px;">{", ".join(m_data['genres'][:3])}</div>
                ''', unsafe_allow_html=True)
