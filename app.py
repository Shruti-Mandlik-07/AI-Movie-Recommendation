import pandas as pd
import json
import ast
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('data/movies.csv')
movies_data = []

# create a simplified dictionary mapping title -> {id, genres, keywords, title}
for index, row in df.iterrows():
    if pd.isna(row['title']):
        continue
    
    try:
        genres = [g['name'] for g in json.loads(row.get('genres', '[]'))]
    except:
        try:
            genres = [g['name'] for g in ast.literal_eval(row.get('genres', '[]'))]
        except:
            genres = []
            
    try:
        keywords = [k['name'] for k in json.loads(row.get('keywords', '[]'))]
    except:
        try:
            keywords = [k['name'] for k in ast.literal_eval(row.get('keywords', '[]'))]
        except:
            keywords = []
            
    movies_data.append({
        'id': row['id'],
        'title': str(row['title']).strip(),
        'genres': genres,
        'keywords': keywords,
        'popularity': row.get('popularity', 0)
    })

# Graph building
# We'll build a graph adjacency list
# To keep memory and time reasonable, we will only add edges between movies sharing >= 2 genres OR >= 1 keyword
graph = {}
movies_dict = {m['title']: m for m in movies_data}
movie_names = list(movies_dict.keys())

for m in movie_names:
    graph[m] = []

print("Building graph...")
# This O(n^2) might be slow. To optimize, let's index by genres and keywords
genre_to_movies = {}
keyword_to_movies = {}

for m in movies_data:
    for g in m['genres']:
        genre_to_movies.setdefault(g, []).append(m['title'])
    for k in m['keywords']:
        keyword_to_movies.setdefault(k, []).append(m['title'])

for m in movies_data:
    title = m['title']
    connected_counts = {}
    
    # Add weights based on keywords
    for k in m['keywords']:
        for other in keyword_to_movies[k]:
            if other != title:
                connected_counts[other] = connected_counts.get(other, 0) + 2
                
    # Add weights based on genres
    for g in m['genres']:
        for other in genre_to_movies[g]:
            if other != title:
                connected_counts[other] = connected_counts.get(other, 0) + 1
    
    # Create edges
    for other, weight in connected_counts.items():
        if weight >= 2: # threshold to limit graph density
            # cost is inversely proportional to similarity (weight)
            cost = max(1, 10 - weight)
            graph[title].append({'node': other, 'cost': cost})

print("Graph built!")

# Algorithms
def bfs_recommendation(start_node, limit=10):
    if start_node not in graph: return []
    visited = set([start_node])
    queue = [start_node]
    recommendations = []
    
    while queue and len(recommendations) < limit:
        curr = queue.pop(0)
        # Sort neighbors by popularity (implicit, or just take as is)
        for edge in graph[curr]:
            neighbor = edge['node']
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if neighbor != start_node:
                    recommendations.append(neighbor)
                if len(recommendations) >= limit: break
    return recommendations

def dfs_recommendation(start_node, limit=10):
    if start_node not in graph: return []
    visited = set()
    recommendations = []
    
    def dfs(node):
        if len(recommendations) >= limit: return
        visited.add(node)
        if node != start_node:
            recommendations.append(node)
        for edge in graph[node]: # to make it interesting, we don't sort
            neighbor = edge['node']
            if neighbor not in visited:
                dfs(neighbor)
                
    dfs(start_node)
    return recommendations

import heapq
def ucs_recommendation(start_node, limit=10):
    if start_node not in graph: return []
    
    frontier = []
    heapq.heappush(frontier, (0, start_node))
    
    came_from = {}
    cost_so_far = {}
    came_from[start_node] = None
    cost_so_far[start_node] = 0
    
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

@app.route('/')
def home():
    return render_template('index.html', movies=movie_names[:1000]) # Pass top 1000 for dropdown

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie = data.get('movie')
    algo = data.get('algo')
    
    if algo == 'BFS':
        recs = bfs_recommendation(movie, 12)
    elif algo == 'DFS':
        recs = dfs_recommendation(movie, 12)
    elif algo == 'UCS':
        recs = ucs_recommendation(movie, 12)
    else:
        recs = []
        
    # Get movie details for the recommendations
    rec_details = []
    for r in recs:
        rec_details.append(movies_dict[r])
        
    return jsonify(rec_details)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
