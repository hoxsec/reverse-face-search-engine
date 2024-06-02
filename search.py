import numpy as np
from database import get_all_features

def search_similar_images(query_features, top_k=10):
    rows = get_all_features()
    
    distances = []
    
    for row in rows:
        img_path, features_blob = row
        features = np.frombuffer(features_blob, dtype=np.float32)
        dist = np.linalg.norm(query_features - features)
        distances.append((dist, img_path))
    
    # Sort by distance and take the top_k results
    distances.sort(key=lambda x: x[0])
    top_matches = distances[:top_k]
    
    return [match[1] for match in top_matches]
