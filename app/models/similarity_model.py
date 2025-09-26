import os
import json
import math
from typing import Dict, Any, List

class EmailClassifierModel:
    """Simple rule-based email classifier model"""
    
    def __init__(self):
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())
    
    def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/topic_keywords.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        with open(data_file, 'r') as f:
            return json.load(f)
    
    def _save_topic_data(self, data: Dict[str, Any]) -> None:
        """Save topic data atomically to avoid corruption"""
        try:
            data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
            with open(data_file, 'r+') as file:
                file_data = json.load(file)
                file_data.update(data)
                file.seek(0)
                json.dump(file_data, file, indent=2)
                file.truncate()
        except FileNotFoundError:
            print(f"Error: The file '{data_file}' was not found.")
        
        except json.JSONDecodeError:
            print(f"Error: The file '{data_file}' is not a valid JSON file.")
    
    def predict(self, features: Dict[str, Any]) -> str:
        """Classify email into one of the topics using feature similarity"""
        scores = {}
        
        # Calculate similarity scores for each topic based on features
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = score
        
        return max(scores, key=scores.get)
    
    def add_topic(self, topic_name: str, description: str) -> Dict[str, Any]:
        """Add a new topic to the JSON file and refresh memory"""
        name = topic_name.lower()
        if name in self.topic_data:
            raise ValueError(f"Topic '{name}' already exists")

        # Add new topic
        self.topic_data[name] = {"description": description}
        self._save_topic_data(self.topic_data)

        # Refresh in-memory lists
        self.topics = list(self.topic_data.keys())

        return {"available_topics": self.topics}
    
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get classification scores for all topics"""
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = float(score)
        
        return scores
    
    def _calculate_topic_score(self, features: Dict[str, Any], topic: str) -> float:
        """Calculate similarity score based on length difference"""
        # Get email embedding from features
        email_embedding = features.get("email_embeddings_average_embedding", 0.0)
        
        # Get topic description and create embedding
        topic_description = self.topic_data[topic]['description']
        topic_embedding = float(len(topic_description))
        
        # Calculate similarity based on inverse distance
        # Smaller distance = higher similarity
        distance = abs(email_embedding - topic_embedding)
        
        # e^(-distance/scale) gives values between 0 and 1 - normalized range
        scale = 50.0
        similarity = math.exp(-distance / scale)
        
        return similarity
    
    def get_topic_description(self, topic: str) -> str:
        """Get description for a specific topic"""
        return self.topic_data[topic]['description']
    
    def get_all_topics_with_descriptions(self) -> Dict[str, str]:
        """Get all topics with their descriptions"""
        return {topic: self.get_topic_description(topic) for topic in self.topics}