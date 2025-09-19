import os
import json
from fastapi import HTTPException, status
from typing import Any, Dict, List, Optional
from app.models.similarity_model import EmailClassifierModel
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email

class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""
    
    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()
        self._emails_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'emails.json'
        )
    
    def add_topic(self, topic_name: str, description: str) -> Dict[str, any]:
        """Append a topic to the topics file."""
        return self.model.add_topic(topic_name, description)

    
    def classify_email(self, email: Email) -> Dict[str, Any]:
        """Classify an email into topics using generated features"""
        
        # Step 1: Generate features from email
        features = self.feature_factory.generate_all_features(email)
        
        # Step 2: Classify using features
        predicted_topic = self.model.predict(features)
        topic_scores = self.model.get_topic_scores(features)
        
        # Return comprehensive results
        return {
            "predicted_topic": predicted_topic,
            "topic_scores": topic_scores,
            "features": features,
            "available_topics": self.model.topics,
            "email": email
        }
    
    def store_email(self, email: Email, ground_truth_topic: Optional[str] = None) -> int:
        """Store an email with a potential ground truth topic (simple append/overwrite style)."""
        # Validate GT if provided (case-insensitive → canonical)
        gt = None
        if ground_truth_topic:
            mapping = {t.lower(): t for t in self.model.topics}
            key = ground_truth_topic.strip().lower()
            if key not in mapping:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Ground truth topic '{ground_truth_topic}' is not in available topics."
                )
            gt = mapping[key]
    
        # Compute features now (optional but useful)
        features = self.feature_factory.generate_all_features(email)
    
        # Build record WITHOUT id. _save_email will assign it.
        record = {
            "subject": email.subject,
            "body": email.body,
            "ground_truth_topic": gt,   # None if not provided
            "features": features,       # consider storing a subset if large
        }
    
        return self._save_email(record)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the inference pipeline"""
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions()
        }
    
    def _save_email(self, record: Dict[str, Any]) -> int:
        """Append an email record to emails.json."""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'data', 'emails.json')
        if not os.path.exists(self._emails_file):
            data = {"emails": []}
        else:
            with open(self._emails_file, "r", encoding="utf-8") as f:
                data = json.load(f)
    
        emails: List[Dict[str, Any]] = data.get("emails", [])
        new_id = (max((e.get("id", 0) for e in emails), default=0) + 1)
        record["id"] = new_id
        emails.append(record)
    
        with open(self._emails_file, "w", encoding="utf-8") as f:
            json.dump({"emails": emails}, f, indent=2, ensure_ascii=False)
    
        return new_id