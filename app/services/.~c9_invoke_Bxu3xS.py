from typing import Dict, Any
from app.models.similarity_model import EmailClassifierModel
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email

class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""
    
    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()
    
    def add_topic(self, topic_name: str, description: str) -> Dict[str, any]:
        """Append a topic to the topics file."""
        existing_lower = {t.lower() for t in self.model.topics}
        if topic_name.lower() in existing_lower:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Topic '{topic_name}' already exists."
            )
        try:
            return self.model.add_topic(topic_name, description)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
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
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the inference pipeline"""
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions()
        }

















