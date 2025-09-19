from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    ground_truth_topic: Optional[str] = None  # optional

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    email_id: int
    
class TopicCreateRequest(BaseModel):
    name: str
    description: str

class TopicCreateResponse(BaseModel):
    message: str
    available_topics: List[str]

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email)
        
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emails", response_model=EmailAddResponse)
async def store_email(request: EmailStoreRequest):
    """Store and email with an optional ground truth topic"""
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        email_id = inference_service.store_email(
            email=email,
            ground_truth_topic=request.ground_truth_topic
        )
        return EmailAddResponse(email_id=email_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.post("/topics", response_model=TopicCreateResponse)
async def add_topics(request: TopicCreateRequest):
    """Dynamically add new topics"""
    try:
        inference_service = EmailTopicInferenceService()
        result = inference_service.add_topic(request.name, request.description)
        return TopicCreateResponse(
            available_topics=result["available_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

