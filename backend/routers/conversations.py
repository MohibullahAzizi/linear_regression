"""Conversation and chat router."""
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models, schemas
from services.copilot import generate_copilot_response

router = APIRouter()

@router.get("", response_model=list[schemas.ConversationOut])
def list_conversations(db: Session = Depends(get_db)):
    return db.query(models.Conversation).order_by(models.Conversation.updated_at.desc()).all()

@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conv = db.query(models.Conversation).filter_by(id=conversation_id).first()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    messages = db.query(models.ChatMessage).filter_by(conversation_id=conversation_id).order_by(models.ChatMessage.created_at).all()
    return {
        "id": conv.id,
        "title": conv.title,
        "assessment_id": conv.assessment_id,
        "selected_agent": conv.selected_agent,
        "selected_model": conv.selected_model,
        "context": conv.context or {},
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "metadata": m.meta or {},
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }

@router.post("", response_model=schemas.ConversationOut)
def create_conversation(data: schemas.ConversationCreate, db: Session = Depends(get_db)):
    conv = models.Conversation(**data.model_dump())
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

@router.put("/{conversation_id}", response_model=schemas.ConversationOut)
def update_conversation(conversation_id: str, data: schemas.ConversationUpdate, db: Session = Depends(get_db)):
    conv = db.query(models.Conversation).filter_by(id=conversation_id).first()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(conv, k, v)
    db.commit()
    db.refresh(conv)
    return conv

@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conv = db.query(models.Conversation).filter_by(id=conversation_id).first()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    db.delete(conv)
    db.commit()
    return {"ok": True}

@router.post("/{conversation_id}/messages")
def send_message(conversation_id: str, content: str = "", db: Session = Depends(get_db)):
    conv = db.query(models.Conversation).filter_by(id=conversation_id).first()
    if not conv:
        raise HTTPException(404, "Conversation not found")

    # Save user message
    user_msg = models.ChatMessage(conversation_id=conversation_id, role="user", content=content)
    db.add(user_msg)

    # Get conversation history
    messages = db.query(models.ChatMessage).filter_by(conversation_id=conversation_id).order_by(models.ChatMessage.created_at).all()

    # Get assessment context if linked
    assessment = None
    if conv.assessment_id:
        assessment = db.query(models.Assessment).filter_by(id=conv.assessment_id).first()

    # Generate copilot response
    response_text, metadata = generate_copilot_response(
        content, conv, messages, assessment, db
    )

    # Save assistant message
    assistant_msg = models.ChatMessage(
        conversation_id=conversation_id,
        role="assistant",
        content=response_text,
        meta=metadata,
    )
    db.add(assistant_msg)

    # Update conversation title if it's the first message
    if len(messages) <= 1 and content:
        conv.title = content[:50] + ("..." if len(content) > 50 else "")

    db.commit()
    return {
        "user_message": {"role": "user", "content": content},
        "assistant_message": {"role": "assistant", "content": response_text, "metadata": metadata},
    }
