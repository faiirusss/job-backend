import asyncio

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.events import bus
from app.models import UserAccount
from app.schemas import (
    ConversationCreateRequest,
    ConversationDTO,
    ConversationDetailDTO,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ConversationUpdateRequest,
)
from app.services import conversation_service, search_service

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"error": {"code": "NOT_FOUND", "message": "conversation not found"}},
    )


@router.get("", response_model=list[ConversationDTO])
async def list_conversations(
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> list[ConversationDTO]:
    return await conversation_service.list_conversations(session, current_user.id)


@router.post("", response_model=ConversationDTO, status_code=201)
async def create_conversation(
    req: ConversationCreateRequest,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> ConversationDTO:
    return await conversation_service.create_conversation(session, current_user.id, req.title)


@router.get("/{conversation_id}", response_model=ConversationDetailDTO)
async def get_conversation(
    conversation_id: int,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> ConversationDetailDTO:
    try:
        return await conversation_service.get_conversation(session, current_user.id, conversation_id)
    except conversation_service.ConversationNotFoundError as e:
        raise _not_found(e) from e


@router.patch("/{conversation_id}", response_model=ConversationDTO)
async def update_conversation(
    conversation_id: int,
    req: ConversationUpdateRequest,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> ConversationDTO:
    try:
        return await conversation_service.update_conversation_title(
            session, current_user.id, conversation_id, req.title
        )
    except conversation_service.ConversationNotFoundError as e:
        raise _not_found(e) from e


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: int,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> Response:
    try:
        await conversation_service.soft_delete_conversation(session, current_user.id, conversation_id)
    except conversation_service.ConversationNotFoundError as e:
        raise _not_found(e) from e
    return Response(status_code=204)


@router.post("/{conversation_id}/messages", response_model=ConversationMessageResponse)
async def post_message(
    conversation_id: int,
    req: ConversationMessageRequest,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> ConversationMessageResponse:
    if not req.content.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "EMPTY_MESSAGE", "message": "Message cannot be empty"}},
        )
    try:
        result = await conversation_service.post_message(
            session,
            user_id=current_user.id,
            conversation_id=conversation_id,
            content=req.content,
            force_refresh=req.force_refresh,
        )
    except conversation_service.ConversationNotFoundError as e:
        raise _not_found(e) from e

    await session.commit()

    if result.response.query_id is not None:
        bus.open(result.response.query_id)
        asyncio.create_task(
            search_service.run_pipeline(
                result.response.query_id,
                req.content,
                result.force_refresh,
                params_override=result.params_override,
            )
        )

    return result.response
