from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm import ChatResolution
from app.ai.llm import get_llm
from app.models import Conversation, ConversationMessage, SearchQuery
from app.schemas import (
    ConversationDTO,
    ConversationDetailDTO,
    ConversationMessageDTO,
    ConversationMessageResponse,
    SearchParams,
)
from app.services import search_service
from app.services.search_intent import normalize_search_params


class ConversationNotFoundError(Exception):
    pass


@dataclass
class MessagePostResult:
    response: ConversationMessageResponse
    params_override: SearchParams | None
    force_refresh: bool


def _title_from_text(text: str) -> str:
    clean = " ".join(text.split())
    if not clean:
        return "Chat baru"
    return clean[:48] + ("..." if len(clean) > 48 else "")


def _conversation_dto(row: Conversation, last_query_id: int | None = None) -> ConversationDTO:
    return ConversationDTO(
        id=row.id,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
        last_query_id=last_query_id,
    )


def _message_dto(row: ConversationMessage) -> ConversationMessageDTO:
    return ConversationMessageDTO(
        id=row.id,
        conversation_id=row.conversation_id,
        role=row.role,  # type: ignore[arg-type]
        content=row.content,
        search_query_id=row.search_query_id,
        metadata=row.metadata_ or {},
        created_at=row.created_at,
    )


async def _last_query_id(session: AsyncSession, conversation_id: int) -> int | None:
    return await session.scalar(
        select(SearchQuery.id)
        .where(SearchQuery.conversation_id == conversation_id)
        .order_by(SearchQuery.created_at.desc(), SearchQuery.id.desc())
        .limit(1)
    )


async def _get_owned_conversation(
    session: AsyncSession, user_id: int, conversation_id: int
) -> Conversation:
    row = (
        await session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise ConversationNotFoundError(str(conversation_id))
    return row


async def list_conversations(session: AsyncSession, user_id: int) -> list[ConversationDTO]:
    rows = (
        await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id, Conversation.deleted_at.is_(None))
            .order_by(desc(Conversation.updated_at), desc(Conversation.id))
        )
    ).scalars().all()
    out: list[ConversationDTO] = []
    for row in rows:
        out.append(_conversation_dto(row, await _last_query_id(session, row.id)))
    return out


async def create_conversation(
    session: AsyncSession, user_id: int, title: str | None = None
) -> ConversationDTO:
    row = Conversation(user_id=user_id, title=(title or "Chat baru").strip() or "Chat baru")
    session.add(row)
    await session.flush()
    await session.refresh(row)
    return _conversation_dto(row)


async def get_conversation(
    session: AsyncSession, user_id: int, conversation_id: int
) -> ConversationDetailDTO:
    conversation = await _get_owned_conversation(session, user_id, conversation_id)
    messages = (
        await session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at, ConversationMessage.id)
        )
    ).scalars().all()
    return ConversationDetailDTO(
        conversation=_conversation_dto(
            conversation, await _last_query_id(session, conversation.id)
        ),
        messages=[_message_dto(row) for row in messages],
    )


async def update_conversation_title(
    session: AsyncSession, user_id: int, conversation_id: int, title: str
) -> ConversationDTO:
    conversation = await _get_owned_conversation(session, user_id, conversation_id)
    conversation.title = title.strip() or "Chat baru"
    conversation.updated_at = datetime.now(UTC)
    await session.flush()
    await session.refresh(conversation)
    return _conversation_dto(conversation, await _last_query_id(session, conversation.id))


async def soft_delete_conversation(session: AsyncSession, user_id: int, conversation_id: int) -> None:
    conversation = await _get_owned_conversation(session, user_id, conversation_id)
    now = datetime.now(UTC)
    conversation.deleted_at = now
    conversation.updated_at = now
    await session.flush()


async def post_message(
    session: AsyncSession,
    *,
    user_id: int,
    conversation_id: int,
    content: str,
    force_refresh: bool = False,
) -> MessagePostResult:
    conversation = await _get_owned_conversation(session, user_id, conversation_id)
    clean = content.strip()
    previous_params = await _previous_params(session, conversation_id)
    recent_messages = await _recent_messages(session, conversation_id)

    user_message = ConversationMessage(
        conversation_id=conversation_id,
        role="user",
        content=clean,
        metadata_={},
    )
    session.add(user_message)
    await session.flush()
    await session.refresh(user_message)

    resolution = await _resolve(clean, previous_params, recent_messages)
    params = normalize_search_params(resolution.params) if resolution.params is not None else None

    query_id: int | None = None
    if resolution.action in {"new_search", "refine_search"} and params is not None:
        query_id = await search_service.create_search_row(
            session,
            clean,
            user_id=user_id,
            conversation_id=conversation_id,
            parsed_params=params.model_dump(),
        )

    assistant_content = resolution.response_text or _assistant_fallback(resolution, params)
    assistant_message = ConversationMessage(
        conversation_id=conversation_id,
        role="assistant",
        content=assistant_content,
        search_query_id=query_id,
        metadata_={
            "action": resolution.action,
            "params": params.model_dump() if params else None,
        },
    )
    session.add(assistant_message)

    if conversation.title == "Chat baru":
        conversation.title = _title_from_text(clean)
    conversation.updated_at = datetime.now(UTC)

    await session.flush()
    await session.refresh(assistant_message)
    await session.refresh(conversation)

    return MessagePostResult(
        response=ConversationMessageResponse(
            conversation_id=conversation_id,
            user_message=_message_dto(user_message),
            assistant_message=_message_dto(assistant_message),
            action=resolution.action,
            query_id=query_id,
        ),
        params_override=params,
        force_refresh=force_refresh,
    )


async def _previous_params(session: AsyncSession, conversation_id: int) -> SearchParams | None:
    data = await session.scalar(
        select(SearchQuery.parsed_params)
        .where(SearchQuery.conversation_id == conversation_id, SearchQuery.parsed_params.is_not(None))
        .order_by(SearchQuery.created_at.desc(), SearchQuery.id.desc())
        .limit(1)
    )
    if not isinstance(data, dict):
        return None
    return SearchParams.model_validate(data)


async def _recent_messages(session: AsyncSession, conversation_id: int) -> list[dict[str, str]]:
    rows = (
        await session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(desc(ConversationMessage.created_at), desc(ConversationMessage.id))
            .limit(10)
        )
    ).scalars().all()
    return [{"role": row.role, "content": row.content} for row in reversed(rows)]


async def _resolve(
    content: str, previous_params: SearchParams | None, recent_messages: list[dict[str, str]]
) -> ChatResolution:
    return await get_llm().resolve_chat_message(content, previous_params, recent_messages)


def _assistant_fallback(resolution: ChatResolution, params: SearchParams | None) -> str:
    if params is None:
        return "Saya bisa bantu cari lowongan. Sebutkan role, lokasi, atau preferensi kerja yang kamu mau."
    role = params.role_keywords[0] if params.role_keywords else "pekerjaan"
    loc = ", ".join(params.location) if params.location else "Indonesia"
    if resolution.action == "refine_search":
        return f"Siap, saya cari ulang lowongan {role} di {loc}."
    return f"Siap, saya cari lowongan {role} di {loc}."
