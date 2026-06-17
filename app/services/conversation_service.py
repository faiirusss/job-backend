from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm import ChatResolution, get_llm
from app.models import Conversation, ConversationMessage, SearchQuery
from app.schemas import (
    ConversationDetailDTO,
    ConversationDTO,
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
    internal_query_id: int | None = None


def _title_from_text(text: str) -> str:
    clean = " ".join(text.split())
    if not clean:
        return "Chat baru"
    return clean[:48] + ("..." if len(clean) > 48 else "")


def public_conversation_id(row: Conversation) -> str:
    return str(row.public_id)


def _conversation_dto(row: Conversation, last_query_id: str | None = None) -> ConversationDTO:
    return ConversationDTO(
        id=public_conversation_id(row),
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
        last_query_id=last_query_id,
    )


def _message_dto(
    row: ConversationMessage,
    conversation_public_id: str,
    search_query_public_id: str | None = None,
) -> ConversationMessageDTO:
    return ConversationMessageDTO(
        id=row.id,
        conversation_id=conversation_public_id,
        role=row.role,  # type: ignore[arg-type]
        content=row.content,
        search_query_id=search_query_public_id,
        metadata=row.metadata_ or {},
        created_at=row.created_at,
    )


async def _last_query_id(session: AsyncSession, conversation_id: int) -> str | None:
    public_id = await session.scalar(
        select(SearchQuery.public_id)
        .where(SearchQuery.conversation_id == conversation_id)
        .order_by(SearchQuery.created_at.desc(), SearchQuery.id.desc())
        .limit(1)
    )
    return str(public_id) if public_id is not None else None


async def _query_public_id_map(
    session: AsyncSession, messages: list[ConversationMessage]
) -> dict[int, str]:
    query_ids = {row.search_query_id for row in messages if row.search_query_id is not None}
    if not query_ids:
        return {}
    rows = await session.execute(
        select(SearchQuery.id, SearchQuery.public_id).where(SearchQuery.id.in_(query_ids))
    )
    return {row_id: str(public_id) for row_id, public_id in rows}


async def _get_owned_conversation(
    session: AsyncSession, user_id: int, conversation_ref: int | str
) -> Conversation:
    stmt = select(Conversation).where(
        Conversation.user_id == user_id,
        Conversation.deleted_at.is_(None),
    )
    if isinstance(conversation_ref, int):
        stmt = stmt.where(Conversation.id == conversation_ref)
    else:
        ref = conversation_ref.strip()
        if ref.isdecimal():
            stmt = stmt.where(Conversation.id == int(ref))
        else:
            try:
                public_id = uuid.UUID(ref)
            except ValueError as e:
                raise ConversationNotFoundError(ref) from e
            stmt = stmt.where(Conversation.public_id == public_id)
    row = (
        await session.execute(stmt)
    ).scalar_one_or_none()
    if row is None:
        raise ConversationNotFoundError(str(conversation_ref))
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
    session: AsyncSession, user_id: int, conversation_id: int | str
) -> ConversationDetailDTO:
    conversation = await _get_owned_conversation(session, user_id, conversation_id)
    internal_conversation_id = conversation.id
    conversation_public_id = public_conversation_id(conversation)
    messages = (
        await session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == internal_conversation_id)
            .order_by(ConversationMessage.created_at, ConversationMessage.id)
        )
    ).scalars().all()
    public_ids = await _query_public_id_map(session, list(messages))
    return ConversationDetailDTO(
        conversation=_conversation_dto(
            conversation, await _last_query_id(session, conversation.id)
        ),
        messages=[
            _message_dto(row, conversation_public_id, public_ids.get(row.search_query_id or 0))
            for row in messages
        ],
    )


async def update_conversation_title(
    session: AsyncSession, user_id: int, conversation_id: int | str, title: str
) -> ConversationDTO:
    conversation = await _get_owned_conversation(session, user_id, conversation_id)
    conversation.title = title.strip() or "Chat baru"
    conversation.updated_at = datetime.now(UTC)
    await session.flush()
    await session.refresh(conversation)
    return _conversation_dto(conversation, await _last_query_id(session, conversation.id))


async def soft_delete_conversation(
    session: AsyncSession, user_id: int, conversation_id: int | str
) -> None:
    conversation = await _get_owned_conversation(session, user_id, conversation_id)
    now = datetime.now(UTC)
    conversation.deleted_at = now
    conversation.updated_at = now
    await session.flush()


async def post_message(
    session: AsyncSession,
    *,
    user_id: int,
    conversation_id: int | str,
    content: str,
    force_refresh: bool = False,
) -> MessagePostResult:
    conversation = await _get_owned_conversation(session, user_id, conversation_id)
    internal_conversation_id = conversation.id
    conversation_public_id = public_conversation_id(conversation)
    clean = content.strip()
    previous_params = await _previous_params(session, internal_conversation_id)
    recent_messages = await _recent_messages(session, internal_conversation_id)

    user_message = ConversationMessage(
        conversation_id=internal_conversation_id,
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
    query_public_id: str | None = None
    if resolution.action in {"new_search", "refine_search"} and params is not None:
        query = await search_service.create_search_row(
            session,
            clean,
            user_id=user_id,
            conversation_id=internal_conversation_id,
            parsed_params=params.model_dump(),
        )
        query_id = query.id
        query_public_id = search_service.public_query_id(query)

    assistant_content = resolution.response_text or _assistant_fallback(resolution, params)
    assistant_message = ConversationMessage(
        conversation_id=internal_conversation_id,
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
            conversation_id=conversation_public_id,
            user_message=_message_dto(user_message, conversation_public_id),
            assistant_message=_message_dto(
                assistant_message, conversation_public_id, query_public_id
            ),
            action=resolution.action,
            query_id=query_public_id,
        ),
        params_override=params,
        force_refresh=force_refresh,
        internal_query_id=query_id,
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
