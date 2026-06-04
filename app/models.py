from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CV(Base):
    __tablename__ = "cv"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class SearchQuery(Base):
    __tablename__ = "search_query"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    raw_query: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cv_id: Mapped[int | None] = mapped_column(
        ForeignKey("cv.id", ondelete="SET NULL"), nullable=True
    )
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    from_cache: Mapped[bool | None] = mapped_column(default=None, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_search_query_created", "created_at"),)


class JobListing(Base):
    __tablename__ = "job_listing"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    portal: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    company: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[str | None] = mapped_column(Text, nullable=True)
    work_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    seniority: Mapped[str | None] = mapped_column(Text, nullable=True)
    salary_min: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    salary_max: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    salary_currency: Mapped[str] = mapped_column(Text, default="IDR")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    requirements: Mapped[str | None] = mapped_column(Text, nullable=True)
    responsibilities: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    mandatory_requirements: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    nice_to_have_requirements: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    skills_tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    benefits: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    detail_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    apply_url: Mapped[str] = mapped_column(Text, nullable=False)
    posted_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("portal", "external_id", name="uq_job_listing_portal_extid"),
        CheckConstraint(
            "portal IN ('linkedin','jobstreet','glints','kalibrr')",
            name="ck_job_listing_portal",
        ),
        Index("idx_job_listing_scraped", "scraped_at"),
    )


class MatchResult(Base):
    __tablename__ = "match_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(
        ForeignKey("search_query.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[int] = mapped_column(
        ForeignKey("job_listing.id", ondelete="CASCADE"), nullable=False
    )
    cv_id: Mapped[int] = mapped_column(ForeignKey("cv.id", ondelete="CASCADE"), nullable=False)
    match_score: Mapped[int] = mapped_column(Integer, nullable=False)
    cosine_score: Mapped[float | None] = mapped_column(nullable=True)
    llm_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    matched_skills: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    missing_skills: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    summary_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_en: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("match_score BETWEEN 0 AND 100", name="ck_match_score_range"),
        Index("idx_match_query", "query_id", "match_score"),
    )


class CoverLetter(Base):
    __tablename__ = "cover_letter"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("job_listing.id", ondelete="CASCADE"), nullable=False
    )
    cv_id: Mapped[int] = mapped_column(ForeignKey("cv.id", ondelete="CASCADE"), nullable=False)
    content_id: Mapped[str] = mapped_column(Text, nullable=False)
    content_en: Mapped[str] = mapped_column(Text, nullable=False)
    word_count_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    word_count_en: Mapped[int | None] = mapped_column(Integer, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (UniqueConstraint("job_id", "cv_id", name="uq_cover_letter_job_cv"),)


class CacheEntry(Base):
    __tablename__ = "cache_entry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    params_hash: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    result_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("idx_cache_expires", "expires_at"),)
