from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

Portal = Literal["linkedin", "jobstreet", "glints", "kalibrr"]
WorkType = Literal["remote", "hybrid", "onsite"]
Seniority = Literal["junior", "mid", "senior"]
DetailStatus = Literal["detail_ready", "listing_only"]
FitActionKind = Literal["load_detail_and_score", "score", "view_analysis"]


class JobListingDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    portal: Portal
    title: str
    company: str
    company_logo_bg: str
    # Logo URL available from the listing itself (no detail enrichment needed);
    # the card falls back to ``company_logo_bg`` when this is absent.
    company_logo_url: str | None = None
    location: str
    work_type: WorkType
    seniority: Seniority
    salary_min: int
    salary_max: int
    posted_date: str
    posted_label: str
    apply_url: str
    match_score: int | None
    cosine: float
    llm_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    summary_id: str
    summary_en: str
    description: str
    requirements: str
    responsibilities: list[str] = Field(default_factory=list)
    mandatory_requirements: list[str] = Field(default_factory=list)
    nice_to_have_requirements: list[str] = Field(default_factory=list)
    skills_tags: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    detail: "NormalizedJob | None" = None

    @computed_field
    @property
    def detail_status(self) -> DetailStatus:
        if self.portal == "linkedin" and not (self.description and self.detail):
            return "listing_only"
        return "detail_ready"

    @computed_field
    @property
    def fit_action_kind(self) -> FitActionKind:
        if self.match_score is not None:
            return "view_analysis"
        if self.detail_status == "listing_only":
            return "load_detail_and_score"
        return "score"

    @computed_field
    @property
    def fit_action_label(self) -> str:
        if self.fit_action_kind == "view_analysis":
            return "Lihat Analisis"
        if self.fit_action_kind == "load_detail_and_score":
            return "Muat Detail & Cek Fit"
        return "Cek Kecocokan"

    @computed_field
    @property
    def fit_action_loading_label(self) -> str:
        if self.fit_action_kind == "load_detail_and_score":
            return "Mengambil detail..."
        return "Menghitung fit..."

    @computed_field
    @property
    def fit_action_hint(self) -> str:
        if self.fit_action_kind == "view_analysis":
            return "Lihat ringkasan kecocokan lowongan ini dengan CV Anda."
        if self.fit_action_kind == "load_detail_and_score":
            return (
                "Mengambil detail lowongan jika belum lengkap, lalu menghitung "
                "kecocokan dengan CV Anda."
            )
        return "Menghitung kecocokan lowongan ini dengan CV Anda."


class NJSocial(BaseModel):
    platform: str = ""
    url: str = ""


class NJSkill(BaseModel):
    name: str = ""
    must_have: bool = False


class NJBenefit(BaseModel):
    title: str = ""
    description: str = ""
    icon_key: str = ""


class NJSalary(BaseModel):
    show: bool = False
    min: int | None = None
    max: int | None = None
    currency: str | None = None
    mode: str | None = None
    label: str = "Gaji tidak ditampilkan"


class NJLocation(BaseModel):
    name: str = ""
    city: str | None = None
    province: str | None = None
    country: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class NJCategory(BaseModel):
    name: str = ""
    breadcrumb: list[str] = Field(default_factory=list)


class NJCompany(BaseModel):
    name: str = ""
    tagline: str | None = None
    logo_url: str | None = None
    banner_url: str | None = None
    website: str | None = None
    industry: str | None = None
    size_label: str | None = None
    address: str | None = None
    description_html: str | None = None
    is_verified: bool = False
    social_media: list[NJSocial] = Field(default_factory=list)
    gallery_urls: list[str] = Field(default_factory=list)


class NormalizedJob(BaseModel):
    id: str = ""
    title: str = ""
    canonical_url: str = ""
    apply_url: str = ""
    status: str = ""
    job_type: str = ""
    job_type_label: str = ""
    work_arrangement: str = ""
    work_arrangement_label: str = ""
    is_remote: bool = False
    category: NJCategory = Field(default_factory=NJCategory)
    min_years_experience: int = 0
    max_years_experience: int = 0
    experience_label: str = ""
    education_level: str = ""
    education_level_label: str = ""
    description_html: str = ""
    requirements_html: str | None = None
    skills: list[NJSkill] = Field(default_factory=list)
    benefits: list[NJBenefit] = Field(default_factory=list)
    salary: NJSalary = Field(default_factory=NJSalary)
    location: NJLocation = Field(default_factory=NJLocation)
    posted_at: str | None = None
    updated_at: str | None = None
    expiry_date: str | None = None
    is_cover_letter_mandatory: bool = False
    company: NJCompany = Field(default_factory=NJCompany)
    # LinkedIn guest detail exposes a "N applicants" caption; absent elsewhere.
    applicants_count: int | None = None


JobListingDTO.model_rebuild()


class CVData(BaseModel):
    filename: str
    size_kb: int
    updated_at: str
    text_length: int
    text_preview: str


class SearchRecord(BaseModel):
    id: str
    query: str
    date: str
    count: int
    duration_ms: int
    from_cache: bool
    conversation_id: str | None = None
    status: str = "completed"


class SearchParams(BaseModel):
    role_keywords: list[str] = Field(default_factory=list)
    location: list[str] = Field(default_factory=list)
    work_type: list[WorkType] = Field(default_factory=list)
    seniority: list[Seniority] | None = None
    salary_min_idr: int | None = None


class CoverLetterResponse(BaseModel):
    content_id: str
    content_en: str
    word_count_id: int
    word_count_en: int
    from_cache: bool
    generated_at: datetime


# --- WebSocket event envelopes (match frontend mock RunEvent) ---


class StatusEvent(BaseModel):
    type: Literal["status"] = "status"
    message: str


class IntroEvent(BaseModel):
    type: Literal["intro"] = "intro"
    message: str


class ParamsEvent(BaseModel):
    type: Literal["params"] = "params"
    payload: dict


class PortalStartEvent(BaseModel):
    type: Literal["portal_start"] = "portal_start"
    portal: Portal


class ProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    portal: Portal
    scraped: int
    total: int


class PartialResultEvent(BaseModel):
    type: Literal["partial_result"] = "partial_result"
    job: JobListingDTO


class PortalCompleteEvent(BaseModel):
    type: Literal["portal_complete"] = "portal_complete"
    portal: Portal


class MatchEvent(BaseModel):
    type: Literal["match"] = "match"
    job_ids: list[str]


class CompleteEvent(BaseModel):
    type: Literal["complete"] = "complete"
    total: int
    durationMs: float


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    severity: Literal["warning", "error"]
    message: str
    portal: Portal | None = None


# --- REST DTOs ---


class SearchRequest(BaseModel):
    query: str
    force_refresh: bool = False


class SearchAccepted(BaseModel):
    query_id: str


class UserDTO(BaseModel):
    id: int
    email: str
    name: str | None = None


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    user: UserDTO


class ConversationDTO(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    last_query_id: str | None = None


class ConversationMessageDTO(BaseModel):
    id: int
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    search_query_id: str | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime


class ConversationDetailDTO(BaseModel):
    conversation: ConversationDTO
    messages: list[ConversationMessageDTO]


class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ConversationUpdateRequest(BaseModel):
    title: str


class ConversationMessageRequest(BaseModel):
    content: str
    force_refresh: bool = False


class ConversationMessageResponse(BaseModel):
    conversation_id: str
    user_message: ConversationMessageDTO
    assistant_message: ConversationMessageDTO
    action: Literal["new_search", "refine_search", "general_chat"]
    query_id: str | None = None


class SearchResultsResponse(BaseModel):
    query_id: str
    conversation_id: str | None = None
    jobs: list[JobListingDTO]
    status: str
    result_count: int
    duration_ms: int | None = None


class CoverLetterRequest(BaseModel):
    tone: str = "professional"


class MatchScoreRequest(BaseModel):
    force_refresh: bool = False
