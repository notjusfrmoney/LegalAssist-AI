export type ChatRole = "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
  followUpQuestion?: string | null;
};

export type ChatCitation = {
  chunk_id?: string;
  source_id?: string;
  source_filename?: string;
  document_title?: string;
  source_type?: string;
  authority_level?: string;
  section?: string;
  section_number?: string;
  page_start?: number;
  page_end?: number;
  module_id?: string;
  module_heading?: string;
  section_heading?: string;
  heading_path?: string[];
  location?: string;
  claim_ids?: string[];
  unresolved_refs?: string[];
  source_url?: string;
  verification_status?: string;
  excerpt?: string;
};

export type ChatResponse = {
  answer: string;
  citations: ChatCitation[];
  needs_clarification: boolean;
  follow_up_question: string | null;
  scope_note: string | null;
};

const DEFAULT_CHAT_ERROR = "The assistant could not respond right now. Please try again.";
const DEFAULT_NETWORK_ERROR = "Unable to connect to the assistant. Check your connection and try again.";

function getApiBaseUrl() {
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
}

function readableError(value: unknown): string | null {
  if (typeof value === "string") {
    const message = value.trim();
    return message && message !== "[object Object]" ? message : null;
  }

  if (Array.isArray(value)) {
    const messages = value
      .map((item) => {
        if (!item || typeof item !== "object") return null;
        const detail = item as { loc?: unknown; msg?: unknown };
        if (typeof detail.msg !== "string") return null;
        const location = Array.isArray(detail.loc)
          ? detail.loc.filter((part): part is string | number => typeof part === "string" || typeof part === "number").join(".")
          : "";
        return location ? `${location}: ${detail.msg}` : detail.msg;
      })
      .filter((message): message is string => message !== null);
    return messages.length ? messages.join("; ") : null;
  }

  if (value && typeof value === "object") {
    const detail = value as Record<string, unknown>;
    for (const key of ["detail", "message", "msg", "error"]) {
      const message = readableError(detail[key]);
      if (message) return message;
    }
  }

  return null;
}

export function getChatErrorMessage(error: unknown): string {
  return readableError(error instanceof Error ? error.message : error) ?? DEFAULT_CHAT_ERROR;
}

export function normalizeChatResponse(payload: any): ChatResponse {
  const rawCitations = Array.isArray(payload?.citations)
    ? payload.citations
    : Array.isArray(payload?.sources)
      ? payload.sources
      : [];
  const citations = rawCitations.map((citation: any) => ({
        chunk_id: citation?.chunk_id ?? citation?.chunkId ?? undefined,
        source_id: citation?.source_id ?? citation?.sourceId ?? undefined,
        source_filename: citation?.source_filename ?? citation?.sourceFilename ?? undefined,
        document_title: citation?.document_title ?? citation?.documentTitle ?? undefined,
        source_type: citation?.source_type ?? citation?.sourceType ?? undefined,
        authority_level: citation?.authority_level ?? citation?.authorityLevel ?? undefined,
        section: citation?.section ?? citation?.section_number ?? citation?.sectionNumber ?? undefined,
        section_number: citation?.section_number ?? citation?.sectionNumber ?? undefined,
        page_start: citation?.page_start ?? citation?.pageStart ?? undefined,
        page_end: citation?.page_end ?? citation?.pageEnd ?? undefined,
        module_id: citation?.module_id ?? citation?.moduleId ?? undefined,
        module_heading: citation?.module_heading ?? citation?.moduleHeading ?? undefined,
        section_heading: citation?.section_heading ?? citation?.sectionHeading ?? undefined,
        heading_path: citation?.heading_path ?? citation?.headingPath ?? undefined,
        location: citation?.location ?? undefined,
        claim_ids: citation?.claim_ids ?? citation?.claimIds ?? undefined,
        unresolved_refs: citation?.unresolved_refs ?? citation?.unresolvedRefs ?? undefined,
        source_url: citation?.source_url ?? citation?.sourceUrl ?? undefined,
        verification_status: citation?.verification_status ?? citation?.verificationStatus ?? undefined,
        excerpt: citation?.excerpt ?? undefined,
      }));

  const answer = typeof payload?.answer === "string" ? payload.answer : "";

  return {
    answer,
    citations,
    needs_clarification: Boolean(payload?.needs_clarification ?? payload?.clarification_needed),
    follow_up_question: payload?.follow_up_question ?? null,
    scope_note: payload?.scope_note ?? null,
  };
}

export async function sendChatMessage(messages: ChatMessage[]): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/v1/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: messages.map(({ role, content, ...extra }) => ({
          role,
          content:
            role === "assistant" && typeof extra.followUpQuestion === "string" && extra.followUpQuestion.trim()
              ? `${content}\n\nFollow-up question: ${extra.followUpQuestion.trim()}`
              : content,
        })),
      }),
    });
  } catch (error) {
    console.error("Chat API request failed", error);
    throw new Error(DEFAULT_NETWORK_ERROR);
  }

  if (!response.ok) {
    let detail: unknown;
    try {
      const data = await response.json();
      detail = data && typeof data === "object" && "detail" in data
        ? (data as { detail: unknown }).detail
        : data;
    } catch (error) {
      console.error("Chat API returned a non-JSON error response", error);
    }
    throw new Error(readableError(detail) ?? DEFAULT_CHAT_ERROR);
  }

  try {
    const payload = await response.json();
    return normalizeChatResponse(payload);
  } catch (error) {
    console.error("Chat API returned an invalid response", error);
    throw new Error(DEFAULT_CHAT_ERROR);
  }
}
