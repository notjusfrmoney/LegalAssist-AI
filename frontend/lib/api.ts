export type ChatRole = "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type ChatCitation = {
  chunk_id?: string;
  source_id?: string;
  source_filename?: string;
  document_title?: string;
  section?: string;
  section_number?: string;
  page_start?: number;
  page_end?: number;
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

function getApiBaseUrl() {
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
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
        section: citation?.section ?? citation?.section_number ?? citation?.sectionNumber ?? undefined,
        section_number: citation?.section_number ?? citation?.sectionNumber ?? undefined,
        page_start: citation?.page_start ?? citation?.pageStart ?? undefined,
        page_end: citation?.page_end ?? citation?.pageEnd ?? undefined,
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
  const response = await fetch(`${getApiBaseUrl()}/api/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messages: messages.map(({ role, content }) => ({ role, content })),
    }),
  });

  if (!response.ok) {
    let detail = "The assistant could not respond right now.";
    try {
      const data = await response.json();
      detail = data?.detail ?? detail;
    } catch {
      // Ignore JSON parse failure and fall back to a friendly default message.
    }
    throw new Error(detail);
  }

  const payload = await response.json();
  return normalizeChatResponse(payload);
}
