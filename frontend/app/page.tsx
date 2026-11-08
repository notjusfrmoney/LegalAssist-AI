'use client';

import { FormEvent, useMemo, useRef, useState } from "react";
import { ChatCitation, ChatMessage, getChatErrorMessage, sendChatMessage } from "@/lib/api";

type ChatEntry = ChatMessage & {
  citations?: ChatCitation[];
  followUpQuestion?: string | null;
};

const starterPrompts = ["Unpaid salary", "Late salary", "Deductions", "Overtime"];

const initialAssistantMessage: ChatEntry = {
  role: "assistant",
  content:
    "Hi — I can help with unpaid wages, delayed salary, deductions and overtime questions within the supported scope. Tell me what happened and which State the workplace is in when that matters.",
};

export default function HomePage() {
  const [messages, setMessages] = useState<ChatEntry[]>([initialAssistantMessage]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

  const hasConversation = useMemo(() => messages.length > 1, [messages]);

  async function submitMessage(nextInput: string) {
    const trimmed = nextInput.trim();
    if (!trimmed || isLoading) return;

    const userMessage: ChatEntry = { role: "user", content: trimmed };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    setError(null);
    setIsLoading(true);

    try {
      const response = await sendChatMessage(nextMessages);
      const assistantMessage: ChatEntry = {
        role: "assistant",
        content: response.answer,
        citations: response.citations,
        followUpQuestion: response.follow_up_question,
      };

      setMessages((current) => [...current, assistantMessage]);
    } catch (err) {
      setError(getChatErrorMessage(err));
    } finally {
      setIsLoading(false);
      composerRef.current?.focus();
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    submitMessage(input);
  }

  function handleStarterPrompt(prompt: string) {
    submitMessage(prompt);
  }

  function handleNewChat() {
    setMessages([initialAssistantMessage]);
    setInput("");
    setError(null);
    setIsLoading(false);
  }

  return (
    <main className="page-shell">
      <div className="phone-frame">
        <header className="topbar">
          <div className="brand-block">
            <div className="brand-mark">L</div>
            <div>
              <p className="brand-name">LegalAssist AI</p>
              <p className="brand-tag">Wage rights help</p>
            </div>
          </div>

          <button type="button" className="ghost-button" onClick={handleNewChat}>
            New chat
          </button>
        </header>

        <section className="chat-shell">
          {!hasConversation && (
            <div className="welcome-panel">
              <div className="welcome-copy">
                <p className="eyebrow">General information only</p>
                <h1>Ask about wage rights in plain language.</h1>
                <p>Get general information on unpaid wages, delayed salary, deductions, minimum wage and overtime.</p>
              </div>

              <div className="prompt-grid">
                {starterPrompts.map((prompt) => (
                  <button key={prompt} type="button" className="starter-prompt" onClick={() => handleStarterPrompt(prompt)}>
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="messages" aria-live="polite">
            {messages.map((message, index) => {
              const isAssistant = message.role === "assistant";
              const lines = message.content.split("\n");
              const citations = message.citations ?? [];
              const followUpQuestion = message.followUpQuestion
                ?.replace(/^\s*Follow[- ]up question:\s*/i, "")
                .replace(/\*/g, "")
                .trim();

              return (
                <div key={`${message.role}-${index}`} className={`message-row ${isAssistant ? "assistant-row" : "user-row"}`}>
                  <div className={`message-bubble ${isAssistant ? "assistant-bubble" : "user-bubble"}`}>
                    {lines.map((line, lineIndex) => {
                      const isEmpty = line.trim().length === 0;

                      if (isEmpty) return <div key={`${lineIndex}-empty`} className="message-line-spacer" />;

                      return <p key={`${lineIndex}-${line}`}>{line}</p>;
                    })}

                    {isAssistant && followUpQuestion && (
                      <p className="follow-up-question">{followUpQuestion}</p>
                    )}

                    {isAssistant && (
                      <details className="source-disclosure">
                        <summary>
                          <span>Sources ({citations.length})</span>
                          <span className="source-chevron" aria-hidden="true">▼</span>
                        </summary>
                        {citations.length > 0 ? (
                          <div className="citation-list">
                            {citations.map((citation, citationIndex) => {
                              const sectionLabel =
                                citation.module_id ??
                                citation.section ??
                                citation.section_number;
                              const pageLabel = citation.page_start
                                ? citation.page_end && citation.page_end !== citation.page_start
                                  ? `${citation.page_start}-${citation.page_end}`
                                  : citation.page_start
                                : null;
                              const verificationStatus = citation.verification_status ?? "Not provided";
                              const authorityLabel =
                                citation.source_type === "practical_guidance"
                                  ? "Practical guidance · non-authoritative"
                                  : citation.source_type === "official_faq"
                                    ? "Official explanatory guidance"
                                    : citation.source_type === "delegated_rules"
                                      ? "Delegated rules"
                                      : citation.source_type === "primary_legislation"
                                        ? "Primary legislation"
                                        : citation.authority_level ?? citation.source_type;
                              const isPendingReview =
                                verificationStatus === "needs_review" ||
                                verificationStatus === "pending_human_review";

                              return (
                                <article
                                  key={`${citation.chunk_id ?? citation.source_id ?? "source"}-${citationIndex}`}
                                  className="citation-card"
                                >
                                  <div className="citation-header">
                                    <span className="citation-title">{citation.document_title ?? "Source"}</span>
                                    {sectionLabel && <span className="citation-section">{sectionLabel}</span>}
                                  </div>
                                  <div className="citation-meta">
                                    {pageLabel ? (
                                      <span>PDF page {pageLabel}</span>
                                    ) : citation.section_heading ? (
                                      <span>Heading: {citation.section_heading}</span>
                                    ) : citation.location ? (
                                      <span>{citation.location}</span>
                                    ) : null}
                                    {authorityLabel && <span>{authorityLabel}</span>}
                                    <span>Verification: {verificationStatus}</span>
                                  </div>
                                  {citation.module_heading && (
                                    <p className="citation-reference">{citation.module_heading}</p>
                                  )}
                                  {citation.source_url && (
                                    <a
                                      className="citation-link"
                                      href={citation.source_url}
                                      target="_blank"
                                      rel="noreferrer"
                                    >
                                      {citation.source_url}
                                    </a>
                                  )}
                                  {!citation.source_url && citation.source_filename && (
                                    <p className="citation-reference">Local source: {citation.source_filename}</p>
                                  )}
                                  {citation.chunk_id && (
                                    <p className="citation-reference">Reference: {citation.chunk_id}</p>
                                  )}
                                  {citation.source_filename && (
                                    <p className="citation-reference">Source file: {citation.source_filename}</p>
                                  )}
                                  {citation.claim_ids && citation.claim_ids.length > 0 && (
                                    <p className="citation-reference">
                                      Claim references: {citation.claim_ids.join(", ")}
                                    </p>
                                  )}
                                  {citation.unresolved_refs && citation.unresolved_refs.length > 0 && (
                                    <p className="source-review-warning">
                                      Unresolved issues (not settled law): {citation.unresolved_refs.join(", ")}
                                    </p>
                                  )}
                                  {isPendingReview && (
                                    <p className="source-review-warning">
                                      {verificationStatus === "pending_human_review"
                                        ? "This practical guide is awaiting human review; it is not an authoritative legal source."
                                        : "This source has not been independently verified."}
                                    </p>
                                  )}
                                </article>
                              );
                            })}
                          </div>
                        ) : (
                          <p className="no-sources">No relevant sources were retrieved for this response.</p>
                        )}
                      </details>
                    )}
                  </div>
                </div>
              );
            })}

            {isLoading && (
              <div className="message-row assistant-row">
                <div className="message-bubble assistant-bubble loading-bubble">Thinking…</div>
              </div>
            )}
            {error && <div className="error-banner" role="alert">{error}</div>}
          </div>
        </section>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            ref={composerRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            rows={1}
            placeholder="Describe the wage issue…"
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submitMessage(input);
              }
            }}
          />
          <button type="submit" disabled={isLoading || !input.trim()} className="send-button">
            {isLoading ? "Sending" : "Send"}
          </button>
        </form>

        <footer className="disclaimer">General legal information only; not a substitute for legal advice.</footer>
      </div>
    </main>
  );
}
