import type { AnalysisResponse } from "@/lib/types";

const LOCAL_API_BASE_URL = "http://localhost:8000";

function getApiBaseUrl(): string {
  return (
    process.env.NEXT_PUBLIC_API_BASE_URL?.trim().replace(/\/+$/, "") ||
    LOCAL_API_BASE_URL
  );
}

function errorMessageFromPayload(payload: unknown, status: number): string {
  if (
    payload &&
    typeof payload === "object" &&
    "detail" in payload &&
    typeof payload.detail === "string"
  ) {
    const detail = payload.detail.trim();
    const looksLikeStackTrace =
      /traceback|stack trace|file ["'].*line \d+|^\s*at\s.+:\d+/im.test(detail);
    if (detail && !looksLikeStackTrace) {
      return detail;
    }
  }

  if (status === 413) {
    return "The recording is too large to analyze. Choose a smaller CSV file.";
  }
  if (status === 429) {
    return "The analysis service is busy. Please wait a moment and try again.";
  }
  return `Analysis could not be completed (HTTP ${status}). Please try again.`;
}

export async function analyzeRecording(
  file: File,
  samplingRateHz: number,
  signal?: AbortSignal,
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("sampling_rate", String(samplingRateHz));

  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/v1/analyze`, {
      method: "POST",
      body: formData,
      signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new Error(
      "Could not reach the analysis service. Check that FastAPI is running and try again.",
    );
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(errorMessageFromPayload(payload, response.status));
  }

  if (!payload || typeof payload !== "object" || !("status" in payload)) {
    throw new Error("The analysis service returned an unexpected response.");
  }

  return payload as AnalysisResponse;
}
