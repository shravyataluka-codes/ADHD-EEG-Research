import { afterEach, describe, expect, it, vi } from "vitest";
import { analyzeRecording } from "@/lib/api";

const response = {
  status: "no_valid_epochs",
  sampling_rate_hz: 256,
  prediction: {
    majority_vote_class: null,
    majority_vote_label: null,
    predicted_epoch_counts: { Control: 0, ADHD: 0 },
    mean_model_output_probabilities: null,
    probabilities_are_calibrated_confidence: false,
  },
  quality_control: {
    candidate_epoch_count: 1,
    accepted_epoch_count: 0,
    rejected_epoch_count: 1,
    rejected_epochs: [
      {
        epoch_index: 0,
        start_sample: 0,
        stop_sample: 1024,
        reason: "Synthetic test rejection.",
        channel_slots: [1],
      },
    ],
    edge_trim_samples_per_side: 256,
    epoch_samples: 1024,
    discarded_tail_samples: 0,
  },
  epoch_analyses: [],
  recording_explanation: {
    method: "joint_integrated_gradients",
    target: "ADHD logit minus Control logit",
    integration_steps: 128,
    baseline_description: "Zero-valued normalized inputs.",
    feature_attributions: [],
    mean_signed_raw_attributions: null,
    mean_absolute_raw_attributions: null,
    epoch_completeness_residuals: [],
  },
  research_use_disclaimer:
    "Research use only; not a clinical diagnostic system.",
  scientific_limitations: [],
};

describe("analyzeRecording", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("posts the required multipart fields without setting Content-Type", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "http://localhost:8000/");
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => response,
    });
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["0,1\n2,3"], "recording.csv", {
      type: "text/csv",
    });

    await expect(analyzeRecording(file, 256)).resolves.toEqual(response);

    const [url, options] = fetchMock.mock.calls[0] as [
      string,
      RequestInit,
    ];
    expect(url).toBe("http://localhost:8000/api/v1/analyze");
    expect(options.method).toBe("POST");
    expect(options.headers).toBeUndefined();
    expect(options.body).toBeInstanceOf(FormData);
    const formData = options.body as FormData;
    expect(formData.get("file")).toBe(file);
    expect(formData.get("sampling_rate")).toBe("256");
  });

  it("shows a safe server detail and suppresses stack traces", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockResolvedValueOnce({
          ok: false,
          status: 400,
          json: async () => ({ detail: "Sampling rate must be 256 Hz." }),
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ detail: "Traceback (most recent call last): ..." }),
        }),
    );
    const file = new File(["1,2"], "recording.csv", { type: "text/csv" });

    await expect(analyzeRecording(file, 128)).rejects.toThrow(
      "Sampling rate must be 256 Hz.",
    );
    await expect(analyzeRecording(file, 256)).rejects.toThrow(
      "Analysis could not be completed (HTTP 500).",
    );
  });

  it("returns a useful network error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    const file = new File(["1,2"], "recording.csv", { type: "text/csv" });

    await expect(analyzeRecording(file, 256)).rejects.toThrow(
      "Could not reach the analysis service.",
    );
  });
});
