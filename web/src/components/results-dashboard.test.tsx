import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ResultsDashboard } from "@/components/results-dashboard";
import type { AnalysisResponse } from "@/lib/types";

vi.mock("@/components/charts/probability-chart", () => ({
  ProbabilityChart: ({
    probabilities,
  }: {
    probabilities: { control: number; adhd: number } | null;
  }) => (
    <div>
      {probabilities ? (
        "Probability chart"
      ) : (
        <span>Probability output is not available.</span>
      )}
    </div>
  ),
}));
vi.mock("@/components/charts/epoch-probability-chart", () => ({
  EpochProbabilityChart: () => <div>Epoch chart</div>,
}));
vi.mock("@/components/charts/feature-attribution-chart", () => ({
  FeatureAttributionChart: () => <div>Feature chart</div>,
}));
vi.mock("@/components/charts/raw-attribution-chart", () => ({
  RawAttributionChart: ({ slot }: { slot: number }) => (
    <div>Raw attribution channel slot {slot}</div>
  ),
}));

function makeResponse(
  status: AnalysisResponse["status"] = "ok",
): AnalysisResponse {
  const hasEpoch = status === "ok";
  const attributions = hasEpoch
    ? [
        {
          feature_name: "test_feature_from_checkpoint",
          signed_contribution: 0.012,
          absolute_magnitude: 0.012,
        },
      ]
    : [];
  return {
    status,
    sampling_rate_hz: 256,
    prediction: {
      majority_vote_class: hasEpoch ? "ADHD" : null,
      majority_vote_label: hasEpoch ? 1 : null,
      predicted_epoch_counts: { Control: 0, ADHD: hasEpoch ? 1 : 0 },
      mean_model_output_probabilities: hasEpoch
        ? { control: 0.31, adhd: 0.69 }
        : null,
      probabilities_are_calibrated_confidence: false,
    },
    quality_control: {
      candidate_epoch_count: 1,
      accepted_epoch_count: hasEpoch ? 1 : 0,
      rejected_epoch_count: hasEpoch ? 0 : 1,
      rejected_epochs: hasEpoch
        ? []
        : [
            {
              epoch_index: 1,
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
    epoch_analyses: hasEpoch
      ? [
          {
            epoch_index: 1,
            start_sample: 256,
            stop_sample: 1280,
            predicted_class: "ADHD",
            model_output_probabilities: { control: 0.31, adhd: 0.69 },
            adhd_minus_control_logit: 0.4,
            completeness_residual: 0.00001,
            feature_attributions: attributions,
            raw_attributions: {
              channel_slot_1: [0.1],
              channel_slot_2: [-0.1],
            },
            raw_attribution_magnitudes: {
              channel_slot_1: [0.1],
              channel_slot_2: [0.1],
            },
          },
        ]
      : [],
    recording_explanation: {
      method: "joint_integrated_gradients",
      target: "ADHD logit minus Control logit",
      integration_steps: 128,
      baseline_description: "Zero-valued normalized inputs.",
      feature_attributions: attributions,
      mean_signed_raw_attributions: hasEpoch
        ? { channel_slot_1: [0.1], channel_slot_2: [-0.1] }
        : null,
      mean_absolute_raw_attributions: hasEpoch
        ? { channel_slot_1: [0.1], channel_slot_2: [0.1] }
        : null,
      epoch_completeness_residuals: hasEpoch ? [0.00001] : [],
    },
    research_use_disclaimer:
      "Research use only. This exploratory tool is not a clinical diagnostic system.",
    scientific_limitations: [
      "Participant identities are not verified.",
      "Subject-independent generalization has not been established.",
    ],
  };
}

describe("ResultsDashboard", () => {
  it("renders returned predictions, probabilities, and scientific limitations", () => {
    render(
      <ResultsDashboard
        result={makeResponse()}
        fileName="recording.csv"
        onReset={vi.fn()}
      />,
    );

    expect(screen.getByRole("heading", { name: "ADHD" })).toBeInTheDocument();
    expect(screen.getAllByText("69.0%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("31.0%").length).toBeGreaterThan(0);
    expect(
      screen.getByText("test_feature_from_checkpoint"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("row", {
        name: "1 ADHD 31.0% 69.0% 1.000e-5",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Epoch 1")).toBeInTheDocument();
    expect(
      screen.getByText("Participant identities are not verified."),
    ).toBeInTheDocument();
  });

  it("handles empty analyses and nullable recording outputs", () => {
    render(
      <ResultsDashboard
        result={makeResponse("no_valid_epochs")}
        fileName="empty-result.csv"
        onReset={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("heading", { name: "No valid epoch prediction" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/did not accept any epochs/i),
    ).toBeInTheDocument();
    expect(screen.getByText("Candidate epoch 1")).toBeInTheDocument();
    expect(
      screen.getByText("Probability output is not available."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/No accepted epoch residuals were returned/),
    ).toBeInTheDocument();
  });

  it("allows switching between output tabs", () => {
    render(
      <ResultsDashboard
        result={makeResponse()}
        fileName="recording.csv"
        onReset={vi.fn()}
      />,
    );

    const overviewTab = screen.getByRole("tab", { name: /overview & classification/i });
    const featureTab = screen.getByRole("tab", { name: /feature attributions/i });
    const rawTab = screen.getByRole("tab", { name: /raw signal attributions/i });
    const qualityTab = screen.getByRole("tab", { name: /quality & epoch details/i });

    expect(overviewTab).toHaveAttribute("aria-selected", "true");
    expect(featureTab).toHaveAttribute("aria-selected", "false");

    // Click feature tab
    fireEvent.click(featureTab);
    expect(featureTab).toHaveAttribute("aria-selected", "true");
    expect(overviewTab).toHaveAttribute("aria-selected", "false");

    // Click raw tab
    fireEvent.click(rawTab);
    expect(rawTab).toHaveAttribute("aria-selected", "true");
    expect(featureTab).toHaveAttribute("aria-selected", "false");

    // Click quality tab
    fireEvent.click(qualityTab);
    expect(qualityTab).toHaveAttribute("aria-selected", "true");
    expect(rawTab).toHaveAttribute("aria-selected", "false");
  });
});
