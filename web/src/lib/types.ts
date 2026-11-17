export interface ModelProbabilities {
  control: number;
  adhd: number;
}

export interface RejectedEpoch {
  epoch_index: number;
  start_sample: number | null;
  stop_sample: number | null;
  reason: string;
  channel_slots: number[];
}

export interface QualityControl {
  candidate_epoch_count: number;
  accepted_epoch_count: number;
  rejected_epoch_count: number;
  rejected_epochs: RejectedEpoch[];
  edge_trim_samples_per_side: number;
  epoch_samples: number;
  discarded_tail_samples: number | null;
}

export interface RecordingPrediction {
  majority_vote_class: "Control" | "ADHD" | null;
  majority_vote_label: number | null;
  predicted_epoch_counts: Record<string, number>;
  mean_model_output_probabilities: ModelProbabilities | null;
  probabilities_are_calibrated_confidence: false;
}

export interface FeatureAttribution {
  feature_name: string;
  signed_contribution: number;
  absolute_magnitude: number;
}

export interface ChannelSlotAttributions {
  channel_slot_1: number[];
  channel_slot_2: number[];
}

export interface EpochAnalysis {
  epoch_index: number;
  start_sample: number;
  stop_sample: number;
  predicted_class: "Control" | "ADHD";
  model_output_probabilities: ModelProbabilities;
  adhd_minus_control_logit: number;
  completeness_residual: number;
  feature_attributions: FeatureAttribution[];
  raw_attributions: ChannelSlotAttributions;
  raw_attribution_magnitudes: ChannelSlotAttributions;
}

export interface RecordingExplanation {
  method: "joint_integrated_gradients";
  target: string;
  integration_steps: number;
  baseline_description: string;
  feature_attributions: FeatureAttribution[];
  mean_signed_raw_attributions: ChannelSlotAttributions | null;
  mean_absolute_raw_attributions: ChannelSlotAttributions | null;
  epoch_completeness_residuals: number[];
}

export interface AnalysisResponse {
  status: "ok" | "no_valid_epochs";
  sampling_rate_hz: number;
  prediction: RecordingPrediction;
  quality_control: QualityControl;
  epoch_analyses: EpochAnalysis[];
  recording_explanation: RecordingExplanation;
  research_use_disclaimer: string;
  scientific_limitations: string[];
}
