"use client";

import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle2,
  RefreshCw,
  XCircle,
} from "lucide-react";
import { EpochProbabilityChart } from "@/components/charts/epoch-probability-chart";
import { FeatureAttributionChart } from "@/components/charts/feature-attribution-chart";
import { ProbabilityChart } from "@/components/charts/probability-chart";
import { RawAttributionChart } from "@/components/charts/raw-attribution-chart";
import { ResearchNotice } from "@/components/research-notice";
import type { AnalysisResponse, FeatureAttribution } from "@/lib/types";

function formatPercent(value: number | null | undefined): string {
  return Number.isFinite(value) ? `${(Number(value) * 100).toFixed(1)}%` : "—";
}

function formatNumber(value: number | null | undefined): string {
  return Number.isFinite(value) ? Number(value).toExponential(3) : "—";
}

function FeatureContributionList({
  attributions,
}: {
  attributions: FeatureAttribution[];
}) {
  const top = [...attributions]
    .filter(
      (item) =>
        item.feature_name &&
        Number.isFinite(item.signed_contribution) &&
        Number.isFinite(item.absolute_magnitude),
    )
    .sort((left, right) => right.absolute_magnitude - left.absolute_magnitude)
    .slice(0, 10);

  if (top.length === 0) {
    return (
      <p className="rounded-xl bg-slate-50 p-4 text-sm text-slate-500">
        No feature attributions are available for this recording.
      </p>
    );
  }

  return (
    <ol className="divide-y divide-slate-100">
      {top.map((item, index) => {
        const isPositive = item.signed_contribution >= 0;
        const DirectionIcon = isPositive ? ArrowUpRight : ArrowDownRight;
        return (
          <li
            key={`${item.feature_name}-${index}`}
            className="flex items-center justify-between gap-3 py-3 first:pt-0 last:pb-0"
          >
            <div className="flex min-w-0 items-center gap-3">
              <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-slate-100 text-xs font-semibold text-slate-500">
                {index + 1}
              </span>
              <span className="truncate text-sm font-medium text-slate-700">
                {item.feature_name}
              </span>
            </div>
            <span
              className={`inline-flex shrink-0 items-center gap-1 font-mono text-xs font-semibold ${
                isPositive ? "text-indigo-700" : "text-rose-700"
              }`}
              title={
                isPositive
                  ? "Positive direction for the ADHD-minus-Control logit target"
                  : "Negative direction for the ADHD-minus-Control logit target"
              }
            >
              <DirectionIcon size={14} aria-hidden="true" />
              {isPositive ? "+" : ""}
              {formatNumber(item.signed_contribution)}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description?: string;
}) {
  return (
    <div>
      <p className="text-[10px] font-bold uppercase tracking-[0.17em] text-indigo-600">
        {eyebrow}
      </p>
      <h2 className="mt-1.5 text-lg font-bold tracking-tight text-slate-900">
        {title}
      </h2>
      {description && (
        <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-500">
          {description}
        </p>
      )}
    </div>
  );
}

export function ResultsDashboard({
  result,
  fileName,
  onReset,
}: {
  result: AnalysisResponse;
  fileName: string;
  onReset: () => void;
}) {
  const probabilities =
    result.prediction.mean_model_output_probabilities ?? null;
  const hasPrediction = result.prediction.majority_vote_class !== null;
  const topFeatures = result.recording_explanation.feature_attributions;

  return (
    <section
      className="mt-8 space-y-6"
      aria-labelledby="results-heading"
      aria-live="polite"
    >
      <div className="flex flex-col justify-between gap-4 rounded-3xl border border-slate-200 bg-white p-5 shadow-[0_12px_42px_-28px_rgba(27,39,76,0.28)] sm:flex-row sm:items-center sm:p-7">
        <div className="flex min-w-0 items-start gap-3.5">
          <span className="grid size-11 shrink-0 place-items-center rounded-2xl bg-indigo-50 text-indigo-700">
            <CheckCircle2 size={21} />
          </span>
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">
              Analysis response
            </p>
            <h2
              id="results-heading"
              className="mt-1 truncate text-xl font-bold tracking-tight text-slate-900"
            >
              {hasPrediction
                ? result.prediction.majority_vote_class
                : "No valid epoch prediction"}
            </h2>
            <p className="mt-1 truncate text-sm text-slate-500">
              {fileName} · {result.sampling_rate_hz} Hz ·{" "}
              {result.quality_control.accepted_epoch_count} accepted epochs
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="inline-flex min-h-10 shrink-0 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-indigo-200 hover:text-indigo-700"
        >
          <RefreshCw size={15} aria-hidden="true" />
          Analyze another recording
        </button>
      </div>

      {result.status === "no_valid_epochs" && (
        <div
          className="flex items-start gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950"
          role="status"
        >
          <Activity size={18} className="mt-0.5 shrink-0" />
          <p>
            The backend did not accept any epochs. No recording-level
            prediction or attribution summary is available; review the quality
            control information below.
          </p>
        </div>
      )}

      <div className="space-y-2">
        <ResearchNotice compact />
        <p className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs leading-5 text-slate-600">
          {result.research_use_disclaimer}
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <article className="min-w-0 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <SectionHeading
            eyebrow="Recording output"
            title="Model output probabilities"
            description="These are model output probabilities, not calibrated confidence estimates."
          />
          <div className="mt-5 min-w-0 w-full">
            <ProbabilityChart probabilities={probabilities} />
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-xl bg-slate-50 px-3.5 py-3">
              <p className="text-xs font-medium text-slate-500">Control</p>
              <p className="mt-1 text-lg font-bold text-slate-700">
                {formatPercent(probabilities?.control)}
              </p>
            </div>
            <div className="rounded-xl bg-indigo-50/70 px-3.5 py-3">
              <p className="text-xs font-medium text-indigo-700">ADHD</p>
              <p className="mt-1 text-lg font-bold text-indigo-900">
                {formatPercent(probabilities?.adhd)}
              </p>
            </div>
          </div>
        </article>

        <article className="min-w-0 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <SectionHeading
            eyebrow="Epoch consistency"
            title="Epoch-level model outputs"
            description="Each point shows the returned model output for one accepted epoch."
          />
          <div className="mt-4 min-w-0 w-full">
            <EpochProbabilityChart epochs={result.epoch_analyses} />
          </div>
        </article>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
        <article className="min-w-0 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <SectionHeading
            eyebrow="Handcrafted feature branch"
            title="Top feature attributions"
            description="Ranked by absolute attribution magnitude. Sign indicates direction for the ADHD-logit minus Control-logit target; it is not causal evidence."
          />
          <div className="mt-4 min-w-0 w-full">
            <FeatureAttributionChart attributions={topFeatures} />
          </div>
          <div className="mt-3 border-t border-slate-100 pt-4">
            <FeatureContributionList attributions={topFeatures} />
          </div>
        </article>

        <article className="min-w-0 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <SectionHeading
            eyebrow="Explanation method"
            title="Integrated Gradients"
            description="Attributions are exploratory and do not explain the filtering operation itself."
          />
          <dl className="mt-5 space-y-4">
            <div>
              <dt className="text-xs font-medium text-slate-500">Method</dt>
              <dd className="mt-1 text-sm font-semibold text-slate-800">
                {result.recording_explanation.method.replaceAll("_", " ")}
              </dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-slate-500">Target</dt>
              <dd className="mt-1 text-sm font-semibold text-slate-800">
                {result.recording_explanation.target}
              </dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-slate-500">
                Integration steps
              </dt>
              <dd className="mt-1 text-sm font-semibold text-slate-800">
                {result.recording_explanation.integration_steps}
              </dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-slate-500">
                Baseline
              </dt>
              <dd className="mt-1 text-sm leading-5 text-slate-700">
                {result.recording_explanation.baseline_description}
              </dd>
            </div>
          </dl>
          <div className="mt-5 rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-semibold text-slate-700">
              Completeness residual by accepted epoch
            </p>
            {result.recording_explanation.epoch_completeness_residuals.length >
            0 ? (
              <ul className="mt-2 max-h-40 space-y-1.5 overflow-auto">
                {result.recording_explanation.epoch_completeness_residuals.map(
                  (residual, index) => (
                    <li
                      key={`${index}-${residual}`}
                      className="flex justify-between gap-3 text-xs text-slate-600"
                    >
                      <span>
                        Epoch{" "}
                        {result.epoch_analyses[index]
                          ? result.epoch_analyses[index].epoch_index
                          : index + 1}
                      </span>
                      <span className="font-mono">
                        {formatNumber(residual)}
                      </span>
                    </li>
                  ),
                )}
              </ul>
            ) : (
              <p className="mt-2 text-xs text-slate-500">
                No accepted epoch residuals were returned.
              </p>
            )}
            <p className="mt-3 text-[11px] leading-5 text-slate-500">
              Residuals are numerical completeness checks, not prediction
              confidence or performance measures.
            </p>
          </div>
        </article>
      </div>

      <article className="min-w-0 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <SectionHeading
          eyebrow="Raw EEG branch"
          title="Recording-level raw attributions"
          description="Mean attribution over accepted epochs, indexed by time within an epoch. Channel positions are labeled as slots only."
        />
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          {([1, 2] as const).map((slot) => (
            <div
              key={slot}
              className="min-w-0 rounded-2xl border border-slate-100 bg-white p-4"
            >
              <h3 className="text-sm font-semibold text-slate-800">
                Channel slot {slot}
              </h3>
              <p className="mt-1 text-xs text-slate-500">
                Time in seconds · attribution in model logit-contribution units
              </p>
              <div className="mt-3 min-w-0 w-full">
                <RawAttributionChart
                  slot={slot}
                  signed={
                    result.recording_explanation.mean_signed_raw_attributions
                  }
                  absolute={
                    result.recording_explanation.mean_absolute_raw_attributions
                  }
                  samplingRateHz={result.sampling_rate_hz}
                />
              </div>
            </div>
          ))}
        </div>
      </article>

      <div className="grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
        <article className="min-w-0 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <SectionHeading
            eyebrow="Signal quality"
            title="Epoch quality control"
            description="Counts and rejection details returned by the analysis service."
          />
          <div className="mt-5 grid grid-cols-2 gap-3">
            <div className="rounded-xl bg-emerald-50 px-3.5 py-3">
              <p className="text-xs font-medium text-emerald-800">
                Accepted epochs
              </p>
              <p className="mt-1 flex items-center gap-1.5 text-xl font-bold text-emerald-950">
                <CheckCircle2 size={17} />
                {result.quality_control.accepted_epoch_count}
              </p>
            </div>
            <div className="rounded-xl bg-rose-50 px-3.5 py-3">
              <p className="text-xs font-medium text-rose-800">
                Rejected epochs
              </p>
              <p className="mt-1 flex items-center gap-1.5 text-xl font-bold text-rose-950">
                <XCircle size={17} />
                {result.quality_control.rejected_epoch_count}
              </p>
            </div>
          </div>
          <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-3 text-xs">
            <div>
              <dt className="text-slate-500">Candidate epochs</dt>
              <dd className="mt-1 font-semibold text-slate-800">
                {result.quality_control.candidate_epoch_count}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Epoch samples</dt>
              <dd className="mt-1 font-semibold text-slate-800">
                {result.quality_control.epoch_samples}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Edge trim per side</dt>
              <dd className="mt-1 font-semibold text-slate-800">
                {result.quality_control.edge_trim_samples_per_side} samples
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Discarded tail</dt>
              <dd className="mt-1 font-semibold text-slate-800">
                {result.quality_control.discarded_tail_samples ?? "—"} samples
              </dd>
            </div>
          </dl>
        </article>

        <article className="min-w-0 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <SectionHeading
            eyebrow="Epoch detail"
            title="Predictions and probabilities"
          />
          {result.epoch_analyses.length > 0 ? (
            <div className="mt-4 max-h-80 overflow-auto rounded-xl border border-slate-100">
              <table className="w-full min-w-[560px] border-collapse text-left text-xs">
                <caption className="sr-only">
                  Epoch predictions, Control and ADHD probabilities, and
                  completeness residuals
                </caption>
                <thead className="sticky top-0 bg-slate-50 text-[10px] uppercase tracking-wider text-slate-500">
                  <tr>
                    <th scope="col" className="px-3 py-3 font-semibold">
                      Epoch
                    </th>
                    <th scope="col" className="px-3 py-3 font-semibold">
                      Predicted class
                    </th>
                    <th scope="col" className="px-3 py-3 font-semibold">
                      Control
                    </th>
                    <th scope="col" className="px-3 py-3 font-semibold">
                      ADHD
                    </th>
                    <th scope="col" className="px-3 py-3 font-semibold">
                      IG residual
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {result.epoch_analyses.map((epoch) => (
                    <tr key={epoch.epoch_index} className="text-slate-700">
                      <td className="px-3 py-3 font-medium">
                        {epoch.epoch_index}
                      </td>
                      <td className="px-3 py-3">{epoch.predicted_class}</td>
                      <td className="px-3 py-3">
                        {formatPercent(epoch.model_output_probabilities.control)}
                      </td>
                      <td className="px-3 py-3">
                        {formatPercent(epoch.model_output_probabilities.adhd)}
                      </td>
                      <td className="px-3 py-3 font-mono">
                        {formatNumber(epoch.completeness_residual)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="mt-4 rounded-xl bg-slate-50 p-4 text-sm text-slate-500">
              The backend returned no accepted epoch predictions.
            </p>
          )}
        </article>
      </div>

      <article className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <SectionHeading
          eyebrow="Rejected signal segments"
          title="Rejected epochs"
          description="Review the reasons returned by the backend. A rejected segment has no epoch prediction or attribution."
        />
        {result.quality_control.rejected_epochs.length > 0 ? (
          <ul className="mt-4 grid gap-3 sm:grid-cols-2">
            {result.quality_control.rejected_epochs.map((epoch) => (
              <li
                key={`${epoch.epoch_index}-${epoch.start_sample ?? "unknown"}`}
                className="rounded-xl border border-rose-100 bg-rose-50/50 p-4"
              >
                <p className="text-sm font-semibold text-slate-800">
                  Candidate epoch {epoch.epoch_index}
                </p>
                <p className="mt-1 text-sm leading-5 text-rose-900">
                  {epoch.reason}
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  Samples {epoch.start_sample ?? "—"}–{epoch.stop_sample ?? "—"}
                  {epoch.channel_slots.length > 0 &&
                    ` · channel slot${epoch.channel_slots.length > 1 ? "s" : ""} ${epoch.channel_slots.join(", ")}`}
                </p>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-4 rounded-xl bg-slate-50 p-4 text-sm text-slate-500">
            No rejected epochs were reported.
          </p>
        )}
      </article>

      {result.scientific_limitations.length > 0 && (
        <article className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <SectionHeading
            eyebrow="Study context"
            title="Scientific limitations returned by the API"
          />
          <ul className="mt-4 grid gap-2 sm:grid-cols-2">
            {result.scientific_limitations.map((limitation) => (
              <li
                key={limitation}
                className="flex gap-2.5 rounded-xl bg-slate-50 p-3.5 text-sm leading-5 text-slate-600"
              >
                <span
                  aria-hidden="true"
                  className="mt-2 size-1.5 shrink-0 rounded-full bg-indigo-500"
                />
                {limitation}
              </li>
            ))}
          </ul>
        </article>
      )}
    </section>
  );
}
