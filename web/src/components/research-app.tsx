"use client";

import { useState } from "react";
import Link from "next/link";
import {
  ArrowDown,
  ArrowRight,
  BrainCircuit,
  FileCheck2,
  Lock,
  ScanLine,
  Sparkles,
  UserCheck,
} from "lucide-react";
import { analyzeRecording } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { ResultsDashboard } from "@/components/results-dashboard";
import { SiteHeader } from "@/components/site-header";
import { UploadPanel } from "@/components/upload-panel";
import type { AnalysisResponse } from "@/lib/types";
import {
  MODEL_SAMPLING_RATE_HZ,
  validateRecordingFile,
} from "@/lib/upload";

const workflow = [
  {
    step: "01",
    title: "Upload",
    detail: "Two-channel CSV",
    icon: FileCheck2,
  },
  {
    step: "02",
    title: "Preprocess",
    detail: "Backend pipeline",
    icon: ScanLine,
  },
  {
    step: "03",
    title: "Analyze",
    detail: "Fusion model",
    icon: BrainCircuit,
  },
  {
    step: "04",
    title: "Explain",
    detail: "Exploratory attributions",
    icon: Sparkles,
  },
];

export function ResearchApp() {
  const { user, isAuthenticated } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [samplingRate, setSamplingRate] = useState(
    String(MODEL_SAMPLING_RATE_HZ),
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [resultFileName, setResultFileName] = useState("");

  function handleFileChange(candidate: File | null) {
    setResult(null);
    if (!candidate) {
      setFile(null);
      setError(null);
      return;
    }

    const validationMessage = validateRecordingFile(candidate);
    if (validationMessage) {
      setFile(null);
      setError(validationMessage);
      return;
    }

    setFile(candidate);
    setError(null);
  }

  async function handleAnalyze() {
    if (busy) return;

    const selectedFile = file;
    const fileError = validateRecordingFile(selectedFile);
    if (fileError || !selectedFile) {
      setError(fileError);
      return;
    }

    const parsedSamplingRate = Number(samplingRate);
    if (
      !Number.isFinite(parsedSamplingRate) ||
      parsedSamplingRate !== MODEL_SAMPLING_RATE_HZ
    ) {
      setError(
        `This model requires a sampling rate of ${MODEL_SAMPLING_RATE_HZ} Hz.`,
      );
      return;
    }

    setError(null);
    setResult(null);
    setBusy(true);
    try {
      const response = await analyzeRecording(
        selectedFile,
        parsedSamplingRate,
      );
      setResult(response);
      setResultFileName(file.name);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The analysis request failed. Please try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  function resetAnalysis() {
    setFile(null);
    setResult(null);
    setResultFileName("");
    setSamplingRate(String(MODEL_SAMPLING_RATE_HZ));
    setError(null);
  }

  return (
    <div className="min-h-screen">
      <SiteHeader />
      <main className="mx-auto w-full max-w-7xl px-5 pb-12 pt-8 sm:px-8 sm:pt-12">
        {/* Auth status bar */}
        {isAuthenticated && user ? (
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-indigo-100 bg-white/80 px-4 py-2.5 text-xs text-slate-600 shadow-sm">
            <div className="flex items-center gap-2">
              <span className="grid size-5 place-items-center rounded-full bg-teal-100 text-teal-800">
                <UserCheck size={12} />
              </span>
              <span>
                Authenticated as <strong className="text-slate-900">{user.name}</strong> ({user.role})
              </span>
            </div>
            <span className="text-[11px] text-slate-400">
              Session Active · Secure Research Pipeline
            </span>
          </div>
        ) : (
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-amber-200 bg-amber-50/80 px-4 py-2.5 text-xs text-amber-950 shadow-sm">
            <div className="flex items-center gap-2">
              <Lock size={14} className="text-amber-700" />
              <span>You are viewing in guest evaluation mode.</span>
            </div>
            <Link
              href="/login"
              className="font-semibold text-indigo-700 hover:text-indigo-900 underline"
            >
              Sign in with researcher credentials →
            </Link>
          </div>
        )}

        <section className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr] lg:items-center">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-white/80 px-3 py-1.5 text-xs font-semibold text-indigo-800 shadow-sm">
              <span className="size-1.5 rounded-full bg-teal-500" />
              EEG explainability research workspace
            </div>
            <h1 className="mt-5 max-w-xl text-4xl font-bold leading-[1.12] tracking-[-0.04em] text-slate-950 sm:text-5xl">
              Explore EEG signals with{" "}
              <span className="text-indigo-700">clarity.</span>
            </h1>
            <p className="mt-5 max-w-xl text-base leading-7 text-slate-600 sm:text-lg sm:leading-8">
              Drushti Detect explores EEG-based ADHD vs Control classification
              from two channel slots. Review model outputs alongside
              epoch-level and feature-level exploratory explanations.
            </p>
            <a
              href="#analysis"
              className="mt-7 inline-flex min-h-12 items-center gap-2 rounded-xl bg-indigo-700 px-5 py-3 text-sm font-semibold text-white shadow-md shadow-indigo-950/10 transition hover:bg-indigo-800"
            >
              Analyze EEG recording
              <ArrowRight size={17} aria-hidden="true" />
            </a>
            <div className="mt-5 flex items-center gap-2 text-xs text-slate-500">
              <span className="grid size-5 place-items-center rounded-full bg-slate-200/70">
                <ArrowDown size={12} />
              </span>
              CSV only · 2 numeric channels · 256 Hz
            </div>
          </div>

          <div
            className="relative overflow-hidden rounded-[1.75rem] border border-indigo-100 bg-white p-5 shadow-[0_24px_70px_-38px_rgba(43,50,112,0.35)] sm:p-7"
            aria-label="EEG research workflow overview"
          >
            <div
              aria-hidden="true"
              className="absolute -right-12 -top-20 size-56 rounded-full bg-violet-100/70 blur-3xl"
            />
            <div className="relative">
              <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
                    Analysis workflow
                  </p>
                  <p className="mt-1 text-sm font-semibold text-slate-800">
                    One recording · four transparent stages
                  </p>
                </div>
                <span className="rounded-xl bg-indigo-50 p-2.5 text-indigo-700">
                  <ActivityGlyph />
                </span>
              </div>
              <ol className="mt-5 space-y-3">
                {workflow.map((item, index) => {
                  const Icon = item.icon;
                  return (
                    <li
                      key={item.step}
                      className="flex items-center gap-3 rounded-2xl border border-slate-100 bg-slate-50/75 p-3.5"
                    >
                      <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-white text-indigo-700 shadow-sm">
                        <Icon size={19} aria-hidden="true" />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-sm font-semibold text-slate-800">
                          {item.title}
                        </span>
                        <span className="mt-0.5 block text-xs text-slate-500">
                          {item.detail}
                        </span>
                      </span>
                      <span className="text-[10px] font-bold tracking-widest text-slate-400">
                        {item.step}
                      </span>
                      {index < workflow.length - 1 && (
                        <span className="sr-only">followed by</span>
                      )}
                    </li>
                  );
                })}
              </ol>
              <p className="mt-4 rounded-xl border border-indigo-100 bg-indigo-50/60 px-3.5 py-3 text-xs leading-5 text-indigo-950">
                Signal processing, model inference, and Integrated Gradients
                run in the existing Python API. This interface does not
                implement a second model.
              </p>
            </div>
          </div>
        </section>

        <div className="mt-8 grid gap-6 xl:grid-cols-[0.88fr_1.12fr] xl:items-start">
          <UploadPanel
            file={file}
            samplingRate={samplingRate}
            busy={busy}
            error={error}
            onFileChange={handleFileChange}
            onSamplingRateChange={setSamplingRate}
            onSubmit={handleAnalyze}
          />
          <aside className="rounded-3xl border border-slate-200 bg-white/70 p-5 sm:p-6">
            <p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">
              Before you upload
            </p>
            <h2 className="mt-2 text-lg font-bold tracking-tight text-slate-900">
              A small, focused input contract
            </h2>
            <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-600">
              <li className="flex gap-2.5">
                <span className="mt-2 size-1.5 shrink-0 rounded-full bg-indigo-500" />
                CSV file with exactly two numeric columns, representing channel
                slots 1 and 2.
              </li>
              <li className="flex gap-2.5">
                <span className="mt-2 size-1.5 shrink-0 rounded-full bg-indigo-500" />
                Sampling rate must be 256 Hz; the backend performs its own
                input and signal validation.
              </li>
              <li className="flex gap-2.5">
                <span className="mt-2 size-1.5 shrink-0 rounded-full bg-indigo-500" />
                The recording is held only in this page&apos;s memory and sent
                to the configured API. It is not written to browser storage.
              </li>
            </ul>
          </aside>
        </div>

        {result && (
          <ResultsDashboard
            result={result}
            fileName={resultFileName}
            onReset={resetAnalysis}
          />
        )}
      </main>
      <footer className="border-t border-slate-200 bg-white/70">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-5 sm:px-8">
          <p className="text-xs leading-5 text-slate-500">
            Drushti Detect · Exploratory EEG research interface
          </p>
        </div>
      </footer>
    </div>
  );
}

function ActivityGlyph() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 12h4l3-8 4 16 3-8h4" />
    </svg>
  );
}
