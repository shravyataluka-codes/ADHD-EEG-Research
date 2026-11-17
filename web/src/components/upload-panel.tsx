"use client";

import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { AlertCircle, Check, FileUp, LoaderCircle, X } from "lucide-react";

interface UploadPanelProps {
  file: File | null;
  samplingRate: string;
  busy: boolean;
  error: string | null;
  onFileChange: (file: File | null) => void;
  onSamplingRateChange: (value: string) => void;
  onSubmit: () => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function UploadPanel({
  file,
  samplingRate,
  busy,
  error,
  onFileChange,
  onSamplingRateChange,
  onSubmit,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function acceptFile(candidate: File | undefined) {
    if (candidate) onFileChange(candidate);
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files[0]);
  }

  return (
    <section
      id="analysis"
      className="scroll-mt-8 rounded-3xl border border-slate-200 bg-white p-5 shadow-[0_12px_42px_-28px_rgba(27,39,76,0.28)] sm:p-7"
      aria-labelledby="upload-heading"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-indigo-600">
            New analysis
          </p>
          <h2
            id="upload-heading"
            className="mt-2 text-xl font-bold tracking-tight text-slate-900"
          >
            Upload an EEG recording
          </h2>
          <p className="mt-1 text-sm leading-6 text-slate-500">
            Your CSV is sent to the configured analysis API for this request.
          </p>
        </div>
        <span className="hidden rounded-full border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-500 sm:inline-flex">
          CSV · max 10 MiB
        </span>
      </div>

      <div
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
            setDragging(false);
          }
        }}
        onDrop={onDrop}
        className={`mt-6 rounded-2xl border border-dashed p-5 transition sm:p-7 ${
          dragging
            ? "border-indigo-500 bg-indigo-50/80"
            : "border-slate-300 bg-slate-50/75 hover:border-indigo-300"
        }`}
      >
        {file ? (
          <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
            <div className="flex min-w-0 items-center gap-3">
              <span className="grid size-11 shrink-0 place-items-center rounded-xl bg-indigo-100 text-indigo-700">
                <Check size={20} />
              </span>
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-slate-800">
                  {file.name}
                </p>
                <p className="mt-0.5 text-xs text-slate-500">
                  {formatFileSize(file.size)}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => onFileChange(null)}
              disabled={busy}
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium text-slate-600 transition hover:bg-white hover:text-slate-900 disabled:opacity-50"
              aria-label="Remove selected recording"
            >
              <X size={15} />
              Remove
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center">
            <span className="grid size-12 place-items-center rounded-2xl bg-indigo-100 text-indigo-700">
              <FileUp size={22} />
            </span>
            <p className="mt-4 text-sm font-semibold text-slate-800">
              Drop your CSV file here
            </p>
            <p className="mt-1 text-xs text-slate-500">
              or browse from your device · maximum 10 MiB
            </p>
            <input
              ref={inputRef}
              id="recording-file"
              type="file"
              accept=".csv,text/csv"
              onChange={onInputChange}
              disabled={busy}
              className="sr-only"
              aria-label="Choose EEG CSV file"
            />
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={busy}
              className="mt-4 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-indigo-200 hover:text-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Choose a file
            </button>
          </div>
        )}
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
        <label className="block">
          <span className="text-sm font-semibold text-slate-700">
            Sampling rate
          </span>
          <span className="mt-1.5 flex max-w-xs items-center rounded-xl border border-slate-200 bg-white focus-within:border-indigo-400 focus-within:ring-4 focus-within:ring-indigo-100">
            <input
              type="number"
              min="1"
              step="1"
              inputMode="numeric"
              value={samplingRate}
              onChange={(event) => onSamplingRateChange(event.target.value)}
              disabled={busy}
              aria-label="Sampling rate in hertz"
              className="w-full rounded-xl bg-transparent px-3.5 py-2.5 text-sm text-slate-900 outline-none"
            />
            <span className="pr-3.5 text-sm text-slate-400">Hz</span>
          </span>
          <span className="mt-1.5 block text-xs text-slate-500">
            This model requires 256 Hz.
          </span>
        </label>
        <button
          type="button"
          onClick={onSubmit}
          disabled={busy || !file}
          className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-indigo-700 px-5 py-3 text-sm font-semibold text-white shadow-sm shadow-indigo-950/10 transition hover:bg-indigo-800 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
        >
          {busy ? (
            <>
              <LoaderCircle size={17} className="animate-spin" />
              Analyzing recording…
            </>
          ) : (
            "Analyze recording"
          )}
        </button>
      </div>

      <p className="mt-4 rounded-xl bg-slate-50 px-3.5 py-3 text-xs leading-5 text-slate-600">
        Expected format: CSV with exactly two numeric channel columns. The
        backend validates recording dimensions and signal values. Channel
        positions are shown only as channel slots 1 and 2; no electrode names
        are assumed.
      </p>

      {busy && (
        <div
          className="mt-4 flex items-center gap-3 rounded-xl border border-indigo-100 bg-indigo-50/70 px-4 py-3"
          role="status"
          aria-live="polite"
        >
          <LoaderCircle
            className="animate-spin text-indigo-700"
            size={18}
            aria-hidden="true"
          />
          <span className="text-sm text-indigo-950">
            Processing, predicting, and generating exploratory attributions.
            Longer recordings may take more time.
          </span>
        </div>
      )}

      {error && (
        <div
          role="alert"
          className="mt-4 flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900"
        >
          <AlertCircle size={18} className="mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </section>
  );
}
