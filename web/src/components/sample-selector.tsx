"use client";

import { useState } from "react";
import { Archive, Download, FileSpreadsheet, FlaskConical, Loader2, Sparkles } from "lucide-react";

export interface SampleItem {
  id: string;
  name: string;
  category: "Control" | "ADHD";
  gender: "Female" | "Male";
  filename: string;
  matFilename: string;
  description: string;
  details: string;
}

export const DATASET_SAMPLES: SampleItem[] = [
  {
    id: "female-hc",
    name: "Female Control (HC)",
    category: "Control",
    gender: "Female",
    filename: "FC_cell01_slot001.csv",
    matFilename: "FC_cell01_slot001.mat",
    description: "Extracted from FC.mat (Cell 1, Subject 1)",
    details: "7,680 samples · 2 channels · 256 Hz · 30.0s",
  },
  {
    id: "female-adhd",
    name: "Female ADHD",
    category: "ADHD",
    gender: "Female",
    filename: "FADHD_cell01_slot001.csv",
    matFilename: "FADHD_cell01_slot001.mat",
    description: "Extracted from FADHD.mat (Cell 1, Subject 1)",
    details: "7,680 samples · 2 channels · 256 Hz · 30.0s",
  },
  {
    id: "male-hc",
    name: "Male Control (HC)",
    category: "Control",
    gender: "Male",
    filename: "MC_cell01_slot001.csv",
    matFilename: "MC_cell01_slot001.mat",
    description: "Extracted from MC.mat (Cell 1, Subject 1)",
    details: "7,680 samples · 2 channels · 256 Hz · 30.0s",
  },
  {
    id: "male-adhd",
    name: "Male ADHD",
    category: "ADHD",
    gender: "Male",
    filename: "MADHD_cell01_slot001.csv",
    matFilename: "MADHD_cell01_slot001.mat",
    description: "Extracted from MADHD.mat (Cell 1, Subject 1)",
    details: "7,680 samples · 2 channels · 256 Hz · 30.0s",
  },
];

interface SampleSelectorProps {
  onSelectSample: (file: File) => void;
  disabled?: boolean;
}

export function SampleSelector({ onSelectSample, disabled }: SampleSelectorProps) {
  const [loadingId, setLoadingId] = useState<string | null>(null);

  async function handleLoadSample(sample: SampleItem) {
    if (disabled) return;
    setLoadingId(sample.id);

    try {
      const response = await fetch(`/samples/${sample.filename}`);
      if (!response.ok) {
        throw new Error(`Failed to load ${sample.filename}`);
      }
      const text = await response.text();
      const file = new File([text], sample.filename, { type: "text/csv" });
      onSelectSample(file);
    } catch (err) {
      console.error("Could not load sample file:", err);
    } finally {
      setLoadingId(null);
    }
  }

  return (
    <div className="rounded-3xl border border-indigo-100 bg-white/80 p-5 shadow-sm sm:p-6">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-[0.16em] text-indigo-700">
            <FlaskConical size={14} />
            Actual Mendeley Dataset Test Recordings
          </div>
          <h2 className="mt-1 text-base font-bold text-slate-900">
            4 Benchmark Testing Files (Male & Female ADHD vs HC)
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Extracted directly from FC.mat, FADHD.mat, MC.mat, and MADHD.mat (Cell 1 baseline). Ready to load or download.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <a
            href="/samples/ADHD_Testing_Samples.zip"
            download="ADHD_Testing_Samples.zip"
            className="inline-flex items-center gap-1.5 rounded-xl border border-indigo-200 bg-indigo-50/80 px-3.5 py-2 text-xs font-semibold text-indigo-900 transition hover:bg-indigo-100 hover:border-indigo-300 shadow-sm"
          >
            <Archive size={14} />
            <span>Download All 4 (ZIP)</span>
          </a>
        </div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {DATASET_SAMPLES.map((sample) => {
          const isAdhd = sample.category === "ADHD";
          const isLoading = loadingId === sample.id;

          return (
            <div
              key={sample.id}
              className={`flex flex-col justify-between rounded-2xl border p-4 transition ${
                isAdhd
                  ? "border-amber-200/90 bg-amber-50/30 hover:border-amber-300 hover:bg-amber-50/60"
                  : "border-indigo-100 bg-indigo-50/20 hover:border-indigo-200 hover:bg-indigo-50/50"
              }`}
            >
              <div>
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                      isAdhd
                        ? "bg-amber-100 text-amber-900"
                        : "bg-teal-100 text-teal-900"
                    }`}
                  >
                    {sample.category}
                  </span>
                  <span className="text-[10px] font-semibold text-slate-500">
                    {sample.gender}
                  </span>
                </div>

                <h3 className="mt-2.5 text-sm font-bold text-slate-900">
                  {sample.name}
                </h3>
                <p className="mt-1 font-mono text-xs font-semibold text-slate-800">
                  {sample.filename}
                </p>
                <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                  {sample.description}
                </p>
                <p className="mt-1.5 font-mono text-[10px] text-slate-400">
                  {sample.details}
                </p>
              </div>

              <div className="mt-4 flex flex-col gap-2 border-t border-slate-100 pt-3">
                <button
                  type="button"
                  onClick={() => handleLoadSample(sample)}
                  disabled={disabled || isLoading}
                  className={`flex w-full items-center justify-center gap-1.5 rounded-xl py-2 px-3 text-xs font-semibold shadow-sm transition ${
                    isAdhd
                      ? "bg-amber-600 text-white hover:bg-amber-700"
                      : "bg-indigo-700 text-white hover:bg-indigo-800"
                  } disabled:opacity-50`}
                >
                  {isLoading ? (
                    <>
                      <Loader2 size={13} className="animate-spin" />
                      Loading...
                    </>
                  ) : (
                    <>
                      <Sparkles size={13} />
                      Load for Testing
                    </>
                  )}
                </button>
                <div className="flex items-center justify-between gap-2 text-[11px]">
                  <a
                    href={`/samples/${sample.filename}`}
                    download={sample.filename}
                    title={`Download ${sample.filename}`}
                    className="flex flex-1 items-center justify-center gap-1 rounded-lg border border-slate-200 bg-white py-1 text-slate-600 transition hover:border-indigo-300 hover:text-indigo-700"
                  >
                    <Download size={12} />
                    <span>Download CSV</span>
                  </a>
                  <a
                    href={`/samples/${sample.matFilename}`}
                    download={sample.matFilename}
                    title={`Download ${sample.matFilename}`}
                    className="flex flex-1 items-center justify-center gap-1 rounded-lg border border-slate-200 bg-white py-1 text-slate-600 transition hover:border-indigo-300 hover:text-indigo-700"
                  >
                    <Download size={12} />
                    <span>MAT</span>
                  </a>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
