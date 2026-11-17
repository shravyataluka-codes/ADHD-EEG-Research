import { describe, expect, it } from "vitest";
import {
  MAX_UPLOAD_BYTES,
  validateRecordingFile,
} from "@/lib/upload";

function makeFile(name: string, size: number): File {
  return new File([new Uint8Array(size)], name, { type: "text/csv" });
}

describe("validateRecordingFile", () => {
  it("accepts a non-empty CSV within the backend upload limit", () => {
    expect(validateRecordingFile(makeFile("recording.CSV", 32))).toBeNull();
  });

  it("requires a CSV file", () => {
    expect(validateRecordingFile(makeFile("recording.npy", 32))).toMatch(
      /\.csv/,
    );
  });

  it("rejects empty and oversized files", () => {
    expect(validateRecordingFile(makeFile("empty.csv", 0))).toMatch(/empty/);
    expect(
      validateRecordingFile(makeFile("large.csv", MAX_UPLOAD_BYTES + 1)),
    ).toMatch(/10 MiB/);
  });

  it("requires a selected file", () => {
    expect(validateRecordingFile(null)).toMatch(/Choose a CSV/);
  });
});
