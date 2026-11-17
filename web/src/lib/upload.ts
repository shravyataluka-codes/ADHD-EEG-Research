export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
export const MODEL_SAMPLING_RATE_HZ = 256;

export function validateRecordingFile(file: File | null): string | null {
  if (!file) {
    return "Choose a CSV recording to continue.";
  }

  if (!file.name.toLowerCase().endsWith(".csv")) {
    return "Upload a .csv file containing exactly two numeric channel columns.";
  }

  if (file.size === 0) {
    return "The selected file is empty.";
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    return "The file exceeds the 10 MiB upload limit.";
  }

  return null;
}
