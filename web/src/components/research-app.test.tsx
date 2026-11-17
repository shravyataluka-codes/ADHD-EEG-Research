import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ResearchApp } from "@/components/research-app";
import { analyzeRecording } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  analyzeRecording: vi.fn(),
}));

describe("ResearchApp upload and errors", () => {
  beforeEach(() => {
    vi.mocked(analyzeRecording).mockReset();
  });

  it("submits a valid CSV and displays a safe service error", async () => {
    const user = userEvent.setup();
    vi.mocked(analyzeRecording).mockRejectedValue(
      new Error("Could not reach the analysis service."),
    );
    render(<ResearchApp />);

    expect(
      screen.getByRole("link", { name: "Drushti Detect home" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Drushti Detect", { exact: true })).toBeInTheDocument();

    const fileInput = screen.getByLabelText("Choose EEG CSV file");
    const file = new File(["channel_a,channel_b\n0.1,0.2"], "sample.csv", {
      type: "text/csv",
    });
    fireEvent.change(fileInput, { target: { files: [file] } });

    await user.click(screen.getByRole("button", { name: "Analyze recording" }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Could not reach the analysis service.",
      );
    });
    expect(analyzeRecording).toHaveBeenCalledWith(file, 256);
  });

  it("rejects an incorrect model sampling rate before the network call", async () => {
    const user = userEvent.setup();
    render(<ResearchApp />);
    fireEvent.change(screen.getByLabelText("Choose EEG CSV file"), {
      target: {
        files: [
          new File(["0.1,0.2"], "sample.csv", { type: "text/csv" }),
        ],
      },
    });
    fireEvent.change(screen.getByLabelText("Sampling rate in hertz"), {
      target: { value: "128" },
    });

    await user.click(screen.getByRole("button", { name: "Analyze recording" }));

    expect(
      screen.getByRole("alert"),
    ).toHaveTextContent("This model requires a sampling rate of 256 Hz.");
    expect(analyzeRecording).not.toHaveBeenCalled();
  });
});
