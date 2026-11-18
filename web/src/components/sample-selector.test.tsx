import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SampleSelector, DATASET_SAMPLES } from "./sample-selector";

describe("SampleSelector", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders all 4 benchmark test samples from Mendeley dataset", () => {
    render(<SampleSelector onSelectSample={vi.fn()} />);

    expect(screen.getByText("Female Control (HC)")).toBeInTheDocument();
    expect(screen.getByText("Female ADHD")).toBeInTheDocument();
    expect(screen.getByText("Male Control (HC)")).toBeInTheDocument();
    expect(screen.getByText("Male ADHD")).toBeInTheDocument();

    const loadButtons = screen.getAllByRole("button", { name: /load for testing/i });
    expect(loadButtons.length).toBe(4);
  });

  it("loads a selected sample file when clicking Load for Testing", async () => {
    const user = userEvent.setup();
    const mockOnSelect = vi.fn();
    const fakeCsvContent = "channel_slot_1,channel_slot_2\n1.0,2.0\n3.0,4.0";

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      text: async () => fakeCsvContent,
    } as Response);

    render(<SampleSelector onSelectSample={mockOnSelect} />);

    const firstLoadButton = screen.getAllByRole("button", { name: /load for testing/i })[0];
    await user.click(firstLoadButton);

    await waitFor(() => {
      expect(mockOnSelect).toHaveBeenCalledTimes(1);
    });

    const passedFile = mockOnSelect.mock.calls[0][0];
    expect(passedFile).toBeInstanceOf(File);
    expect(passedFile.name).toBe(DATASET_SAMPLES[0].filename);
  });
});
