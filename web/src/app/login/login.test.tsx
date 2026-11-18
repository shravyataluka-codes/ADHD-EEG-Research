import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import LoginPage from "./page";
import { AuthProvider, DEMO_CREDENTIALS } from "@/lib/auth-context";

// Mock next/navigation
const mockPush = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

describe("LoginPage", () => {
  it("renders the sign in form with demo credentials prefilled and visible", () => {
    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );

    expect(screen.getByRole("heading", { name: "Sign in to Drushti Detect" })).toBeInTheDocument();
    expect(screen.getByLabelText("Institutional Email")).toHaveValue(DEMO_CREDENTIALS.email);
    expect(screen.getByLabelText("Password")).toHaveValue(DEMO_CREDENTIALS.password);
    expect(screen.getByRole("button", { name: /quick demo sign-in/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign in to workspace" })).toBeInTheDocument();
  });

  it("submits the login form and redirects to workspace", async () => {
    const user = userEvent.setup();
    mockPush.mockReset();

    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );

    const submitBtn = screen.getByRole("button", { name: "Sign in to workspace" });
    await user.click(submitBtn);

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith("/");
    });
  });

  it("toggles password visibility", async () => {
    render(
      <AuthProvider>
        <LoginPage />
      </AuthProvider>,
    );

    const passwordInput = screen.getByLabelText("Password");
    expect(passwordInput).toHaveAttribute("type", "password");

    const toggleBtn = screen.getByRole("button", { name: "Show password" });
    fireEvent.click(toggleBtn);

    expect(passwordInput).toHaveAttribute("type", "text");
  });
});
