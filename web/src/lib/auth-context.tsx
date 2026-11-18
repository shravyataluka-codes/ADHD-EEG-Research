"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  institution?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password?: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
}

const STORAGE_KEY = "drushti_detect_auth_user";

// Default demo credentials and default researcher account
export const DEMO_CREDENTIALS = {
  email: "researcher@drushti.ai",
  password: "adhd-eeg-2026",
  name: "Dr. Sarah Chen",
  role: "Lead EEG Researcher",
  institution: "NeuroAnalytics Research Lab",
};

const defaultContextValue: AuthContextType = {
  user: {
    id: "demo-user-1",
    name: DEMO_CREDENTIALS.name,
    email: DEMO_CREDENTIALS.email,
    role: DEMO_CREDENTIALS.role,
    institution: DEMO_CREDENTIALS.institution,
  },
  isAuthenticated: true,
  isLoading: false,
  login: async () => ({ success: true }),
  logout: () => {},
};

const AuthContext = createContext<AuthContextType>(defaultContextValue);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setUser(JSON.parse(stored));
      } else {
        // Default to demo session for ease of clinical evaluation
        const initialUser: User = {
          id: "demo-user-1",
          name: DEMO_CREDENTIALS.name,
          email: DEMO_CREDENTIALS.email,
          role: DEMO_CREDENTIALS.role,
          institution: DEMO_CREDENTIALS.institution,
        };
        setUser(initialUser);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(initialUser));
      }
    } catch {
      // In case localStorage is blocked or unavailable
      setUser({
        id: "demo-user-1",
        name: DEMO_CREDENTIALS.name,
        email: DEMO_CREDENTIALS.email,
        role: DEMO_CREDENTIALS.role,
        institution: DEMO_CREDENTIALS.institution,
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(
    async (email: string, password?: string): Promise<{ success: boolean; error?: string }> => {
      // Simulate network authentication delay
      await new Promise((resolve) => setTimeout(resolve, 350));

      const cleanEmail = email.trim().toLowerCase();
      if (!cleanEmail) {
        return { success: false, error: "Please enter an email address." };
      }

      if (password && password.length < 4) {
        return { success: false, error: "Password must be at least 4 characters." };
      }

      // Check if credentials match demo or allow any institutional email with correct demo password
      const isDemo = cleanEmail === DEMO_CREDENTIALS.email.toLowerCase();
      if (password && isDemo && password !== DEMO_CREDENTIALS.password) {
        return {
          success: false,
          error: "Invalid password for this account. Use demo password: " + DEMO_CREDENTIALS.password,
        };
      }

      const loggedUser: User = {
        id: isDemo ? "demo-user-1" : `user-${Date.now()}`,
        name: isDemo ? DEMO_CREDENTIALS.name : cleanEmail.split("@")[0].replace(/[\._]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
        email: cleanEmail,
        role: isDemo ? DEMO_CREDENTIALS.role : "Investigator",
        institution: isDemo ? DEMO_CREDENTIALS.institution : "Clinical Research Center",
      };

      setUser(loggedUser);
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(loggedUser));
      } catch {
        // ignore storage errors
      }

      return { success: true };
    },
    [],
  );

  const logout = useCallback(() => {
    setUser(null);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore storage errors
    }
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
