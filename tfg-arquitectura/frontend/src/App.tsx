import { Routes, Route, Navigate } from "react-router-dom";

import Landing from "./pages/Landing";
import Layout from "./pages/Layout";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import ExperimentsList from "./pages/ExperimentsList";
import ExperimentDetail from "./pages/ExperimentDetail";
import NewExperiment from "./pages/NewExperiment";
import Comparison from "./pages/Comparison";
import RunDetail from "./pages/RunDetail";
import Simulator from "./pages/Simulator";
import Today from "./pages/Today";
import { useAuth } from "./lib/auth";

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { token } = useAuth();
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />

      {/* Protected */}
      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route path="/experiments" element={<ExperimentsList />} />
        <Route path="/experiments/new" element={<NewExperiment />} />
        <Route path="/experiments/:id" element={<ExperimentDetail />} />
        <Route path="/runs/:id" element={<RunDetail />} />
        <Route path="/compare" element={<Comparison />} />
        <Route path="/simulator" element={<Simulator />} />
        <Route path="/today" element={<Today />} />
      </Route>
    </Routes>
  );
}
