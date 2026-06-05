import React, { useState } from "react";
import { Briefcase, Users, BarChart3, MessageSquare } from "lucide-react";
import { JobUpload } from "./components/JobUpload";
import { RankingsTable } from "./components/RankingsTable";
import { rankingsAPI, jobsAPI } from "./services/api";
import "./index.css";

function App() {
  const [currentTab, setCurrentTab] = useState<
    "jobs" | "candidates" | "rankings" | "copilot"
  >("jobs");
  const [selectedJob, setSelectedJob] = useState<any>(null);
  const [rankings, setRankings] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [copilotResponse, setCopilotResponse] = useState("");
  const [copilotInput, setCopilotInput] = useState("");

  const handleJobCreated = async (job: any) => {
    setSelectedJob(job);
    setCurrentTab("rankings");

    // Auto-evaluate the job
    setLoading(true);
    try {
      await rankingsAPI.evaluateJob(job.id);

      // Fetch rankings
      const response = await rankingsAPI.get(job.id);
      setRankings(response.data.rankings);
    } catch (error) {
      console.error("Error evaluating job:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCopilotQuery = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedJob || !copilotInput.trim()) return;

    setLoading(true);
    try {
      const response = await rankingsAPI.copilotQuery({
        query: copilotInput,
        job_id: selectedJob.id,
      });

      setCopilotResponse(response.data.response);
      setCopilotInput("");
    } catch (error: any) {
      setCopilotResponse("Error processing query: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-3xl font-bold text-gray-900">
            🤖 Redrob AI Recruiter
          </h1>
          <p className="text-gray-600 mt-1">
            Intelligent candidate ranking with explainable AI
          </p>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-8">
            {[
              { id: "jobs", label: "Jobs", icon: Briefcase },
              { id: "candidates", label: "Candidates", icon: Users },
              { id: "rankings", label: "Rankings", icon: BarChart3 },
              { id: "copilot", label: "Copilot", icon: MessageSquare },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() =>
                  setCurrentTab(
                    id as "jobs" | "candidates" | "rankings" | "copilot",
                  )
                }
                className={`px-4 py-3 font-medium flex items-center gap-2 border-b-2 transition ${
                  currentTab === id
                    ? "border-blue-600 text-blue-600"
                    : "border-transparent text-gray-600 hover:text-gray-900"
                }`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {currentTab === "jobs" && (
          <div className="space-y-6">
            <JobUpload onJobCreated={handleJobCreated} />

            {selectedJob && (
              <div className="card">
                <h3 className="text-xl font-bold mb-4">Current Job</h3>
                <div className="grid gap-4">
                  <div>
                    <label className="text-sm font-semibold text-gray-600">
                      Title
                    </label>
                    <p className="text-lg">{selectedJob.title}</p>
                  </div>
                  <div>
                    <label className="text-sm font-semibold text-gray-600">
                      Seniority
                    </label>
                    <p className="text-lg capitalize">
                      {selectedJob.role_seniority || "Not detected"}
                    </p>
                  </div>
                  <div>
                    <label className="text-sm font-semibold text-gray-600">
                      Must-Have Skills
                    </label>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {selectedJob.must_have.map((skill: string, i: number) => (
                        <span
                          key={i}
                          className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-semibold text-gray-600">
                      Good-to-Have Skills
                    </label>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {selectedJob.good_to_have.map(
                        (skill: string, i: number) => (
                          <span
                            key={i}
                            className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm"
                          >
                            {skill}
                          </span>
                        ),
                      )}
                    </div>
                  </div>
                  {selectedJob.requirement_confidence && (
                    <div>
                      <label className="text-sm font-semibold text-gray-600">
                        Confidence
                      </label>
                      <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full"
                          style={{
                            width: `${selectedJob.requirement_confidence * 100}%`,
                          }}
                        ></div>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">
                        {(selectedJob.requirement_confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {currentTab === "rankings" && (
          <div className="space-y-6">
            {!selectedJob ? (
              <div className="card text-center py-12">
                <p className="text-gray-600">
                  Create a job first to see rankings
                </p>
              </div>
            ) : (
              <>
                <div className="card">
                  <h2 className="text-2xl font-bold">
                    Rankings for: {selectedJob.title}
                  </h2>
                </div>
                <RankingsTable rankings={rankings} loading={loading} />
              </>
            )}
          </div>
        )}

        {currentTab === "copilot" && (
          <div className="space-y-6">
            {!selectedJob ? (
              <div className="card text-center py-12">
                <p className="text-gray-600">
                  Create a job first to use the copilot
                </p>
              </div>
            ) : (
              <>
                <div className="card">
                  <h2 className="text-2xl font-bold mb-4">Recruiter Copilot</h2>
                  <p className="text-gray-600 mb-4">
                    Ask questions about candidates:
                  </p>

                  <form onSubmit={handleCopilotQuery} className="space-y-4">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={copilotInput}
                        onChange={(e) => setCopilotInput(e.target.value)}
                        placeholder="E.g., Why is Candidate A ranked above Candidate B? Or: Find candidates with leadership potential"
                        className="input-base flex-1"
                      />
                      <button
                        type="submit"
                        disabled={loading || !copilotInput.trim()}
                        className="btn-primary"
                      >
                        Ask
                      </button>
                    </div>
                  </form>

                  {copilotResponse && (
                    <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                      <p className="whitespace-pre-wrap text-sm">
                        {copilotResponse}
                      </p>
                    </div>
                  )}
                </div>

                <div className="card">
                  <h3 className="font-semibold mb-4">Example Questions:</h3>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li>• Why is the top candidate ranked above others?</li>
                    <li>• Show candidates missing only one skill</li>
                    <li>• Who has strong leadership potential?</li>
                    <li>
                      • What are the main strengths of the top 3 candidates?
                    </li>
                  </ul>
                </div>
              </>
            )}
          </div>
        )}

        {currentTab === "candidates" && (
          <div className="card">
            <h2 className="text-2xl font-bold mb-4">Candidate Management</h2>
            <p className="text-gray-600">
              Candidate management interface coming soon. Candidates are added
              automatically when you evaluate jobs.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <p className="text-gray-600 text-sm">
            © 2024 Redrob AI Recruiter. Intelligent recruitment with explainable
            AI.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
