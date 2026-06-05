import React, { useState } from "react";
import { Upload, Loader } from "lucide-react";
import { jobsAPI } from "../services/api";

interface JobUploadProps {
  onJobCreated?: (job: any) => void;
}

export const JobUpload: React.FC<JobUploadProps> = ({ onJobCreated }) => {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await jobsAPI.create({
        title,
        description,
        created_by: "system",
      });

      setTitle("");
      setDescription("");

      if (onJobCreated) {
        onJobCreated(response.data);
      }
    } catch (err: any) {
      setError(err.message || "Failed to create job");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card max-w-2xl">
      <h2 className="text-2xl font-bold mb-6">Create Job Posting</h2>

      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-800 rounded-lg">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Job Title</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g., Senior AI Engineer"
            className="input-base w-full"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">
            Job Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Paste job description here. Include must-have skills, good-to-have skills, and role requirements."
            className="input-base w-full h-48 resize-none"
            required
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader className="animate-spin w-5 h-5" />
              Processing...
            </>
          ) : (
            <>
              <Upload className="w-5 h-5" />
              Create Job
            </>
          )}
        </button>
      </form>

      <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-2">Example Format:</h3>
        <pre className="text-xs text-blue-800 whitespace-pre-wrap">
          {`Role: AI Engineer

Must Have:
- Python
- Machine Learning
- FastAPI

Good To Have:
- Docker
- AWS
- LLM Experience

Soft Skills:
- Communication
- Problem Solving`}
        </pre>
      </div>
    </div>
  );
};
