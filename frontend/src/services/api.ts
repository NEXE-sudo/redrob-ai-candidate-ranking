import axios, { AxiosInstance } from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Jobs API
export const jobsAPI = {
  create: (data: any) => apiClient.post("/api/jobs", data),
  get: (jobId: string) => apiClient.get(`/api/jobs/${jobId}`),
  list: (limit = 50, offset = 0) =>
    apiClient.get("/api/jobs", { params: { limit, offset } }),
  checkAmbiguity: (jobId: string) =>
    apiClient.post(`/api/jobs/${jobId}/check-ambiguity`),
  getRankings: (jobId: string, limit = 50) =>
    apiClient.get(`/api/jobs/${jobId}/rankings`, { params: { limit } }),
};

// Candidates API
export const candidatesAPI = {
  create: (data: any) => apiClient.post("/api/candidates", data),
  get: (candidateId: string) => apiClient.get(`/api/candidates/${candidateId}`),
  list: (limit = 50, offset = 0) =>
    apiClient.get("/api/candidates", { params: { limit, offset } }),
  getInsights: (candidateId: string) =>
    apiClient.get(`/api/candidates/${candidateId}/insights`),
};

// Rankings API
export const rankingsAPI = {
  evaluateJob: (jobId: string) =>
    apiClient.post(`/api/rankings/evaluate-job/${jobId}`),
  get: (jobId: string, limit = 50) =>
    apiClient.get(`/api/rankings/${jobId}`, { params: { limit } }),
  copilotQuery: (data: any) =>
    apiClient.post("/api/rankings/copilot/query", data),
};
