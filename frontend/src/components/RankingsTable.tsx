import React from "react";
import { BarChart, TrendingUp, AlertCircle } from "lucide-react";

interface RankingScore {
  technical_match: number;
  experience_match: number;
  project_relevance: number;
  behavior_signal: number;
  semantic_similarity: number;
}

interface Ranking {
  job_id: string;
  candidate_id: string;
  candidate_name: string;
  rank: number;
  final_score: number;
  component_scores: RankingScore;
  explanation: {
    matched_skills: string[];
    missing_skills: string[];
    strengths: string[];
    weaknesses: string[];
  };
}

interface RankingsTableProps {
  rankings: Ranking[];
  loading?: boolean;
}

export const RankingsTable: React.FC<RankingsTableProps> = ({
  rankings,
  loading = false,
}) => {
  const getScoreColor = (score: number) => {
    if (score >= 80) return "bg-green-100 text-green-800";
    if (score >= 60) return "bg-yellow-100 text-yellow-800";
    return "bg-red-100 text-red-800";
  };

  const getScoreBadgeClass = (score: number) => {
    if (score >= 80) return "high";
    if (score >= 60) return "medium";
    return "low";
  };

  if (loading) {
    return (
      <div className="card flex items-center justify-center h-48">
        <div className="text-center">
          <div className="animate-spin mb-2">⏳</div>
          <p>Computing rankings...</p>
        </div>
      </div>
    );
  }

  if (rankings.length === 0) {
    return (
      <div className="card text-center py-8">
        <p className="text-gray-500">No rankings yet. Evaluate a job first.</p>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden">
      <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
        <BarChart className="w-5 h-5" />
        Candidate Rankings
      </h3>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 border-b">
            <tr>
              <th className="px-4 py-3 text-left text-sm font-semibold">#</th>
              <th className="px-4 py-3 text-left text-sm font-semibold">
                Candidate
              </th>
              <th className="px-4 py-3 text-center text-sm font-semibold">
                Overall Score
              </th>
              <th className="px-4 py-3 text-center text-sm font-semibold">
                Technical
              </th>
              <th className="px-4 py-3 text-center text-sm font-semibold">
                Experience
              </th>
              <th className="px-4 py-3 text-center text-sm font-semibold">
                Projects
              </th>
              <th className="px-4 py-3 text-left text-sm font-semibold">
                Insights
              </th>
            </tr>
          </thead>
          <tbody>
            {rankings.map((ranking, idx) => (
              <tr
                key={ranking.candidate_id}
                className="border-b hover:bg-gray-50 transition"
              >
                <td className="px-4 py-3 font-semibold text-lg w-8">
                  {ranking.rank || idx + 1}
                </td>
                <td className="px-4 py-3">
                  <div className="font-semibold">{ranking.candidate_name}</div>
                  <div className="text-xs text-gray-500 truncate">
                    {ranking.candidate_id}
                  </div>
                </td>
                <td className="px-4 py-3 text-center">
                  <span
                    className={`score-badge ${getScoreBadgeClass(
                      ranking.final_score,
                    )}`}
                  >
                    {ranking.final_score.toFixed(1)}%
                  </span>
                </td>
                <td className="px-4 py-3 text-center text-sm">
                  {ranking.component_scores.technical_match.toFixed(0)}%
                </td>
                <td className="px-4 py-3 text-center text-sm">
                  {ranking.component_scores.experience_match.toFixed(0)}%
                </td>
                <td className="px-4 py-3 text-center text-sm">
                  {ranking.component_scores.project_relevance.toFixed(0)}%
                </td>
                <td className="px-4 py-3 text-xs">
                  <div className="space-y-1">
                    {ranking.explanation.matched_skills.length > 0 && (
                      <div className="text-green-700">
                        ✓ {ranking.explanation.matched_skills.length} skills
                      </div>
                    )}
                    {ranking.explanation.missing_skills.length > 0 && (
                      <div className="text-red-700">
                        ✗ {ranking.explanation.missing_skills.length} gaps
                      </div>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
