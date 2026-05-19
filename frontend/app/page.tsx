"use client";

import { useState } from "react";

interface PredictionResponse {
  requestId: string;
  timestamp: string;
  readmissionRiskScore: number;
  riskCategory: string;
  flagForIntervention: boolean;
  confidence: string;
  inferenceTimeMs: number;
  modelVersion: string;
}

const defaultForm = {
  race: "Caucasian",
  gender: "Female",
  age: 65,
  admissionTypeId: 1,
  dischargeDispositionId: 1,
  admissionSourceId: 7,
  timeInHospital: 5,
  numLabProcedures: 45,
  numProcedures: 2,
  numMedications: 15,
  numberOutpatient: 0,
  numberEmergency: 1,
  numberInpatient: 2,
  diag1: 4,
  diag2: 1,
  diag3: 0,
  numberDiagnoses: 9,
  metformin: 1,
  repaglinide: 0,
  nateglinide: 0,
  chlorpropamide: 0,
  glimepiride: 0,
  acetohexamide: 0,
  glipizide: 0,
  glyburide: 0,
  tolbutamide: 0,
  pioglitazone: 0,
  rosiglitazone: 0,
  acarbose: 0,
  miglitol: 0,
  troglitazone: 0,
  tolazamide: 0,
  examide: 0,
  citoglipton: 0,
  insulin: 2,
  glyburideMetformin: 0,
  glipizideMetformin: 0,
  glimepridePioglitazone: 0,
  metforminRosiglitazone: 0,
  metforminPioglitazone: 0,
  change: 1,
  diabetesMed: 1,
};

function getRiskColor(category: string) {
  if (category === "HIGH") return "bg-red-50 border-red-400 text-red-800";
  if (category === "MEDIUM") return "bg-yellow-50 border-yellow-400 text-yellow-800";
  return "bg-green-50 border-green-400 text-green-800";
}

function getRiskBadgeColor(category: string) {
  if (category === "HIGH") return "bg-red-600";
  if (category === "MEDIUM") return "bg-yellow-500";
  return "bg-green-600";
}

export default function Home() {
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: isNaN(Number(value)) ? value : Number(value),
    }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("http://localhost:8080/api/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error(`Backend error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const Field = ({
    label, name, type = "number", options,
  }: {
    label: string;
    name: string;
    type?: string;
    options?: string[];
  }) => (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">
        {label}
      </label>
      {options ? (
        <select
          name={name}
          value={String(form[name as keyof typeof form])}
          onChange={handleChange}
          className="border border-gray-300 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        >
          {options.map((o) => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
      ) : (
        <input
          type={type}
          name={name}
          value={form[name as keyof typeof form]}
          onChange={handleChange}
          className="border border-gray-300 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      )}
    </div>
  );

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-8 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">
              Clinical AI — Readmission Risk Platform
            </h1>
            <p className="text-sm text-gray-500 mt-0.5">
              30-day readmission prediction for diabetic patients · UCI Diabetes Dataset
            </p>
          </div>
          <span className="text-xs bg-blue-100 text-blue-800 px-3 py-1 rounded-full font-medium">
            v1.0.0 · PyTorch MLP · AUC 0.67
          </span>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-8 py-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
          <h2 className="text-base font-semibold text-gray-800 mb-4">
            Patient Encounter Data
          </h2>

          <div className="grid grid-cols-2 gap-4">
            <Field label="Race" name="race"
              options={["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]} />
            <Field label="Gender" name="gender" options={["Female", "Male"]} />
            <Field label="Age" name="age" />
            <Field label="Time in Hospital (days)" name="timeInHospital" />
            <Field label="Admission Type ID" name="admissionTypeId" />
            <Field label="Discharge Disposition ID" name="dischargeDispositionId" />
            <Field label="Admission Source ID" name="admissionSourceId" />
            <Field label="Num Lab Procedures" name="numLabProcedures" />
            <Field label="Num Procedures" name="numProcedures" />
            <Field label="Num Medications" name="numMedications" />
            <Field label="Outpatient Visits" name="numberOutpatient" />
            <Field label="Emergency Visits" name="numberEmergency" />
            <Field label="Inpatient Visits" name="numberInpatient" />
            <Field label="Num Diagnoses" name="numberDiagnoses" />
            <Field label="Diag 1 Category (0-8)" name="diag1" />
            <Field label="Diag 2 Category (0-8)" name="diag2" />
            <Field label="Diag 3 Category (0-8)" name="diag3" />
            <Field label="Insulin (0=No,1=Steady,2=Up,3=Down)" name="insulin" />
            <Field label="Metformin (0-3)" name="metformin" />
            <Field label="Diabetes Med (0/1)" name="diabetesMed" />
            <Field label="Med Change (0/1)" name="change" />
          </div>

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="mt-6 w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold py-2.5 rounded-lg transition-colors text-sm"
          >
            {loading ? "Running inference..." : "Predict Readmission Risk"}
          </button>

          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
              {error}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="flex flex-col gap-4">
          {result ? (
            <>
              {/* Risk Score Card */}
              <div className={`rounded-xl border-2 p-6 ${getRiskColor(result.riskCategory)}`}>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-semibold uppercase tracking-wide opacity-70">
                    Readmission Risk
                  </span>
                  <span className={`text-white text-xs font-bold px-3 py-1 rounded-full ${getRiskBadgeColor(result.riskCategory)}`}>
                    {result.riskCategory}
                  </span>
                </div>
                <div className="text-5xl font-bold mb-1">
                  {(result.readmissionRiskScore * 100).toFixed(1)}%
                </div>
                <div className="text-sm opacity-70">
                  Probability of 30-day readmission
                </div>

                {/* Risk bar */}
                <div className="mt-4 h-2 bg-black/10 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-current transition-all duration-700"
                    style={{ width: `${result.readmissionRiskScore * 100}%` }}
                  />
                </div>
              </div>

              {/* Intervention Flag */}
              <div className={`rounded-xl border p-4 flex items-center gap-3 ${
                result.flagForIntervention
                  ? "bg-red-50 border-red-300"
                  : "bg-green-50 border-green-300"
              }`}>
                <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-lg ${
                  result.flagForIntervention ? "bg-red-500" : "bg-green-500"
                }`}>
                  {result.flagForIntervention ? "!" : "✓"}
                </div>
                <div>
                  <div className="font-semibold text-sm text-gray-800">
                    {result.flagForIntervention
                      ? "Flag for Care Coordination"
                      : "No Intervention Required"}
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5">
                    {result.flagForIntervention
                      ? "Assign care coordinator before discharge"
                      : "Standard discharge protocol"}
                  </div>
                </div>
              </div>

              {/* Metadata */}
              <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
                  Prediction Metadata
                </h3>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <div className="text-xs text-gray-400">Confidence</div>
                    <div className="font-medium">{result.confidence}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Inference Time</div>
                    <div className="font-medium">{result.inferenceTimeMs}ms</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Model Version</div>
                    <div className="font-medium font-mono text-xs">{result.modelVersion}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Request ID</div>
                    <div className="font-medium font-mono text-xs truncate">{result.requestId}</div>
                  </div>
                  <div className="col-span-2">
                    <div className="text-xs text-gray-400">Timestamp</div>
                    <div className="font-medium text-xs">{new Date(result.timestamp).toLocaleString()}</div>
                  </div>
                </div>
              </div>

              {/* Clinical disclaimer */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                <p className="text-xs text-gray-500">
                  <strong>Clinical Use Notice:</strong> This prediction is a decision-support
                  tool only. Final clinical decisions must be made by qualified healthcare
                  professionals. Model trained on UCI Diabetes 1999–2008 data (AUC 0.67).
                </p>
              </div>
            </>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 p-12 shadow-sm flex flex-col items-center justify-center text-center h-full">
              <div className="text-4xl mb-4">🏥</div>
              <div className="text-gray-500 text-sm">
                Fill in patient encounter data and click predict to see the readmission risk assessment.
              </div>
              <div className="mt-4 text-xs text-gray-400">
                Predictions are logged to PostgreSQL with full audit trail
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}