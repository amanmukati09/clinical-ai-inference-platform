package com.clinicalai.platform.service;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.UUID;

import org.springframework.stereotype.Service;

import com.clinicalai.platform.dto.PredictionRequest;
import com.clinicalai.platform.dto.PredictionResponse;
import com.clinicalai.platform.model.PredictionRecord;
import com.clinicalai.platform.repository.PredictionRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class PredictionService {

    private final MlServiceClient mlServiceClient;
    private final PredictionRepository predictionRepository;

    public PredictionResponse predict(PredictionRequest request) {
        String requestId = UUID.randomUUID().toString();
        LocalDateTime now = LocalDateTime.now();
        log.info("Processing prediction request: {}", requestId);

        PredictionRecord record = PredictionRecord.builder()
                .requestId(requestId)
                .requestedAt(now)
                .patientAge(request.getAge())
                .patientGender(request.getGender())
                .timeInHospital(request.getTimeInHospital())
                .numberInpatient(request.getNumberInpatient())
                .numberEmergency(request.getNumberEmergency())
                .build();

        try {
            Map<String, Object> mlResponse =
                    mlServiceClient.predict(request);

            double riskScore = ((Number)
                    mlResponse.get("readmission_risk_score")).doubleValue();
            String riskCategory = (String)
                    mlResponse.get("risk_category");
            boolean flagged = (Boolean)
                    mlResponse.get("flag_for_intervention");
            String confidence = (String)
                    mlResponse.get("confidence");
            double inferenceMs = ((Number)
                    mlResponse.get("inference_time_ms")).doubleValue();
            String modelVersion = (String)
                    mlResponse.get("model_version");

            record.setRiskScore(riskScore);
            record.setRiskCategory(riskCategory);
            record.setFlagForIntervention(flagged);
            record.setConfidence(confidence);
            record.setInferenceTimeMs(inferenceMs);
            record.setModelVersion(modelVersion);
            record.setMlServiceStatus("SUCCESS");

            predictionRepository.save(record);
            log.info("Prediction saved: requestId={}, riskScore={}, category={}",
                    requestId, riskScore, riskCategory);

            return PredictionResponse.builder()
                    .requestId(requestId)
                    .timestamp(now)
                    .readmissionRiskScore(riskScore)
                    .riskCategory(riskCategory)
                    .flagForIntervention(flagged)
                    .confidence(confidence)
                    .inferenceTimeMs(inferenceMs)
                    .modelVersion(modelVersion)
                    .build();

        } catch (Exception e) {
            record.setRiskScore(-1.0);
            record.setRiskCategory("ERROR");
            record.setFlagForIntervention(false);
            record.setMlServiceStatus("FAILED");
            record.setErrorMessage(e.getMessage());
            predictionRepository.save(record);
            log.error("Prediction failed for requestId={}: {}",
                    requestId, e.getMessage());
            throw new RuntimeException("Prediction failed: " + e.getMessage());
        }
    }
}