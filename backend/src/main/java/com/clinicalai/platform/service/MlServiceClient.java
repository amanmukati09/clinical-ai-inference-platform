package com.clinicalai.platform.service;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import com.clinicalai.platform.dto.PredictionRequest;

import lombok.extern.slf4j.Slf4j;

/**
 * HTTP client for the FastAPI ML inference service.
 * Handles timeouts, error mapping, and request translation.
 */
@SuppressWarnings("unused")
@Slf4j
@Service
public class MlServiceClient {

    private final WebClient webClient;
    private final long timeoutSeconds;

    public MlServiceClient(
            @Value("${ml.service.base-url}") String baseUrl,
            @Value("${ml.service.timeout-seconds}") long timeoutSeconds) {
        this.webClient = WebClient.builder()
                .baseUrl(baseUrl)
                .build();
        this.timeoutSeconds = timeoutSeconds;
    }

    public Map<String, Object> predict(PredictionRequest request) {
        Map<String, Object> payload = buildMlPayload(request);

        try {
            Map response = webClient.post()
                    .uri("/predict")
                    .bodyValue(payload)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .timeout(Duration.ofSeconds(timeoutSeconds))
                    .block();

            log.info("ML service response received: riskScore={}",
                    response.get("readmission_risk_score"));
            return response;

        } catch (WebClientResponseException e) {
            log.error("ML service returned error: {} - {}",
                    e.getStatusCode(), e.getResponseBodyAsString());
            throw new RuntimeException("ML service error: " + e.getMessage());
        } catch (Exception e) {
            log.error("ML service call failed: {}", e.getMessage());
            throw new RuntimeException("ML service unreachable: " + e.getMessage());
        }
    }

    private Map<String, Object> buildMlPayload(PredictionRequest r) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("race", r.getRace());
        payload.put("gender", r.getGender());
        payload.put("age", r.getAge());
        payload.put("admission_type_id", r.getAdmissionTypeId());
        payload.put("discharge_disposition_id", r.getDischargeDispositionId());
        payload.put("admission_source_id", r.getAdmissionSourceId());
        payload.put("time_in_hospital", r.getTimeInHospital());
        payload.put("num_lab_procedures", r.getNumLabProcedures());
        payload.put("num_procedures", r.getNumProcedures());
        payload.put("num_medications", r.getNumMedications());
        payload.put("number_outpatient", r.getNumberOutpatient());
        payload.put("number_emergency", r.getNumberEmergency());
        payload.put("number_inpatient", r.getNumberInpatient());
        payload.put("diag_1", r.getDiag1());
        payload.put("diag_2", r.getDiag2());
        payload.put("diag_3", r.getDiag3());
        payload.put("number_diagnoses", r.getNumberDiagnoses());
        payload.put("metformin", r.getMetformin());
        payload.put("repaglinide", r.getRepaglinide());
        payload.put("nateglinide", r.getNateglinide());
        payload.put("chlorpropamide", r.getChlorpropamide());
        payload.put("glimepiride", r.getGlimepiride());
        payload.put("acetohexamide", r.getAcetohexamide());
        payload.put("glipizide", r.getGlipizide());
        payload.put("glyburide", r.getGlyburide());
        payload.put("tolbutamide", r.getTolbutamide());
        payload.put("pioglitazone", r.getPioglitazone());
        payload.put("rosiglitazone", r.getRosiglitazone());
        payload.put("acarbose", r.getAcarbose());
        payload.put("miglitol", r.getMiglitol());
        payload.put("troglitazone", r.getTroglitazone());
        payload.put("tolazamide", r.getTolazamide());
        payload.put("examide", r.getExamide());
        payload.put("citoglipton", r.getCitoglipton());
        payload.put("insulin", r.getInsulin());
        payload.put("glyburide_metformin", r.getGlyburideMetformin());
        payload.put("glipizide_metformin", r.getGlipizideMetformin());
        payload.put("glimepiride_pioglitazone", r.getGlimepridePioglitazone());
        payload.put("metformin_rosiglitazone", r.getMetforminRosiglitazone());
        payload.put("metformin_pioglitazone", r.getMetforminPioglitazone());
        payload.put("change", r.getChange());
        payload.put("diabetesMed", r.getDiabetesMed());
        return payload;
    }
}