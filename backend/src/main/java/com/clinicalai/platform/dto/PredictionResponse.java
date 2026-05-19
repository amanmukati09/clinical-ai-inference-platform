package com.clinicalai.platform.dto;

import java.time.LocalDateTime;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class PredictionResponse {
    private String requestId;
    private LocalDateTime timestamp;
    private Double readmissionRiskScore;
    private String riskCategory;
    private Boolean flagForIntervention;
    private String confidence;
    private Double inferenceTimeMs;
    private String modelVersion;
}