package com.clinicalai.platform.model;

import java.time.LocalDateTime;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Audit record for every prediction made by the system.
 * In a clinical setting, every model decision must be logged
 * with full input context for regulatory and safety review.
 */
@Entity
@Table(name = "prediction_records")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PredictionRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // Who/what requested this prediction
    @Column(name = "request_id", unique = true, nullable = false)
    private String requestId;

    @Column(name = "requested_at", nullable = false)
    private LocalDateTime requestedAt;

    // Patient context (anonymized — no PII stored)
    @Column(name = "patient_age")
    private Integer patientAge;

    @Column(name = "patient_gender")
    private String patientGender;

    @Column(name = "time_in_hospital")
    private Integer timeInHospital;

    @Column(name = "number_inpatient")
    private Integer numberInpatient;

    @Column(name = "number_emergency")
    private Integer numberEmergency;

    // Model output
    @Column(name = "risk_score", nullable = false)
    private Double riskScore;

    @Column(name = "risk_category", nullable = false)
    private String riskCategory;

    @Column(name = "flag_for_intervention", nullable = false)
    private Boolean flagForIntervention;

    @Column(name = "confidence")
    private String confidence;

    @Column(name = "inference_time_ms")
    private Double inferenceTimeMs;

    @Column(name = "model_version")
    private String modelVersion;

    // System metadata
    @Column(name = "ml_service_status")
    private String mlServiceStatus;

    @Column(name = "error_message")
    private String errorMessage;
}