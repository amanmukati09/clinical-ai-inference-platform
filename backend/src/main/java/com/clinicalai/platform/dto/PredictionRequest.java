package com.clinicalai.platform.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

/**
 * Incoming request from frontend or API consumer.
 * Validated before forwarding to ML service.
 */
@Data
public class PredictionRequest {

    @NotBlank
    private String race;

    @NotBlank
    @Pattern(regexp = "Male|Female")
    private String gender;

    @NotNull @Min(0) @Max(100)
    private Integer age;

    @NotNull @Min(1) @Max(8)
    private Integer admissionTypeId;

    @NotNull @Min(1) @Max(30)
    private Integer dischargeDispositionId;

    @NotNull @Min(1) @Max(26)
    private Integer admissionSourceId;

    @NotNull @Min(1) @Max(14)
    private Integer timeInHospital;

    @NotNull @Min(0)
    private Integer numLabProcedures;

    @NotNull @Min(0)
    private Integer numProcedures;

    @NotNull @Min(0)
    private Integer numMedications;

    @NotNull @Min(0)
    private Integer numberOutpatient;

    @NotNull @Min(0)
    private Integer numberEmergency;

    @NotNull @Min(0)
    private Integer numberInpatient;

    @NotNull @Min(0) @Max(8)
    private Integer diag1;

    @NotNull @Min(0) @Max(8)
    private Integer diag2;

    @NotNull @Min(0) @Max(8)
    private Integer diag3;

    @NotNull @Min(0)
    private Integer numberDiagnoses;

    @NotNull @Min(0) @Max(3)
    private Integer metformin;

    @NotNull @Min(0) @Max(3)
    private Integer repaglinide;

    @NotNull @Min(0) @Max(3)
    private Integer nateglinide;

    @NotNull @Min(0) @Max(3)
    private Integer chlorpropamide;

    @NotNull @Min(0) @Max(3)
    private Integer glimepiride;

    @NotNull @Min(0) @Max(3)
    private Integer acetohexamide;

    @NotNull @Min(0) @Max(3)
    private Integer glipizide;

    @NotNull @Min(0) @Max(3)
    private Integer glyburide;

    @NotNull @Min(0) @Max(3)
    private Integer tolbutamide;

    @NotNull @Min(0) @Max(3)
    private Integer pioglitazone;

    @NotNull @Min(0) @Max(3)
    private Integer rosiglitazone;

    @NotNull @Min(0) @Max(3)
    private Integer acarbose;

    @NotNull @Min(0) @Max(3)
    private Integer miglitol;

    @NotNull @Min(0) @Max(3)
    private Integer troglitazone;

    @NotNull @Min(0) @Max(3)
    private Integer tolazamide;

    @NotNull @Min(0) @Max(3)
    private Integer examide;

    @NotNull @Min(0) @Max(3)
    private Integer citoglipton;

    @NotNull @Min(0) @Max(3)
    private Integer insulin;

    @NotNull @Min(0) @Max(3)
    private Integer glyburideMetformin;

    @NotNull @Min(0) @Max(3)
    private Integer glipizideMetformin;

    @NotNull @Min(0) @Max(3)
    private Integer glimepridePioglitazone;

    @NotNull @Min(0) @Max(3)
    private Integer metforminRosiglitazone;

    @NotNull @Min(0) @Max(3)
    private Integer metforminPioglitazone;

    @NotNull @Min(0) @Max(1)
    private Integer change;

    @NotNull @Min(0) @Max(1)
    private Integer diabetesMed;
}