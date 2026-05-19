package com.clinicalai.platform.repository;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import com.clinicalai.platform.model.PredictionRecord;

@Repository
public interface PredictionRepository
        extends JpaRepository<PredictionRecord, Long> {

    List<PredictionRecord> findByRequestedAtBetween(
            LocalDateTime start, LocalDateTime end
    );

    List<PredictionRecord> findByRiskCategory(String riskCategory);

    @Query("SELECT AVG(p.riskScore) FROM PredictionRecord p " +
           "WHERE p.requestedAt >= :since")
    Double findAverageRiskScoreSince(LocalDateTime since);

    @Query("SELECT COUNT(p) FROM PredictionRecord p " +
           "WHERE p.flagForIntervention = true " +
           "AND p.requestedAt >= :since")
    Long countInterventionsFlaggedSince(LocalDateTime since);
}