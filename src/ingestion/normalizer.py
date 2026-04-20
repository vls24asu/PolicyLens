"""Post-extraction normalisation helpers.

Enriches drug and indication entities with authoritative identifiers
(RxNorm CUIs, ICD-10 codes) after the VLM has produced an ExtractedPolicy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.policy import Drug, ExtractedPolicy, Indication
    from src.reference.icd10 import ICD10Client
    from src.reference.rxnorm import RxNormClient

logger = logging.getLogger(__name__)


def enrich_drugs(
    drugs: list["Drug"],
    rxnorm: "RxNormClient",
) -> list["Drug"]:
    """Fill in missing RxNorm CUIs and drug classes for a list of drugs.

    Only calls the API for drugs that don't already have a CUI.  Mutates
    and returns the same list so the caller can do ``drugs = enrich_drugs(drugs, client)``.
    """
    for drug in drugs:
        if drug.rxnorm_cui:
            continue
        result = rxnorm.normalize(drug.name)
        if result:
            drug.rxnorm_cui = result.rxnorm_cui
            if not drug.generic_name and result.standard_name:
                drug.generic_name = result.standard_name
            if not drug.brand_names and result.brand_names:
                drug.brand_names = result.brand_names
            if not drug.drug_class and result.drug_classes:
                drug.drug_class = result.drug_classes[0]
            logger.debug(
                "Enriched %s → CUI=%s, class=%s",
                drug.name, drug.rxnorm_cui, drug.drug_class,
            )
        else:
            logger.debug("RxNorm: no match for '%s'", drug.name)
    return drugs


def enrich_indications(
    indications: list["Indication"],
    icd10: "ICD10Client",
) -> list["Indication"]:
    """Fill in missing ICD-10 codes for a list of indications."""
    for ind in indications:
        if ind.icd10_codes:
            continue
        results = icd10.search(ind.name, max_results=5)
        if results:
            ind.icd10_codes = [r.code for r in results]
            logger.debug(
                "Enriched indication '%s' → %s",
                ind.name, ind.icd10_codes,
            )
        else:
            logger.debug("ICD-10: no match for '%s'", ind.name)
    return indications


def enrich_policy(
    extracted: "ExtractedPolicy",
    rxnorm: "RxNormClient",
    icd10: "ICD10Client",
) -> "ExtractedPolicy":
    """Run all enrichment passes on a full ExtractedPolicy in place."""
    enrich_drugs(extracted.drugs, rxnorm)
    enrich_indications(extracted.indications, icd10)
    return extracted
