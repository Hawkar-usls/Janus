# Rowan BBC1 Stage 2: eight-mutant preregistered pilot

Scientific question: can the predicted model confidence distinguish BBC1 point
mutants spanning different processed ProteinGym-DMS stability-assay labels?

Do not interpret confidence (pTM/avg lDDT) as protein fitness, stability or
delta-delta-G. This is an exploratory correspondence test, not new wet lab
data and not a general benchmark accuracy score.

PRE-REGISTRATION BEFORE PROVIDER SUBMISSION:
Source fixed: genbio-ai/ProteinGYM-DMS revision
cc6b0b56553e694dbf27dc84c6b37a8686e64b85;
BBC1_YEAST_Tsuboyama_2023_1TG0.tsv;
SHA256 ca355a655381c95fe899d61a406251e0cd17d68bbabe287651d526fdeb821f87.
Exactly 1,084 unique canonical single-amino-acid variants are required.
Remove previously computed K5A. Sort all remaining candidates by
(processed label, sequence); target percentiles 5,15,30,45,60,75,90,98.
Pick nearest mutant with a previously unused residue position; ties resolve
by deterministic sequence ordering and closest index. Eight distinct sites
must result. No selection based on Rowan outputs is permitted.

SPENDING:
One submission per daily run, eight total distinct names ever. The provider
must verify correct folder UUID, row identity, free weekly organizational
balance >=30 and no personal paid subscription. max_credits=10 each;
stage-2 upper budget 80, stage 1 upper budget 20. No retries of failed
cases. User-supplied 500 nonweekly bonus stays reserved and is never used
as fallback when weekly balance is below floor.

RESULTS:
Every run reads statuses of all eight sample tasks and the existing WT
baseline, saving sanitized public-only aggregate evidence. The Spearman
rank association with processed labels is computed only when all eight
have finite scores, and is labeled DESCRIPTIVE OF STRATIFIED SAMPLE ONLY.
Selection by observed assay label deliberately distorts population sampling;
therefore such a Spearman number must NOT be presented as an unbiased
estimate of model quality for all 1084 mutations. Independent TOPA
scientific review and original raw assay score-transform calibration
remain mandatory before any biological interpretation.

Automatic expansion beyond these eight cases is NOT authorized.
GitHub schedule is best effort, not a continuously running research agent.
