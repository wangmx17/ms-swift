# fix(examples): preserve GDN beta-zero gradients

GDN backward lost part of dbeta at beta=0. Apply a hash-pinned patch to a separately obtained extension archive; internal upstream source is not vendored. Original dbeta relative L2 versus FP32 4.8083% failed the unchanged 2% gate; repaired0.431698%, six packed and three dense regressions passed. No performance gain is claimed.

Stack base: main

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
