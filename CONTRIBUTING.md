# Contributing

Install the development extra and run the checks listed in the README before
submitting a change. Keep changes focused and include a test that demonstrates
any corrected behavior.

When changing preprocessing, splitting, model selection or checkpoint formats,
inspect both the producer and consumer. Preserve the independence of the test
split. Do not tune settings against held-out test results.

Use synthetic images in tests. Keep source datasets, model weights, generated
runs, credentials and academic documents out of commits. Never include real
patient data in issue attachments or test fixtures.

Describe the behavior changed, why it changed and how it was validated in the
pull request. A dependency or hardware change must not be presented as a model
quality improvement without a new controlled experiment.
