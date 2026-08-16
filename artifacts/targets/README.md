# Paper target data

Targets are immutable acceptance anchors, never simulator inputs. Each series records its source class:

- `reported`: printed in prose/table/annotation;
- `digitized`: read from the supplied raster plot;
- `derived`: computed only from reported/digitized anchors.

Raster values include an uncertainty field. The eventual 10% gate is evaluated against the central value, while the uncertainty is shown so that scan precision is not mistaken for author-supplied raw data. Values are transcribed independently of simulator runs.
