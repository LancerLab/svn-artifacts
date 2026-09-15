"""Marker package for the benchmark2 schema directory.

Exists so that `from schema import class_axis` resolves to *this* directory via
`sys.path` rather than to any same-named distribution that happens to be
installed: a regular package found earlier on `sys.path` wins over a namespace
package elsewhere. `class-axis.json` is the canonical mutation-class axis and
`schema/check_class_axis.py` is the drift checker that keeps every consumer
honest about it.
"""
