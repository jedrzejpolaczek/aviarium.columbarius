"""Fixtures shared by the scripts/ entry-point tests.

These tests call each script's ``main()`` directly, so every side effect a real
invocation has happens inside the test run unless it is redirected. The two that
used to land in the project's real ``logs/`` directory — timestamped pipeline
logs and the durable alert log — are handled suite-wide by autouse fixtures in
tests/conftest.py, which cover every ``scripts.*`` module rather than a
hand-maintained list.
"""
