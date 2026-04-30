"""
DEKS Viver — AI Interaction Runtime.

Package structure (как советовал GPT):
  session.py      — ViewerSession (state + memory)
  controller.py   — ViewerController (abstract backend API)
  browser.py      — PlaywrightBackend (concrete implementation)
  goals.py        — GoalPlanner (observe → plan → act → verify → retry)
  permissions.py  — Action sandbox (safe vs dangerous)
  server.py       — HTTP Runtime entry point (run as __main__)
"""
