#!/usr/bin/env python3
"""018-A: frozen F-291 program, one real-feedback fix vs one same-budget restart.

Frozen: F-291 final/code byte-identical to 017 v1 outputs (SHA in run.json).
Feedback = demo_feedback() on FRESH GPU-side execution of the frozen program
on all 4 demos; no human patch. 2 model calls, 12288 each (budget 24576).
Seeds: both new calls 5103 (R2 differs from v1 R-291's 5102 to avoid replay).
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import run_reasoning_probe as P
import solve_program as SP

VLLM_WHEEL_NAME = "vllm-0.27.2rc1.dev91+g1f7427bc0-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256 = "5cbeedeb8262fbc7e77c5b4c33203e4dab74e8da1649a977bae833df62304a"
INSTALL_BUDGET_S = 20 * 60
SAVE_MARGIN_S = 300
NEW_REQUEST_CUTOFF_S = 80 * 60
HARD_STOP_S = 85 * 60

QUERY = "291dc1e1_0"
TASK = "291dc1e1"
GEN_MAX = 12288
THINK_BUDGET = 8192
EFFORT = "xhigh"
SEED = 5103
NEW_CALL_BUDGET = 24576
SUNK_COMPLETION_TOKENS = 22130  # v1 P0-291 12288 + F-291 9842
EXEC_PER_GRID_S = 2.0
EXEC_PER_PROGRAM_S = 20.0
FEEDBACK_MAX_CHARS = 3000
FROZEN_BLOBS = {
    'F-291dc1e1_0.final.txt': "CmBgYHB5dGhvbgpkZWYgdHJhbnNmb3JtKGdyaWQpOgogICAgaWYgbm90IGdyaWQgb3Igbm90IGdyaWRbMF06CiAgICAgICAgcmV0dXJuIFtbMF1dCgogICAgSCA9IGxlbihncmlkKQogICAgVyA9IGxlbihncmlkWzBdKQoKICAgICMgTG9jYXRlIHRoZSAwIGNvcm5lci4KICAgIHplcm8gPSBOb25lCiAgICBmb3IgciBpbiByYW5nZShIKToKICAgICAgICBmb3IgYyBpbiByYW5nZShXKToKICAgICAgICAgICAgaWYgZ3JpZFtyXVtjXSA9PSAwOgogICAgICAgICAgICAgICAgemVybyA9IChyLCBjKQogICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICBpZiB6ZXJvIGlzIG5vdCBOb25lOgogICAgICAgICAgICBicmVhawoKICAgICMgRmluZCBhbGwgbm9uLTggY2VsbHMgYW5kIGtlZXAgb25seSB0aG9zZSBjb25uZWN0ZWQgdG8gdGhlIDAgY29ybmVyLAogICAgIyB3aGljaCBzZWxlY3RzIHRoZSBmcmFtZSBhbmQgZXhjbHVkZXMgaXNvbGF0ZWQgb2JqZWN0cyBpZiBhbnkuCiAgICBzZWVuID0gW1tGYWxzZV0gKiBXIGZvciBfIGluIHJhbmdlKEgpXQogICAgc3RhY2sgPSBbemVyb10KICAgIHNlZW5bemVyb1swXV1bemVyb1sxXV0gPSBUcnVlCiAgICBmcmFtZV9jZWxscyA9IFtdCiAgICB3aGlsZSBzdGFjazoKICAgICAgICByLCBjID0gc3RhY2sucG9wKCkKICAgICAgICBmcmFtZV9jZWxscy5hcHBlbmQoKHIsIGMpKQogICAgICAgIGZvciBkciwgZGMgaW4gKCgxLCAwKSwgKC0xLCAwKSwgKDAsIDEpLCAoMCwgLTEpKToKICAgICAgICAgICAgbnIgPSByICsgZHIKICAgICAgICAgICAgbmMgPSBjICsgZGMKICAgICAgICAgICAgaWYgMCA8PSBuciA8IEggYW5kIDAgPD0gbmMgPCBXIGFuZCBub3Qgc2Vlbltucl1bbmNdIGFuZCBncmlkW25yXVtuY10gIT0gODoKICAgICAgICAgICAgICAgIHNlZW5bbnJdW25jXSA9IFRydWUKICAgICAgICAgICAgICAgIHN0YWNrLmFwcGVuZCgobnIsIG5jKSkKCiAgICBybWluID0gbWluKHIgZm9yIHIsIGMgaW4gZnJhbWVfY2VsbHMpCiAgICBybWF4ID0gbWF4KHIgZm9yIHIsIGMgaW4gZnJhbWVfY2VsbHMpCiAgICBjbWluID0gbWluKGMgZm9yIHIsIGMgaW4gZnJhbWVfY2VsbHMpCiAgICBjbWF4ID0gbWF4KGMgZm9yIHIsIGMgaW4gZnJhbWVfY2VsbHMpCgogICAgIyBGcmFtZSBpcyBlaXRoZXIgdG9wK2xlZnQgb3IgdG9wK3JpZ2h0LiAgSW5uZXIgcmVnaW9uIGlzIGp1c3QgaW5zaWRlIGl0LgogICAgaWYgemVyb1sxXSA9PSAwOgogICAgICAgICMgdG9wLWxlZnQ6IHRvcCBmcmFtZSByb3cgcm1pbiwgbGVmdCBmcmFtZSBjb2x1bW4gY21pbgogICAgICAgIGlyMCA9IHJtaW4gKyAxCiAgICAgICAgaXIxID0gcm1heCArIDEKICAgICAgICBpYzAgPSBjbWluICsgMQogICAgICAgIGljMSA9IGNtYXggKyAxCiAgICAgICAgY29sX2RpciA9IDEKICAgIGVsc2U6CiAgICAgICAgIyB0b3AtcmlnaHQ6IHRvcCBmcmFtZSByb3cgcm1pbiwgcmlnaHQgZnJhbWUgY29sdW1uIGNtYXgKICAgICAgICBpcjAgPSBybWluICsgMQogICAgICAgIGlyMSA9IHJtYXggKyAxCiAgICAgICAgaWMwID0gY21pbgogICAgICAgIGljMSA9IGNtYXgKICAgICAgICBjb2xfZGlyID0gLTEKCiAgICBoID0gaXIxIC0gaXIwCiAgICB3ID0gaWMxIC0gaWMwCgogICAgIyBEZXRlY3QgY29ubmVjdGVkIG5vbi04IHJlY3RhbmdsZXMgaW5zaWRlIHRoZSBpbm5lciByZWdpb24uCiAgICBpbm5lcl9zZWVuID0gW1tGYWxzZV0gKiB3IGZvciBfIGluIHJhbmdlKGgpXQogICAgcmVjdHMgPSBbXQogICAgZm9yIHNyIGluIHJhbmdlKGgpOgogICAgICAgIGZvciBzYyBpbiByYW5nZSh3KToKICAgICAgICAgICAgaWYgaW5uZXJfc2Vlbltzcl1bc2NdIG9yIGdyaWRbaXIwICsgc3JdW2ljMCArIHNjXSA9PSA4OgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgY29sb3IgPSBncmlkW2lyMCArIHNyXVtpYzAgKyBzY10KICAgICAgICAgICAgc3RhY2sgPSBbKHNyLCBzYyldCiAgICAgICAgICAgIGlubmVyX3NlZW5bc3JdW3NjXSA9IFRydWUKICAgICAgICAgICAgY2VsbHMgPSBbXQogICAgICAgICAgICB3aGlsZSBzdGFjazoKICAgICAgICAgICAgICAgIHIsIGMgPSBzdGFjay5wb3AoKQogICAgICAgICAgICAgICAgY2VsbHMuYXBwZW5kKChyLCBjKSkKICAgICAgICAgICAgICAgIGZvciBkciwgZGMgaW4gKCgxLCAwKSwgKC0xLCAwKSwgKDAsIDEpLCAoMCwgLTEpKToKICAgICAgICAgICAgICAgICAgICBuciA9IHIgKyBkcgogICAgICAgICAgICAgICAgICAgIG5jID0gYyArIGRjCiAgICAgICAgICAgICAgICAgICAgaWYgKDAgPD0gbnIgPCBoIGFuZCAwIDw9IG5jIDwgdyBhbmQgbm90IGlubmVyX3NlZW5bbnJdW25jXQogICAgICAgICAgICAgICAgICAgICAgICAgICAgYW5kIGdyaWRbaXIwICsgbnJdW2ljMCArIG5jXSA9PSBjb2xvcik6CiAgICAgICAgICAgICAgICAgICAgICAgIGlubmVyX3NlZW5bbnJdW25jXSA9IFRydWUKICAgICAgICAgICAgICAgICAgICAgICAgc3RhY2suYXBwZW5kKChuciwgbmMpKQogICAgICAgICAgICAjIEZvciBlYWNoIG1vbm9jaHJvbWUgY29ubmVjdGVkIGNlbGwgZ3JvdXAsIHRha2UgaXRzIGJvdW5kaW5nIGJveC4KICAgICAgICAgICAgcjEgPSBtaW4ociBmb3IgciwgYyBpbiBjZWxscykKICAgICAgICAgICAgcjIgPSBtYXgociBmb3IgciwgYyBpbiBjZWxscykgKyAxCiAgICAgICAgICAgIGMxID0gbWluKGMgZm9yIHIsIGMgaW4gY2VsbHMpCiAgICAgICAgICAgIGMyID0gbWF4KGMgZm9yIHIsIGMgaW4gY2VsbHMpICsgMQogICAgICAgICAgICByZWN0cy5hcHBlbmQoKHIxLCBjMSwgcjIsIGMyLCBjb2xvcikpCgogICAgIyBUaGUgdmlzaWJsZSBvYmplY3RzIGFyZSBleGFjdGx5IHRoZSBtYXhpbWFsIGFsbC1ub24tOCByZWN0YW5nbGVzLgogICAgIyBHcm91cCBtb25vY2hyb21lIGNvbXBvbmVudHMgYnkgdGhlaXIgYm91bmRpbmcgYm94LgogICAgYm94ZXMgPSB7fQogICAgZm9yIHIxLCBjMSwgcjIsIGMyLCBjb2xvciBpbiByZWN0czoKICAgICAgICBib3hlc1socjEsIGMxLCByMiwgYzIpXSA9IGNvbG9yCgogICAgIyBGb3IgZWFjaCBib3VuZGluZyBib3gsIHZlcmlmeSBpdCBpcyBmdWxsIG9mIG5vbi04IGFuZCByZWNvcmQgaXRzIGNlbGxzLgogICAgb2JqZWN0cyA9IFtdCiAgICBmb3IgKHIxLCBjMSwgcjIsIGMyKSwgY29sb3IgaW4gYm94ZXMuaXRlbXMoKToKICAgICAgICBvayA9IFRydWUKICAgICAgICBmb3IgcnIgaW4gcmFuZ2UocjEsIHIyKToKICAgICAgICAgICAgZm9yIGNjIGluIHJhbmdlKGMxLCBjMik6CiAgICAgICAgICAgICAgICBpZiBncmlkW2lyMCArIHJyXVtpYzAgKyBjY10gPT0gODoKICAgICAgICAgICAgICAgICAgICBvayA9IEZhbHNlCiAgICAgICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgaWYgbm90IG9rOgogICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICBpZiBvazoKICAgICAgICAgICAgb2JqID0gW10KICAgICAgICAgICAgZm9yIHJyIGluIHJhbmdlKHIxLCByMik6CiAgICAgICAgICAgICAgICBvYmouYXBwZW5kKFtncmlkW2lyMCArIHJyXVtpYzAgKyBjY10gZm9yIGNjIGluIHJhbmdlKGMxLCBjMildKQogICAgICAgICAgICBvYmplY3RzLmFwcGVuZCgocjEsIGMxLCByMiAtIHIxLCBjMiAtIGMxLCBvYmopKQoKICAgICMgRGVjaWRlIHJlYWRpbmcgb3JkZXI6IHVzZSB0aGUgbG9uZ2VyIGZyYW1lIHNpZGUgYXMgcHJpbWFyeSBkaXJlY3Rpb24uCiAgICAjIFRvcCBlZGdlIGxlbmd0aCBpcyB3ICsgMSwgc2lkZSBlZGdlIGxlbmd0aCBpcyBoICsgMS4KICAgIGlmIGggPiB3OgogICAgICAgICMgY29sdW1uLW1ham9yCiAgICAgICAgY29scyA9IHNvcnRlZChzZXQoYzEgZm9yIF8sIGMxLCBfLCBfLCBfIGluIG9iamVjdHMpLCBrZXk9bGFtYmRhIGM6IC1jIGlmIGNvbF9kaXIgPCAwIGVsc2UgYykKICAgICAgICBvcmRlcmVkID0gW10KICAgICAgICBmb3IgYyBpbiBjb2xzOgogICAgICAgICAgICBvYmpzID0gW28gZm9yIG8gaW4gb2JqZWN0cyBpZiBvWzFdID09IGNdCiAgICAgICAgICAgIG9ianMuc29ydChrZXk9bGFtYmRhIG86IG9bMF0pCiAgICAgICAgICAgIG9yZGVyZWQuZXh0ZW5kKG9ianMpCiAgICBlbHNlOgogICAgICAgICMgcm93LW1ham9yCiAgICAgICAgcm93cyA9IHNvcnRlZChzZXQocjEgZm9yIHIxLCBfLCBfLCBfLCBfIGluIG9iamVjdHMpLCBrZXk9bGFtYmRhIHI6IHIpCiAgICAgICAgb3JkZXJlZCA9IFtdCiAgICAgICAgZm9yIHIgaW4gcm93czoKICAgICAgICAgICAgb2JqcyA9IFtvIGZvciBvIGluIG9iamVjdHMgaWYgb1swXSA9PSByXQogICAgICAgICAgICBpZiBjb2xfZGlyIDwgMDoKICAgICAgICAgICAgICAgIG9ianMuc29ydChrZXk9bGFtYmRhIG86IC1vWzFdKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgb2Jqcy5zb3J0KGtleT1sYW1iZGEgbzogb1sxXSkKICAgICAgICAgICAgb3JkZXJlZC5leHRlbmQob2JqcykKCiAgICAjIFJvdGF0ZSB2ZXJ0aWNhbCByZWN0YW5nbGVzIDkwIGRlZ3JlZXMgY291bnRlci1jbG9ja3dpc2Ugc28gdGhlaXIgbG9uZwogICAgIyBzaWRlIGJlY29tZXMgaG9yaXpvbnRhbC4gIFNxdWFyZXMgYW5kIGhvcml6b250YWwgcmVjdGFuZ2xlcyBzdGF5IGFzIGlzLgogICAgcGllY2VzID0gW10KICAgIGZvciBfLCBfLCBoaCwgd3csIG1hdCBpbiBvcmRlcmVkOgogICAgICAgIGlmIGhoID4gd3c6CiAgICAgICAgICAgICMgQ291bnRlci1jbG9ja3dpc2U6IG5ldyByb3cgaiA9IG9sZCBjb2x1bW4gKHd3LTEtaiksIG9sZCByb3cgaQogICAgICAgICAgICBuZXdfaCA9IHd3CiAgICAgICAgICAgIG5ld193ID0gaGgKICAgICAgICAgICAgcm90YXRlZCA9IFtbMF0gKiBuZXdfdyBmb3IgXyBpbiByYW5nZShuZXdfaCldCiAgICAgICAgICAgIGZvciBpIGluIHJhbmdlKGhoKToKICAgICAgICAgICAgICAgIGZvciBqIGluIHJhbmdlKHd3KToKICAgICAgICAgICAgICAgICAgICByb3RhdGVkW2pdW2ldID0gbWF0W2ldW3d3IC0gMSAtIGpdCiAgICAgICAgICAgIHBpZWNlcy5hcHBlbmQocm90YXRlZCkKICAgICAgICBlbHNlOgogICAgICAgICAgICBwaWVjZXMuYXBwZW5kKG1hdCkKCiAgICBvdXRfdyA9IG1heChtYXgobGVuKHJvdykgZm9yIHJvdyBpbiBwKSBmb3IgcCBpbiBwaWVjZXMpCiAgICBvdXQgPSBbXQogICAgZm9yIHAgaW4gcGllY2VzOgogICAgICAgIHBoID0gbGVuKHApCiAgICAgICAgcHcgPSBsZW4ocFswXSkKICAgICAgICBsZWZ0ID0gKG91dF93IC0gcHcpIC8vIDIKICAgICAgICBmb3Igcm93IGluIHA6CiAgICAgICAgICAgIG91dC5hcHBlbmQoWzhdICogbGVmdCArIHJvdyArIFs4XSAqIChvdXRfdyAtIHB3IC0gbGVmdCkpCgogICAgcmV0dXJuIG91dApgYGA=",
    'F-291dc1e1_0.code.py': "ZGVmIHRyYW5zZm9ybShncmlkKToKICAgIGlmIG5vdCBncmlkIG9yIG5vdCBncmlkWzBdOgogICAgICAgIHJldHVybiBbWzBdXQoKICAgIEggPSBsZW4oZ3JpZCkKICAgIFcgPSBsZW4oZ3JpZFswXSkKCiAgICAjIExvY2F0ZSB0aGUgMCBjb3JuZXIuCiAgICB6ZXJvID0gTm9uZQogICAgZm9yIHIgaW4gcmFuZ2UoSCk6CiAgICAgICAgZm9yIGMgaW4gcmFuZ2UoVyk6CiAgICAgICAgICAgIGlmIGdyaWRbcl1bY10gPT0gMDoKICAgICAgICAgICAgICAgIHplcm8gPSAociwgYykKICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgaWYgemVybyBpcyBub3QgTm9uZToKICAgICAgICAgICAgYnJlYWsKCiAgICAjIEZpbmQgYWxsIG5vbi04IGNlbGxzIGFuZCBrZWVwIG9ubHkgdGhvc2UgY29ubmVjdGVkIHRvIHRoZSAwIGNvcm5lciwKICAgICMgd2hpY2ggc2VsZWN0cyB0aGUgZnJhbWUgYW5kIGV4Y2x1ZGVzIGlzb2xhdGVkIG9iamVjdHMgaWYgYW55LgogICAgc2VlbiA9IFtbRmFsc2VdICogVyBmb3IgXyBpbiByYW5nZShIKV0KICAgIHN0YWNrID0gW3plcm9dCiAgICBzZWVuW3plcm9bMF1dW3plcm9bMV1dID0gVHJ1ZQogICAgZnJhbWVfY2VsbHMgPSBbXQogICAgd2hpbGUgc3RhY2s6CiAgICAgICAgciwgYyA9IHN0YWNrLnBvcCgpCiAgICAgICAgZnJhbWVfY2VsbHMuYXBwZW5kKChyLCBjKSkKICAgICAgICBmb3IgZHIsIGRjIGluICgoMSwgMCksICgtMSwgMCksICgwLCAxKSwgKDAsIC0xKSk6CiAgICAgICAgICAgIG5yID0gciArIGRyCiAgICAgICAgICAgIG5jID0gYyArIGRjCiAgICAgICAgICAgIGlmIDAgPD0gbnIgPCBIIGFuZCAwIDw9IG5jIDwgVyBhbmQgbm90IHNlZW5bbnJdW25jXSBhbmQgZ3JpZFtucl1bbmNdICE9IDg6CiAgICAgICAgICAgICAgICBzZWVuW25yXVtuY10gPSBUcnVlCiAgICAgICAgICAgICAgICBzdGFjay5hcHBlbmQoKG5yLCBuYykpCgogICAgcm1pbiA9IG1pbihyIGZvciByLCBjIGluIGZyYW1lX2NlbGxzKQogICAgcm1heCA9IG1heChyIGZvciByLCBjIGluIGZyYW1lX2NlbGxzKQogICAgY21pbiA9IG1pbihjIGZvciByLCBjIGluIGZyYW1lX2NlbGxzKQogICAgY21heCA9IG1heChjIGZvciByLCBjIGluIGZyYW1lX2NlbGxzKQoKICAgICMgRnJhbWUgaXMgZWl0aGVyIHRvcCtsZWZ0IG9yIHRvcCtyaWdodC4gIElubmVyIHJlZ2lvbiBpcyBqdXN0IGluc2lkZSBpdC4KICAgIGlmIHplcm9bMV0gPT0gMDoKICAgICAgICAjIHRvcC1sZWZ0OiB0b3AgZnJhbWUgcm93IHJtaW4sIGxlZnQgZnJhbWUgY29sdW1uIGNtaW4KICAgICAgICBpcjAgPSBybWluICsgMQogICAgICAgIGlyMSA9IHJtYXggKyAxCiAgICAgICAgaWMwID0gY21pbiArIDEKICAgICAgICBpYzEgPSBjbWF4ICsgMQogICAgICAgIGNvbF9kaXIgPSAxCiAgICBlbHNlOgogICAgICAgICMgdG9wLXJpZ2h0OiB0b3AgZnJhbWUgcm93IHJtaW4sIHJpZ2h0IGZyYW1lIGNvbHVtbiBjbWF4CiAgICAgICAgaXIwID0gcm1pbiArIDEKICAgICAgICBpcjEgPSBybWF4ICsgMQogICAgICAgIGljMCA9IGNtaW4KICAgICAgICBpYzEgPSBjbWF4CiAgICAgICAgY29sX2RpciA9IC0xCgogICAgaCA9IGlyMSAtIGlyMAogICAgdyA9IGljMSAtIGljMAoKICAgICMgRGV0ZWN0IGNvbm5lY3RlZCBub24tOCByZWN0YW5nbGVzIGluc2lkZSB0aGUgaW5uZXIgcmVnaW9uLgogICAgaW5uZXJfc2VlbiA9IFtbRmFsc2VdICogdyBmb3IgXyBpbiByYW5nZShoKV0KICAgIHJlY3RzID0gW10KICAgIGZvciBzciBpbiByYW5nZShoKToKICAgICAgICBmb3Igc2MgaW4gcmFuZ2Uodyk6CiAgICAgICAgICAgIGlmIGlubmVyX3NlZW5bc3JdW3NjXSBvciBncmlkW2lyMCArIHNyXVtpYzAgKyBzY10gPT0gODoKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIGNvbG9yID0gZ3JpZFtpcjAgKyBzcl1baWMwICsgc2NdCiAgICAgICAgICAgIHN0YWNrID0gWyhzciwgc2MpXQogICAgICAgICAgICBpbm5lcl9zZWVuW3NyXVtzY10gPSBUcnVlCiAgICAgICAgICAgIGNlbGxzID0gW10KICAgICAgICAgICAgd2hpbGUgc3RhY2s6CiAgICAgICAgICAgICAgICByLCBjID0gc3RhY2sucG9wKCkKICAgICAgICAgICAgICAgIGNlbGxzLmFwcGVuZCgociwgYykpCiAgICAgICAgICAgICAgICBmb3IgZHIsIGRjIGluICgoMSwgMCksICgtMSwgMCksICgwLCAxKSwgKDAsIC0xKSk6CiAgICAgICAgICAgICAgICAgICAgbnIgPSByICsgZHIKICAgICAgICAgICAgICAgICAgICBuYyA9IGMgKyBkYwogICAgICAgICAgICAgICAgICAgIGlmICgwIDw9IG5yIDwgaCBhbmQgMCA8PSBuYyA8IHcgYW5kIG5vdCBpbm5lcl9zZWVuW25yXVtuY10KICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFuZCBncmlkW2lyMCArIG5yXVtpYzAgKyBuY10gPT0gY29sb3IpOgogICAgICAgICAgICAgICAgICAgICAgICBpbm5lcl9zZWVuW25yXVtuY10gPSBUcnVlCiAgICAgICAgICAgICAgICAgICAgICAgIHN0YWNrLmFwcGVuZCgobnIsIG5jKSkKICAgICAgICAgICAgIyBGb3IgZWFjaCBtb25vY2hyb21lIGNvbm5lY3RlZCBjZWxsIGdyb3VwLCB0YWtlIGl0cyBib3VuZGluZyBib3guCiAgICAgICAgICAgIHIxID0gbWluKHIgZm9yIHIsIGMgaW4gY2VsbHMpCiAgICAgICAgICAgIHIyID0gbWF4KHIgZm9yIHIsIGMgaW4gY2VsbHMpICsgMQogICAgICAgICAgICBjMSA9IG1pbihjIGZvciByLCBjIGluIGNlbGxzKQogICAgICAgICAgICBjMiA9IG1heChjIGZvciByLCBjIGluIGNlbGxzKSArIDEKICAgICAgICAgICAgcmVjdHMuYXBwZW5kKChyMSwgYzEsIHIyLCBjMiwgY29sb3IpKQoKICAgICMgVGhlIHZpc2libGUgb2JqZWN0cyBhcmUgZXhhY3RseSB0aGUgbWF4aW1hbCBhbGwtbm9uLTggcmVjdGFuZ2xlcy4KICAgICMgR3JvdXAgbW9ub2Nocm9tZSBjb21wb25lbnRzIGJ5IHRoZWlyIGJvdW5kaW5nIGJveC4KICAgIGJveGVzID0ge30KICAgIGZvciByMSwgYzEsIHIyLCBjMiwgY29sb3IgaW4gcmVjdHM6CiAgICAgICAgYm94ZXNbKHIxLCBjMSwgcjIsIGMyKV0gPSBjb2xvcgoKICAgICMgRm9yIGVhY2ggYm91bmRpbmcgYm94LCB2ZXJpZnkgaXQgaXMgZnVsbCBvZiBub24tOCBhbmQgcmVjb3JkIGl0cyBjZWxscy4KICAgIG9iamVjdHMgPSBbXQogICAgZm9yIChyMSwgYzEsIHIyLCBjMiksIGNvbG9yIGluIGJveGVzLml0ZW1zKCk6CiAgICAgICAgb2sgPSBUcnVlCiAgICAgICAgZm9yIHJyIGluIHJhbmdlKHIxLCByMik6CiAgICAgICAgICAgIGZvciBjYyBpbiByYW5nZShjMSwgYzIpOgogICAgICAgICAgICAgICAgaWYgZ3JpZFtpcjAgKyBycl1baWMwICsgY2NdID09IDg6CiAgICAgICAgICAgICAgICAgICAgb2sgPSBGYWxzZQogICAgICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgIGlmIG5vdCBvazoKICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgaWYgb2s6CiAgICAgICAgICAgIG9iaiA9IFtdCiAgICAgICAgICAgIGZvciByciBpbiByYW5nZShyMSwgcjIpOgogICAgICAgICAgICAgICAgb2JqLmFwcGVuZChbZ3JpZFtpcjAgKyBycl1baWMwICsgY2NdIGZvciBjYyBpbiByYW5nZShjMSwgYzIpXSkKICAgICAgICAgICAgb2JqZWN0cy5hcHBlbmQoKHIxLCBjMSwgcjIgLSByMSwgYzIgLSBjMSwgb2JqKSkKCiAgICAjIERlY2lkZSByZWFkaW5nIG9yZGVyOiB1c2UgdGhlIGxvbmdlciBmcmFtZSBzaWRlIGFzIHByaW1hcnkgZGlyZWN0aW9uLgogICAgIyBUb3AgZWRnZSBsZW5ndGggaXMgdyArIDEsIHNpZGUgZWRnZSBsZW5ndGggaXMgaCArIDEuCiAgICBpZiBoID4gdzoKICAgICAgICAjIGNvbHVtbi1tYWpvcgogICAgICAgIGNvbHMgPSBzb3J0ZWQoc2V0KGMxIGZvciBfLCBjMSwgXywgXywgXyBpbiBvYmplY3RzKSwga2V5PWxhbWJkYSBjOiAtYyBpZiBjb2xfZGlyIDwgMCBlbHNlIGMpCiAgICAgICAgb3JkZXJlZCA9IFtdCiAgICAgICAgZm9yIGMgaW4gY29sczoKICAgICAgICAgICAgb2JqcyA9IFtvIGZvciBvIGluIG9iamVjdHMgaWYgb1sxXSA9PSBjXQogICAgICAgICAgICBvYmpzLnNvcnQoa2V5PWxhbWJkYSBvOiBvWzBdKQogICAgICAgICAgICBvcmRlcmVkLmV4dGVuZChvYmpzKQogICAgZWxzZToKICAgICAgICAjIHJvdy1tYWpvcgogICAgICAgIHJvd3MgPSBzb3J0ZWQoc2V0KHIxIGZvciByMSwgXywgXywgXywgXyBpbiBvYmplY3RzKSwga2V5PWxhbWJkYSByOiByKQogICAgICAgIG9yZGVyZWQgPSBbXQogICAgICAgIGZvciByIGluIHJvd3M6CiAgICAgICAgICAgIG9ianMgPSBbbyBmb3IgbyBpbiBvYmplY3RzIGlmIG9bMF0gPT0gcl0KICAgICAgICAgICAgaWYgY29sX2RpciA8IDA6CiAgICAgICAgICAgICAgICBvYmpzLnNvcnQoa2V5PWxhbWJkYSBvOiAtb1sxXSkKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIG9ianMuc29ydChrZXk9bGFtYmRhIG86IG9bMV0pCiAgICAgICAgICAgIG9yZGVyZWQuZXh0ZW5kKG9ianMpCgogICAgIyBSb3RhdGUgdmVydGljYWwgcmVjdGFuZ2xlcyA5MCBkZWdyZWVzIGNvdW50ZXItY2xvY2t3aXNlIHNvIHRoZWlyIGxvbmcKICAgICMgc2lkZSBiZWNvbWVzIGhvcml6b250YWwuICBTcXVhcmVzIGFuZCBob3Jpem9udGFsIHJlY3RhbmdsZXMgc3RheSBhcyBpcy4KICAgIHBpZWNlcyA9IFtdCiAgICBmb3IgXywgXywgaGgsIHd3LCBtYXQgaW4gb3JkZXJlZDoKICAgICAgICBpZiBoaCA+IHd3OgogICAgICAgICAgICAjIENvdW50ZXItY2xvY2t3aXNlOiBuZXcgcm93IGogPSBvbGQgY29sdW1uICh3dy0xLWopLCBvbGQgcm93IGkKICAgICAgICAgICAgbmV3X2ggPSB3dwogICAgICAgICAgICBuZXdfdyA9IGhoCiAgICAgICAgICAgIHJvdGF0ZWQgPSBbWzBdICogbmV3X3cgZm9yIF8gaW4gcmFuZ2UobmV3X2gpXQogICAgICAgICAgICBmb3IgaSBpbiByYW5nZShoaCk6CiAgICAgICAgICAgICAgICBmb3IgaiBpbiByYW5nZSh3dyk6CiAgICAgICAgICAgICAgICAgICAgcm90YXRlZFtqXVtpXSA9IG1hdFtpXVt3dyAtIDEgLSBqXQogICAgICAgICAgICBwaWVjZXMuYXBwZW5kKHJvdGF0ZWQpCiAgICAgICAgZWxzZToKICAgICAgICAgICAgcGllY2VzLmFwcGVuZChtYXQpCgogICAgb3V0X3cgPSBtYXgobWF4KGxlbihyb3cpIGZvciByb3cgaW4gcCkgZm9yIHAgaW4gcGllY2VzKQogICAgb3V0ID0gW10KICAgIGZvciBwIGluIHBpZWNlczoKICAgICAgICBwaCA9IGxlbihwKQogICAgICAgIHB3ID0gbGVuKHBbMF0pCiAgICAgICAgbGVmdCA9IChvdXRfdyAtIHB3KSAvLyAyCiAgICAgICAgZm9yIHJvdyBpbiBwOgogICAgICAgICAgICBvdXQuYXBwZW5kKFs4XSAqIGxlZnQgKyByb3cgKyBbOF0gKiAob3V0X3cgLSBwdyAtIGxlZnQpKQoKICAgIHJldHVybiBvdXQ=",
    'task-291-clean.json': "eyJ0cmFpbiI6W3siaW5wdXQiOltbMCwyLDIsMiwyLDIsMiwyXSxbMSw4LDgsOCw4LDgsOCw4XSxbMSw4LDQsNCw4LDksOSw4XSxbMSw4LDQsNCw4LDksOSw4XSxbMSw4LDgsOCw4LDgsOCw4XSxbMSw4LDgsOCw4LDgsOCw4XSxbMSw4LDEsNiw4LDQsNCw4XSxbMSw4LDIsNiw4LDQsNCw4XSxbMSw4LDYsNiw4LDgsOCw4XSxbMSw4LDYsNiw4LDgsOCw4XSxbMSw4LDYsNiw4LDQsNyw4XSxbMSw4LDYsNiw4LDcsNCw4XSxbMSw4LDgsOCw4LDcsNyw4XSxbMSw4LDgsOCw4LDcsNyw4XSxbMSw4LDMsNiw4LDcsNyw4XSxbMSw4LDMsNiw4LDcsNyw4XSxbMSw4LDMsMyw4LDgsOCw4XSxbMSw4LDMsMyw4LDgsOCw4XSxbMSw4LDgsOCw4LDgsOCw4XSxbMSw4LDgsOCw4LDgsOCw4XV0sIm91dHB1dCI6W1s4LDgsNCw0LDgsOF0sWzgsOCw0LDQsOCw4XSxbNiw2LDYsNiw2LDZdLFsxLDIsNiw2LDYsNl0sWzgsNiw2LDMsMyw4XSxbOCwzLDMsMywzLDhdLFs4LDgsOSw5LDgsOF0sWzgsOCw5LDksOCw4XSxbOCw4LDQsNCw4LDhdLFs4LDgsNCw0LDgsOF0sWzcsNCw3LDcsNyw3XSxbNCw3LDcsNyw3LDddXX0seyJpbnB1dCI6W1swLDEsMSwxLDEsMSwxLDEsMSwxLDFdLFsyLDgsOCw4LDgsOCw4LDgsOCw4LDhdLFsyLDgsMywzLDgsOCw0LDQsNCw0LDhdLFsyLDgsMywzLDgsOCw0LDQsNCw0LDhdLFsyLDgsOCw4LDgsOCw4LDgsOCw4LDhdLFsyLDgsNiw2LDgsOCw5LDksOCw4LDhdLFsyLDgsNiw2LDgsOCw5LDksOCw4LDhdLFsyLDgsOCw4LDgsOCw4LDgsOCw4LDhdXSwib3V0cHV0IjpbWzgsMywzLDhdLFs4LDMsMyw4XSxbNCw0LDQsNF0sWzQsNCw0LDRdLFs4LDYsNiw4XSxbOCw2LDYsOF0sWzgsOSw5LDhdLFs4LDksOSw4XV19LHsiaW5wdXQiOltbMCwxLDEsMSwxLDEsMSwxLDEsMSwxLDEsMSwxLDEsMSwxLDEsMSwxLDEsMSwxLDEsMSwxLDEsMV0sWzIsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDhdLFsyLDgsMywzLDMsMywzLDMsMywzLDgsOCw5LDksOCw4LDcsNyw3LDcsOCw4LDQsNCw0LDQsOCw4XSxbMiw4LDMsMywzLDMsMywzLDMsMyw4LDgsOSw5LDgsOCw3LDcsNyw3LDgsOCw0LDQsNCw0LDgsOF0sWzIsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDhdLFsyLDgsOSw5LDgsOCw2LDYsNiw2LDYsNiw2LDYsOCw4LDQsNCw4LDgsOCw4LDgsOCw4LDgsOCw4XSxbMiw4LDksOSw4LDgsNiw2LDYsNiw2LDYsNiw2LDgsOCw0LDQsOCw4LDgsOCw4LDgsOCw4LDgsOF0sWzIsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDhdXSwib3V0cHV0IjpbWzMsMywzLDMsMywzLDMsM10sWzMsMywzLDMsMywzLDMsM10sWzgsOCw4LDksOSw4LDgsOF0sWzgsOCw4LDksOSw4LDgsOF0sWzgsOCw3LDcsNyw3LDgsOF0sWzgsOCw3LDcsNyw3LDgsOF0sWzgsOCw0LDQsNCw0LDgsOF0sWzgsOCw0LDQsNCw0LDgsOF0sWzgsOCw4LDksOSw4LDgsOF0sWzgsOCw4LDksOSw4LDgsOF0sWzYsNiw2LDYsNiw2LDYsNl0sWzYsNiw2LDYsNiw2LDYsNl0sWzgsOCw4LDQsNCw4LDgsOF0sWzgsOCw4LDQsNCw4LDgsOF1dfSx7ImlucHV0IjpbWzEsMSwxLDEsMSwxLDEsMSwxLDEsMSwxLDEsMSwxLDEsMSwxLDEsMSwxLDEsMF0sWzgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsMl0sWzgsMywzLDMsMywzLDMsOCw4LDcsNyw4LDgsMywzLDgsOCw0LDQsNCw0LDgsMl0sWzgsMywzLDMsMywzLDMsOCw4LDcsNyw4LDgsMywzLDgsOCw0LDQsNCw0LDgsMl0sWzgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsMl0sWzgsOCw4LDgsOCwzLDMsMywzLDgsOCw2LDYsNiw2LDYsNiw4LDgsOSw5LDgsMl0sWzgsOCw4LDgsOCwzLDMsMywzLDgsOCw2LDYsNiw2LDYsNiw4LDgsOSw5LDgsMl0sWzgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsOCw4LDgsMl1dLCJvdXRwdXQiOltbOCw0LDQsNCw0LDhdLFs4LDQsNCw0LDQsOF0sWzgsOCwzLDMsOCw4XSxbOCw4LDMsMyw4LDhdLFs4LDgsNyw3LDgsOF0sWzgsOCw3LDcsOCw4XSxbMywzLDMsMywzLDNdLFszLDMsMywzLDMsM10sWzgsOCw5LDksOCw4XSxbOCw4LDksOSw4LDhdLFs2LDYsNiw2LDYsNl0sWzYsNiw2LDYsNiw2XSxbOCwzLDMsMywzLDhdLFs4LDMsMywzLDMsOF1dfV0sInRlc3QiOlt7ImlucHV0IjpbWzIsMiwyLDIsMiwyLDIsMF0sWzgsOCw4LDgsOCw4LDgsMV0sWzgsMywzLDgsNCw0LDgsMV0sWzgsMywzLDgsNCw0LDgsMV0sWzgsMywzLDgsOCw4LDgsMV0sWzgsMywzLDgsOCw4LDgsMV0sWzgsMywzLDgsNiwzLDgsMV0sWzgsMyw3LDgsNiwzLDgsMV0sWzgsNywzLDgsMywzLDgsMV0sWzgsNywzLDgsMywzLDgsMV0sWzgsOCw4LDgsOCw4LDgsMV0sWzgsOCw4LDgsOCw4LDgsMV0sWzgsNSw1LDgsNiw2LDgsMV0sWzgsNSw1LDgsNiw2LDgsMV0sWzgsOCw4LDgsNiw2LDgsMV0sWzgsOCw4LDgsNiw2LDgsMV0sWzgsNyw3LDgsNiw2LDgsMV0sWzgsNyw5LDgsNiw2LDgsMV0sWzgsOSw3LDgsOCw4LDgsMV0sWzgsNyw3LDgsOCw4LDgsMV0sWzgsOCw4LDgsOSw5LDgsMV0sWzgsOCw4LDgsOSw5LDgsMV0sWzgsOSw5LDgsOCw4LDgsMV0sWzgsOSw5LDgsOCw4LDgsMV0sWzgsOSw5LDgsNCw0LDgsMV0sWzgsOSw5LDgsNCw0LDgsMV0sWzgsOCw4LDgsOCw4LDgsMV0sWzgsOCw4LDgsOCw4LDgsMV1dfV19",
}

FROZEN_SHA = {
    "F-291dc1e1_0.final.txt": "16e3c18ea494b1f866c54136db4bde6a91ad6a071f59e84ddc63af4d42f709d1",
    "F-291dc1e1_0.code.py": "cc808fdb875a395dced5601f22616e60606306e57d115812125e98a6f0a1f9de",
    "task-291-clean.json": "6a0c60cfb64acd658deb95fd4ba2fca7b932023584fabc921b691f04ae8ff1ef",
}

STATE = {"proc": None, "work": None, "run": None, "closed": False, "phase": "init"}
GPU_SAMPLER_STOP = threading.Event()


def watchdog(work: Path, interval_s: float = 60.0):
    while not GPU_SAMPLER_STOP.is_set():
        time.sleep(interval_s)
        if GPU_SAMPLER_STOP.is_set():
            break
        try:
            with open(work / "driver-watch.log", "a") as h:
                h.write(f"[{time.strftime('%H:%M:%S')}] alive phase={STATE.get('phase')}\n")
                faulthandler.dump_traceback(file=h)
                h.write("---\n")
        except Exception:
            pass


def log(msg):
    line = f"[018a {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        if STATE.get("work") is not None:
            with open(STATE["work"] / "driver-stdout.log", "a") as h:
                h.write(line + "\n")
    except Exception:
        pass


def sh(cmd, timeout, cwd=None, env=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for ch in iter(lambda: h.read(1 << 20), 1):
            d.update(ch)
    return d.hexdigest()


def save(work: Path, run: dict) -> None:
    (work / "run.json").write_text(json.dumps(run, indent=1))


def orderly(work, run, status, code=0):
    run["status"] = status
    started = run.pop("_t0", None)
    run["elapsed_s"] = round(time.time() - (started if started is not None else time.time()), 1)
    import shutil
    shutil.rmtree(work / "site", ignore_errors=True)
    GPU_SAMPLER_STOP.set()
    save(work, run)
    log(f"EVIDENCE-EXIT status={status}")
    return code


def finish(status, code=0, detail=None):
    if STATE["closed"]:
        return code
    STATE["closed"] = True
    run, work = STATE["run"], STATE["work"]
    if detail and run is not None and "first_error" not in run:
        run["first_error"] = str(detail)[:800]
    proc = STATE.get("proc")
    if proc is not None and work is not None:
        P.kill_server(proc, work)
        STATE["proc"] = None
    if run is None or work is None:
        GPU_SAMPLER_STOP.set()
        return code
    return orderly(work, run, status, code)


def on_term(signum, _frame):
    finish(f"signal_{signum}")
    os._exit(0)


def server_alive() -> bool:
    proc = STATE.get("proc")
    return proc is not None and proc.poll() is None


def server_tail(work: Path, n=40) -> str:
    p = work / "vllm-server.log"
    if not exists(p):
        return ""
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:])


def exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def count_preempt(log_text: str) -> int:
    t = log_text.lower()
    return t.count("preempt") + t.count("recompute")


def gpu_sampler(work: Path, interval_s: float = 5.0):
    out = work / "gpu-samples.jsonl"
    while not GPU_SAMPLER_STOP.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            line = {"t_wall": round(time.time(), 1),
                    "raw": r.stdout.strip()[:800], "rc": r.returncode}
        except Exception as exc:
            line = {"t_wall": round(time.time(), 1),
                    "error": f"{type(exc).__name__}:{exc}"[:200]}
        try:
            with out.open("a") as h:
                h.write(json.dumps(line) + "\n")
        except Exception:
            pass
        GPU_SAMPLER_STOP.wait(interval_s)


def time_left(session_deadline: float, margin: float = SAVE_MARGIN_S) -> float:
    return session_deadline - time.monotonic() - margin


def past_cutoff(t0_mono: float) -> bool:
    return (time.monotonic() - t0_mono) >= NEW_REQUEST_CUTOFF_S


def compact(grid) -> str:
    return json.dumps(grid, separators=(",", ":"), ensure_ascii=False)


def build_base_prompt(task: dict) -> tuple[str, str, list]:
    lines = ["Solve the grid transformation. Demonstrations (JSON input -> output):"]
    for i, d in enumerate(task["train"]):
        lines.append(f"demo{i} in={compact(d['input'])} out={compact(d['output'])}")
    lines.append(f"test in={compact(task['test'][0]['input'])}")
    lines.append("Return exactly one closed ```python block with transform(grid). No other text after the block.")
    user = "\n".join(lines)
    messages = [{"role": "system", "content": SP.SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    return SP.SYSTEM_PROMPT, user, messages


def is_legal_grid(grid) -> bool:
    try:
        SP.checked_grid(grid)
        return True
    except Exception:
        return False


def parse_generation(rec: dict, res: dict) -> dict:
    finish = res.get("finish")
    if finish == "stop" and not rec.get("client_error") and not rec.get("cancelled_by_client"):
        parsed = SP.split_program_response(res.get("content") or "", thinking_mode="direct", generation=None)
        parsed["gen_ok"] = parsed.get("status") == "program_ready"
        return parsed
    reason = rec.get("client_error") or f"finish={finish}"
    return {"status": "generation_not_ok", "reason": str(reason)[:200],
            "reasoning_content": res.get("reasoning") or "", "final": res.get("content") or "",
            "code": None, "gen_ok": False}


def exec_program(code, parse_status, demos, test_input, task_deadline):
    demo_results: list = []
    spent = 0.0
    if code is None:
        for _ in demos:
            demo_results.append({"status": parse_status, "error": "no admitted program", "elapsed_s": 0.0})
        return demo_results, {"status": parse_status, "error": "no admitted program"}, 0.0
    for demo in demos:
        remaining = min(EXEC_PER_PROGRAM_S - spent, task_deadline - time.monotonic(), EXEC_PER_GRID_S)
        if remaining <= 0:
            demo_results.append({"status": "budget_interrupted", "elapsed_s": 0.0})
            continue
        r = SP.execute_grid(code, demo["input"], wall_seconds=min(EXEC_PER_GRID_S, remaining), memory_mb=512)
        spent += r.get("elapsed_s", 0.0)
        demo_results.append(r)
    assert len(demo_results) == len(demos)
    remaining = min(EXEC_PER_PROGRAM_S - spent, task_deadline - time.monotonic(), EXEC_PER_GRID_S)
    if remaining <= 0:
        query_result = {"status": "budget_interrupted", "elapsed_s": 0.0}
    else:
        query_result = SP.execute_grid(code, test_input, wall_seconds=remaining, memory_mb=512)
        spent += query_result.get("elapsed_s", 0.0)
    return demo_results, query_result, round(spent, 2)


def save_reply(directory: Path, stem: str, rec, res, parsed, demo_results, query_result, extra) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.reasoning.txt").write_text(res.get("reasoning") or "")
    (directory / f"{stem}.final.txt").write_text(res.get("content") or "")
    if parsed.get("code"):
        (directory / f"{stem}.code.py").write_text(parsed["code"])
    summary = {k: rec.get(k) for k in (
        "phase", "query", "arm", "request", "finish_reason", "usage", "wall_s", "ttft_s",
        "agg_tok_s", "client_error", "cancelled_by_client", "usage_present",
        "content_chars", "reasoning_chars", "status", "events")}
    summary.update({"parse_status": parsed.get("status"), "gen_ok": parsed.get("gen_ok"),
                    "validation_error": parsed.get("validation_error"),
                    "demo_matches": extra.get("demo_matches"),
                    "all_demos_pass": extra.get("all_demos_pass"),
                    "query_legal": extra.get("query_legal"),
                    "exec_seconds": extra.get("exec_seconds"),
                    "messages_sha256": extra.get("messages_sha256"),
                    "prompt_chars": extra.get("prompt_chars")})
    summary["grid"] = query_result.get("grid") if is_legal_grid(query_result.get("grid")) else None
    (directory / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    rec.update({k: summary[k] for k in (
        "parse_status", "gen_ok", "demo_matches", "all_demos_pass", "query_legal", "exec_seconds")})
    rec["grid"] = summary["grid"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working/vprobe")
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--pip-python", default=None)
    ap.add_argument("--session-seconds", type=float, default=5400.0)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    frozen = Path(args.frozen)
    t0_wall = time.time()
    t0_mono = time.monotonic()
    session_deadline = t0_mono + args.session_seconds
    run = {"config": {"tp": P.TP_SIZE, "max_model_len": P.MAX_MODEL_LEN,
                       "service_max_num_seqs": P.SERVICE_MAX_NUM_SEQS,
                       "gpu_mem_util": P.GPU_MEM_UTIL, "kv_dtype": "default", "mtp": False,
                       "sampling": dict(P.SAMPLING), "effort": EFFORT,
                       "gen_max": GEN_MAX, "think_budget": THINK_BUDGET, "seed": SEED,
                       "vllm_wheel": VLLM_WHEEL_NAME, "vllm_wheel_sha256": VLLM_WHEEL_SHA256,
                       "query": QUERY, "new_call_budget": NEW_CALL_BUDGET,
                       "sunk_completion_tokens": SUNK_COMPLETION_TOKENS},
           "budgets": {"session_s": args.session_seconds, "save_margin_s": SAVE_MARGIN_S,
                       "new_cutoff_s": NEW_REQUEST_CUTOFF_S},
           "calls": [], "_t0": t0_wall}
    STATE["work"] = work
    STATE["run"] = run
    signal.signal(signal.SIGTERM, on_term)
    try:
        lock_fd = os.open(work / "driver.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, str(os.getpid()).encode())
    except FileExistsError:
        print("[018a] lock busy, another driver holds it; exiting quietly", flush=True)
        return 0
    log(f"driver start session_s={args.session_seconds} frozen_dir={frozen}")
    P.record_versions(run)
    log("versions recorded")
    STATE["phase"] = "frozen"
    wd = threading.Thread(target=watchdog, args=(work, 60.0), daemon=True)
    wd.start()

    for name, want in FROZEN_SHA.items():
        # v5: decode embedded blob, verify SHA in memory; no working-volume read.
        # (v1-v4 proved reads from /kaggle/working can stall 89min; the driver
        # source itself imports fine, so embedded bytes are the safe channel.)
        import base64 as _b64
        raw = _b64.b64decode(FROZEN_BLOBS[name])
        got = hashlib.sha256(raw).hexdigest()
        if got != want:
            run["frozen_mismatch"] = {"file": name, "got": got, "want": want}
            return finish("frozen_mismatch", detail=name)
        if name.endswith(".final.txt"):
            frozen_final = raw.decode()
        elif name.endswith(".code.py"):
            frozen_code = raw.decode()
        else:
            task = json.loads(raw.decode())
    run["frozen_sha_ok"] = True
    run["frozen_source"] = "embedded_in_driver_v5"
    save(work, run)
    log("frozen SHA match 018 (embedded)")
    demos = task["train"]
    test_input = task["test"][0]["input"]

    py = sys.version_info
    if (py.major, py.minor) != (3, 12):
        return finish(f"env_incompatible_python_{py.major}_{py.minor}")

    try:
        model_root = P.one("/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/config.json").parent
        wh = P.one("/kaggle/input/**/qwen38-vllm0272-cu130-wheelhouse-v1/**/requirements-qwen38-runtime.txt").parent
    except FileNotFoundError as exc:
        run["mount_error"] = str(exc)[:200]
        return finish("mount_missing")
    run["mounts"] = {"model_root": str(model_root), "wheelhouse": str(wh)}

    site = work / "site"
    site.mkdir(exist_ok=True)
    wheels = wh / "wheels"
    req = wh / "requirements-qwen38-runtime.txt"
    wheel_path = wheels / VLLM_WHEEL_NAME
    if not wheels.is_dir() or not req.is_file() or not wheel_path.is_file():
        return finish("vllm_wheel_missing")
    if sha256_file(wheel_path) != VLLM_WHEEL_SHA256:
        return finish("vllm_wheel_hash_mismatch")
    pip_probe = sh([sys.executable, "-m", "pip", "--version"], timeout=60)
    if pip_probe.returncode:
        ens = sh([sys.executable, "-m", "ensurepip", "--default-pip"], timeout=180)
        (work / "ensurepip.log").write_text((ens.stdout + "\n---STDERR---\n" + ens.stderr)[-4000:])
        pip_probe = sh([sys.executable, "-m", "pip", "--version"], timeout=60)
    use_fallback = pip_probe.returncode != 0
    if use_fallback and not args.pip_python:
        return finish("no_pip_for_312_and_no_fallback")

    def _base():
        if use_fallback:
            return [args.pip_python, "-m", "pip", "install", "--python-version", "3.12",
                    "--only-binary", ":all:", "--no-index"]
        return [sys.executable, "-m", "pip", "install", "--no-index"]
    log(f"installer mode: {'fallback' if use_fallback else 'py312 pip'}")
    STATE["phase"] = "pip-req"
    try:
        log("phase: requirements install start")
        ins = sh(_base() + ["--find-links", str(wheels), "--requirement", str(req),
                            "--target", str(site)], timeout=INSTALL_BUDGET_S)
        log(f"phase: requirements install rc={ins.returncode}")
    except subprocess.TimeoutExpired:
        return finish("install_timeout_over_20min")
    (work / "pip-install.log").write_text(ins.stdout[-8000:] + "\n---STDERR---\n" + ins.stderr[-4000:])
    run["pip_requirements_rc"] = ins.returncode
    save(work, run)
    if ins.returncode:
        return finish("install_failed", detail=(ins.stderr or ins.stdout)[-400:])
    try:
        log("phase: vllm wheel install start")
        vins = sh(_base() + ["--no-deps", "--find-links", str(wheels), str(wheel_path),
                             "--target", str(site)], timeout=600)
        log(f"phase: vllm wheel install rc={vins.returncode}")
    except subprocess.TimeoutExpired:
        return finish("vllm_wheel_install_timeout")
    (work / "pip-vllm.log").write_text(vins.stdout[-4000:] + "\n---STDERR---\n" + vins.stderr[-4000:])
    run["pip_wheel_rc"] = vins.returncode
    save(work, run)
    STATE["phase"] = "import"
    if vins.returncode:
        return finish("vllm_wheel_install_failed")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
    log("phase: import check start")
    imp = sh([sys.executable, "-c", "import vllm, torch; print(vllm.__version__)"], timeout=300,
             env=dict(os.environ))
    log(f"phase: import check rc={imp.returncode}")
    run["import_check"] = {"rc": imp.returncode, "out": (imp.stdout + imp.stderr)[-500:]}
    save(work, run)
    if imp.returncode:
        return finish("import_vllm_failed")
    run["install_s"] = round(time.time() - t0_wall, 1)
    STATE["phase"] = "template"

    tpl = sh([sys.executable, "run_reasoning_probe.py", "--skip-serve", "--work", str(work / "tpl")],
             timeout=300, cwd=str(Path(__file__).resolve().parent), env=dict(os.environ))
    try:
        tpl_run = json.loads((work / "tpl" / "run.json").read_text())
    except Exception as exc:
        return finish("template_check_failed", detail=f"{type(exc).__name__}:{exc}")
    run["template_render"] = tpl_run.get("template_render")
    save(work, run)
    if tpl_run.get("template_render", {}).get("status") != "d differ":
        return finish("template_efforts_identical")

    install_deadline = (session_deadline - args.session_seconds) + INSTALL_BUDGET_S
    if time.monotonic() >= install_deadline:
        return finish("install_load_over_20min")
    try:
        proc = P.start_server(model_root, work, extra=[], deadline=install_deadline)
    except Exception as exc:
        run["serve_tail"] = server_tail(work)
        return finish("serve_failed", detail=f"{type(exc).__name__}:{exc}")
    STATE["proc"] = proc
    run["install_load_s"] = round(time.time() - t0_wall, 1)
    save(work, run)
    STATE["phase"] = "preflight"

    pre = SP.execute_grid("def transform(grid):\n    return [row[:] for row in grid]", [[1]],
                          wall_seconds=2.0, memory_mb=512)
    run["executor_preflight"] = {"status": pre.get("status"), "grid": pre.get("grid")}
    save(work, run)
    if pre.get("status") != "ok" or pre.get("grid") != [[1]]:
        return finish("executor_infra_failed")

    th = threading.Thread(target=gpu_sampler, args=(work, 5.0), daemon=True)
    th.start()

    # ---- 冻结程序真实执行 → 反馈（零模型调用） ----
    task_deadline = session_deadline - SAVE_MARGIN_S
    frozen_demo, frozen_query, frozen_spent = exec_program(
        frozen_code, "program_ready", demos, test_input, task_deadline)
    assert len(frozen_demo) == 4, "feedback must cover all 4 demos"
    try:
        feedback = SP.demo_feedback(demos, frozen_demo, FEEDBACK_MAX_CHARS)
    except Exception as exc:
        return finish("feedback_build_failed", detail=f"{type(exc).__name__}:{exc}")
    frozen_matches = sum(1 for d, r in zip(demos, frozen_demo)
                         if r.get("status") == "ok" and r.get("grid") == d["output"])
    run["frozen_rerun"] = {"demo_matches": frozen_matches,
                           "query_legal": is_legal_grid(frozen_query.get("grid")),
                           "feedback_chars": len(feedback),
                           "exec_seconds": frozen_spent}
    (work / "frozen-feedback.json").write_text(feedback)
    save(work, run)
    log(f"frozen rerun demo_matches={frozen_matches} feedback_chars={len(feedback)}")
    STATE["phase"] = "calls"

    # ---- 2次模型调用 ----
    _, _, base_messages = build_base_prompt(task)
    fix_text = (f"Your previous program:\n```python\n{frozen_final}\n```\n"
                f"Execution feedback (JSON, real demo results):\n{feedback}\n"
                f"Write the corrected complete transform(grid) as exactly one closed ```python block.")
    f_messages = base_messages + [{"role": "user", "content": fix_text}]
    for label, msgs in (("F-fix", f_messages), ("R2", base_messages)):
        if sum(len(m.get("content", "")) for m in msgs) > 100000:
            return finish("prompt_too_long", detail=label)
    if past_cutoff(t0_mono) or not server_alive():
        return finish("cutoff_before_calls")
    limit = min(time.monotonic() + 1500, session_deadline - SAVE_MARGIN_S)
    specs = [{"messages": f_messages, "effort": EFFORT, "seed": SEED, "max_tokens": GEN_MAX,
              "timeout_s": max(1.0, limit - time.monotonic()), "deadline": limit,
              "thinking_token_budget": THINK_BUDGET, "enable_thinking": True,
              "request_id": "F-fix-291dc1e1_0", "query": QUERY, "arm": "F-fix"},
             {"messages": base_messages, "effort": EFFORT, "seed": SEED, "max_tokens": GEN_MAX,
              "timeout_s": max(1.0, limit - time.monotonic()), "deadline": limit,
              "thinking_token_budget": THINK_BUDGET, "enable_thinking": True,
              "request_id": "R2-291dc1e1_0", "query": QUERY, "arm": "R2"}]
    w = P.chat_wave(specs, client_concurrency=8)
    for spec, slot in zip(specs, w["calls"]):
        rec, res = slot["rec"], slot["res"]
        rec["phase"] = "A291"
        rec["query"] = spec["query"]
        rec["arm"] = spec["arm"]
        parsed = parse_generation(rec, res)
        demo_results, query_result, spent = exec_program(
            parsed.get("code") if parsed.get("gen_ok") else None,
            parsed.get("status"), demos, test_input, task_deadline)
        matches = sum(1 for d, r in zip(demos, demo_results)
                      if r.get("status") == "ok" and r.get("grid") == d["output"])
        extra = {"demo_matches": matches, "all_demos_pass": matches == 4,
                 "query_legal": is_legal_grid(query_result.get("grid")),
                 "exec_seconds": spent,
                 "messages_sha256": hashlib.sha256(
                     json.dumps(spec["messages"], ensure_ascii=False).encode()).hexdigest(),
                 "prompt_chars": sum(len(m.get("content", "")) for m in spec["messages"])}
        save_reply(work / "A291", spec["request_id"], rec, res, parsed,
                   demo_results, query_result, extra)
        run["calls"].append(rec)
    save(work, run)

    new_tok = sum(((c.get("usage") or {}).get("completion_tokens") or 0) for c in run["calls"])
    run["ledger"] = {"sunk_completion_tokens": SUNK_COMPLETION_TOKENS,
                     "new_completion_tokens": new_tok,
                     "new_budget": NEW_CALL_BUDGET,
                     "over_budget": new_tok > NEW_CALL_BUDGET,
                     "total_with_sunk": SUNK_COMPLETION_TOKENS + new_tok}
    full = (work / "vllm-server.log").read_text(errors="replace") if exists(work / "vllm-server.log") else ""
    run["preempt_total"] = count_preempt(full)
    save(work, run)
    return finish("complete_with_evidence")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        if STATE.get("run") is not None:
            STATE["run"]["traceback_tail"] = traceback.format_exc()[-1500:]
        raise SystemExit(finish("unexpected_error", 2, f"{type(exc).__name__}:{exc}"))
