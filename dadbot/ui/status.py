"""Status dashboard helpers for Streamlit surfaces."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_confluence_status_card(confluence: dict[str, Any] | None) -> None:
	"""Render the confluence law operator panel in the status dashboard."""
	payload = dict(confluence or {})
	with st.container(border=True):
		st.subheader("Confluence Law")
		conf_col1, conf_col2, conf_col3, conf_col4 = st.columns(4)
		conf_mode = str(payload.get("mode") or "off").strip().lower() or "off"
		conf_col1.metric("Mode", conf_mode.upper())
		conf_col2.metric("Strict Legacy", "ON" if bool(payload.get("strict_legacy_disabled", False)) else "OFF")
		conf_col3.metric("Enforcement Rate", f"{int(float(payload.get('enforcement_rate', 0.0) or 0.0) * 100)}%")
		conf_col4.metric("Blocked Turns", int(payload.get("enforced_blocked", 0) or 0))

		conf_stats1, conf_stats2, conf_stats3, conf_stats4 = st.columns(4)
		conf_stats1.metric("Attempted", int(payload.get("attempted", 0) or 0))
		conf_stats2.metric("Matched", int(payload.get("matched", 0) or 0))
		conf_stats3.metric("Mismatches", int(payload.get("mismatch", 0) or 0))
		conf_stats4.metric("First Bind", int(payload.get("bound_first_observation", 0) or 0))

		last_action = str(payload.get("action") or "none")
		last_key = str(payload.get("key") or "")
		if last_key:
			st.caption(f"Last action: {last_action} on key {last_key}")
		else:
			st.caption(f"Last action: {last_action}")


__all__ = ["render_confluence_status_card"]
