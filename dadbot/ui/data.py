"""Data and memory surface renderers."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date as _date_cls, datetime
from typing import TYPE_CHECKING

import streamlit as st

from dadbot.heritage_import import build_heritage_memories
from dadbot.ui.helpers import export_bundle_payload, fetch_ical_events, heritage_files_with_limits
from dadbot.ui.utils import filter_memory_entries, maybe_fragment

if TYPE_CHECKING:
    from dadbot.core.dadbot import DadBot

__all__ = ["render_data_tab", "render_memory_garden"]


@maybe_fragment
def render_memory_garden(bot: "DadBot") -> None:
    with st.container(border=True):
        st.subheader("Memory Garden")
        st.caption("A calmer view of Dad's memories. Search by phrase, then browse like a notebook.")
        all_memories = bot.memory_catalog()
        memory_categories = sorted(set(str(item.get("category") or "general") for item in all_memories))
        garden_col1, garden_col2, garden_col3 = st.columns([3, 2, 1])
        with garden_col1:
            garden_search = st.text_input("Search memories", placeholder="e.g., work, sleep, family", key="garden-search", label_visibility="collapsed")
        with garden_col2:
            garden_category = st.selectbox("Category", options=["all"] + memory_categories, key="garden-category", label_visibility="collapsed")
        with garden_col3:
            chapters_view = st.checkbox("Chapters", value=False, key="garden-chapters", help="Group memories by Life Chapter (category)")

        filtered_memories = filter_memory_entries(all_memories, garden_search, garden_category)
        display_limit = 30
        st.caption(f"Showing {min(len(filtered_memories), display_limit)} of {len(all_memories)} memory notes")

        def _render_memory_card(memory_entry, card_key_suffix):
            memory_category = str(memory_entry.get("category") or "general").replace("_", " ").title()
            memory_summary = str(memory_entry.get("summary") or "")
            memory_importance = float(memory_entry.get("importance_score") or memory_entry.get("importance", 0.5) or 0.5)
            memory_date_raw = str(memory_entry.get("created_at") or memory_entry.get("updated_at") or "")[:10]
            try:
                memory_age_days = (_date_cls.today() - _date_cls.fromisoformat(memory_date_raw)).days if memory_date_raw else 0
            except Exception:
                memory_age_days = 0
            age_fade = max(0.15, 1.0 - min(1.0, memory_age_days / 180.0))
            if memory_age_days > 120:
                age_label, age_color = "aged", "#a08060"
            elif memory_age_days > 60:
                age_label, age_color = "maturing", "#7a9070"
            elif memory_age_days > 14:
                age_label, age_color = "settling", "#5a8a9a"
            else:
                age_label, age_color = "fresh", "#3a8a5a"
            pinned = bool(memory_entry.get("pinned", False))
            with st.container(border=True):
                pin_badge = " ⭐" if pinned else ""
                st.markdown(
                    f"<span style='opacity:{age_fade:.2f}'>"
                    f"<strong>[{memory_category}]</strong> {memory_summary}{pin_badge}</span>",
                    unsafe_allow_html=True,
                )
                mc1, mc2, mc3, mc4 = st.columns([2, 2, 2, 1])
                mc1.caption(f"Importance: {memory_importance:.2f}")
                mc2.caption(f"Added: {memory_date_raw or 'unknown'}")
                mc3.markdown(
                    f"<span style='color:{age_color};font-size:0.82rem;'>● {age_label} ({memory_age_days}d)</span>",
                    unsafe_allow_html=True,
                )
                star_label = "★ Unstar" if pinned else "☆ Star"
                if mc4.button(star_label, key=f"pin-{card_key_suffix}-{hash(memory_summary)}", use_container_width=True):
                    catalog = bot.memory_catalog()
                    norm_summary = bot.normalize_memory_text(memory_summary)
                    for catalog_entry in catalog:
                        if bot.normalize_memory_text(catalog_entry.get("summary", "")) == norm_summary:
                            catalog_entry["pinned"] = not pinned
                            break
                    bot.save_memory_catalog(catalog)
                    st.rerun()

        if chapters_view:
            chapter_groups: dict[str, list] = defaultdict(list)
            for memory_entry in filtered_memories[:display_limit]:
                chapter_groups[str(memory_entry.get("category") or "general")].append(memory_entry)
            for chapter_category in sorted(chapter_groups.keys()):
                chapter_memories = chapter_groups[chapter_category]
                chapter_label = chapter_category.replace("_", " ").title()
                starred_count = sum(1 for entry in chapter_memories if entry.get("pinned"))
                chapter_title = f"📖 {chapter_label}  ({len(chapter_memories)} memories{f', {starred_count} ⭐' if starred_count else ''})"
                with st.expander(chapter_title, expanded=False):
                    for chapter_idx, memory_entry in enumerate(chapter_memories):
                        _render_memory_card(memory_entry, f"chap-{chapter_category}-{chapter_idx}")
        else:
            for memory_idx, memory_entry in enumerate(filtered_memories[:display_limit]):
                _render_memory_card(memory_entry, str(memory_idx))

        if len(filtered_memories) > display_limit:
            st.caption(f"...and {len(filtered_memories) - display_limit} more. Refine your search to see specific entries.")


def render_data_tab(bot: "DadBot") -> None:
    bundle_payload = export_bundle_payload(bot)
    memory_payload = json.dumps(bundle_payload["memory_store"], indent=2)
    profile_payload = json.dumps(bot.PROFILE, indent=2)
    bundle_json = json.dumps(bundle_payload, indent=2)
    living = bot.profile_runtime.living_dad_snapshot(limit=3)
    col1, col2, col3 = st.columns(3)
    col1.metric("Saved memories", len(bot.memory_catalog()))
    col2.metric("Archived sessions", len(bot.session_archive()))
    col3.metric("Reminders", len(bot.reminder_catalog()))

    with st.container(border=True):
        st.subheader("Calendar feed")
        st.caption("Connect a public iCal/ICS feed (Google Calendar, Apple Calendar, Outlook) so Dad sees your upcoming events.")
        _ical_url = str((bot.PROFILE.get("ical_feed_url") if isinstance(bot.PROFILE, dict) else None) or "").strip()
        _ical_input = st.text_input(
            "Public iCal feed URL (.ics)",
            value=_ical_url,
            placeholder="https://calendar.google.com/calendar/ical/.../@public/basic.ics",
            help="Paste the public ICS link. In Google Calendar: Settings → your calendar → Integrate calendar → Public address in iCal format.",
        )
        _ical_save_col, _ical_fetch_col = st.columns(2)
        if _ical_save_col.button("Save feed URL", use_container_width=True):
            if isinstance(bot.PROFILE, dict):
                bot.PROFILE["ical_feed_url"] = str(_ical_input or "").strip()
                bot.save_profile()
                st.success("Calendar feed URL saved.")
                st.rerun()
        if _ical_fetch_col.button("Preview upcoming events", use_container_width=True, disabled=not str(_ical_input or "").strip()):
            with st.spinner("Fetching calendar feed..."):
                _events, _err = fetch_ical_events(str(_ical_input).strip(), max_events=12)
            if _err:
                st.error(_err)
            elif not _events:
                st.info("No upcoming events found in this feed.")
            else:
                st.success(f"Found {len(_events)} upcoming event(s):")
                for _ev in _events:
                    _ev_date = _ev.get("start") or "?"
                    _ev_end = _ev.get("end") or ""
                    _ev_loc = f" @ {_ev['location']}" if _ev.get("location") else ""
                    _ev_desc = f"  \n_{_ev['description']}_" if _ev.get("description") else ""
                    st.markdown(f"**{_ev_date}{'–'+_ev_end if _ev_end and _ev_end!=_ev_date else ''}** — {_ev['summary']}{_ev_loc}{_ev_desc}")
        if _ical_url:
            st.caption(f"Configured feed: `{_ical_url[:80]}{'...' if len(_ical_url)>80 else ''}`")
            st.caption("Dad will be aware of your upcoming events when this feed is set.")

    with st.container(border=True):
        st.subheader("Build My Heritage")
        st.caption("Upload journals, exports, and photo metadata so Dad can bootstrap long-term memory from your history.")
        uploaded_files = st.file_uploader(
            "Upload heritage sources",
            type=["json", "txt", "md", "csv", "mbox", "jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="You can upload mixed files in one pass. JSON, text, mbox, and photos are supported.",
        )
        onboarding_notes = st.text_area(
            "Onboarding note (optional)",
            value="",
            placeholder="Example: Use these imports to understand my family history and recurring stress triggers.",
            height=90,
        )
        max_items = st.slider("Max extracted memories per file", min_value=5, max_value=120, value=40, step=5)
        run_consolidation = st.checkbox("Run consolidation after import", value=True)
        preview_col, import_col = st.columns(2)
        _validated_uploads, _upload_issues = heritage_files_with_limits(uploaded_files)
        if _upload_issues:
            for _issue in _upload_issues[:4]:
                st.warning(_issue)
        if preview_col.button("Preview heritage import", use_container_width=True, disabled=not _validated_uploads):
            with st.spinner("Parsing uploaded heritage files..."):
                st.session_state.heritage_preview = build_heritage_memories(
                    _validated_uploads,
                    notes=onboarding_notes,
                    max_items_per_file=max_items,
                )
        preview_payload = st.session_state.get("heritage_preview", {})
        preview_memories = list(preview_payload.get("memories", [])) if isinstance(preview_payload, dict) else []
        if preview_memories:
            stats = dict(preview_payload.get("stats", {}))
            st.write(
                {
                    "files": stats.get("files", 0),
                    "extracted_memories": stats.get("memories", 0),
                    "failed_files": stats.get("failed", 0),
                    "file_types": stats.get("by_type", {}),
                }
            )
            st.caption("Preview of extracted memories")
            for entry in preview_memories[:10]:
                st.markdown(f"- [{entry.get('category', 'heritage')}] {entry.get('summary', '')}")
            if len(preview_memories) > 10:
                st.caption(f"...and {len(preview_memories) - 10} more")
        if import_col.button("Import into Dad memory", use_container_width=True, type="primary", disabled=not preview_memories):
            added = 0
            with st.spinner("Importing heritage memories..."):
                for entry in preview_memories:
                    summary = str(entry.get("summary") or "").strip()
                    category = str(entry.get("category") or "heritage").strip().lower() or "heritage"
                    if not summary:
                        continue
                    if bot.add_memory(summary, category=category) is not None:
                        added += 1
                if run_consolidation:
                    bot.consolidate_memories(force=True)
            st.success(f"Imported {added} heritage memories.")
            st.rerun()

    with st.container(border=True):
        st.subheader("Download data")
        st.caption("Pull out the current memory store, profile, or a combined support bundle.")
        download_col1, download_col2, download_col3 = st.columns(3)
        download_col1.download_button("Download memory JSON", data=memory_payload, file_name="dad_memory_export.json", mime="application/json", use_container_width=True)
        download_col2.download_button("Download profile JSON", data=profile_payload, file_name="dad_profile_export.json", mime="application/json", use_container_width=True)
        download_col3.download_button("Download full bundle", data=bundle_json, file_name=f"dadbot_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", use_container_width=True)
        path_col1, path_col2 = st.columns([3, 1])
        export_path = path_col1.text_input("Export memory to workspace path", value=st.session_state.export_path)
        st.session_state.export_path = export_path
        if path_col2.button("Write file", use_container_width=True):
            bot.memory.export_memory_store(export_path)
            st.success(f"Memory store exported to {export_path}.")

    render_memory_garden(bot)

    with st.container(border=True):
        st.subheader("Memory management")
        memories = bot.memory_catalog()
        memory_labels = [memory.get("summary", "") for memory in memories]
        selected_memory = st.selectbox("Delete a saved memory", options=[""] + memory_labels)
        if st.button("Delete selected memory", disabled=not selected_memory):
            removed = bot.delete_memory_entry(selected_memory)
            if removed:
                st.success(f"Removed {len(removed)} memory entry." if len(removed) == 1 else f"Removed {len(removed)} memory entries.")
                st.rerun()
            st.warning("No matching memory entry was removed.")
        confirm_clear = st.checkbox("I understand that clearing memory is destructive.")
        if st.button("Clear all memory", type="primary", disabled=not confirm_clear):
            bot.memory.clear_memory_store()
            st.success("Dad's saved memory store was cleared.")
            st.rerun()
        if st.button("Run controlled forgetting", use_container_width=True):
            result = bot.apply_controlled_forgetting(force=True)
            removed = int(result.get("removed", 0) or 0)
            if removed > 0:
                st.success(f"Forgot {removed} low-importance memories. Backup: {result.get('backup_path', '')}")
                st.rerun()
            st.info("No memories met forgetting criteria.")

    with st.container(border=True):
        st.subheader("Consolidated memory feedback")
        st.caption("Upvote or downvote these to reinforce what Dad should treat as durable truth.")
        consolidated = [entry for entry in bot.consolidated_memories() if not bool(entry.get("superseded", False))]
        if not consolidated:
            st.caption("No consolidated memories yet.")
        else:
            options = [entry.get("summary", "") for entry in consolidated if entry.get("summary")]
            selected_summary = st.selectbox("Choose consolidated memory", options=options, index=0)
            vote_col1, vote_col2 = st.columns(2)
            with vote_col1:
                if st.button("Upvote memory", use_container_width=True):
                    updated = bot.apply_consolidated_feedback(selected_summary, 1)
                    if updated:
                        st.success("Reinforced this memory.")
                        st.rerun()
                    st.warning("Could not find that memory to reinforce.")
            with vote_col2:
                if st.button("Downvote memory", use_container_width=True):
                    updated = bot.apply_consolidated_feedback(selected_summary, -1)
                    if updated:
                        st.info("Weakened this memory signal.")
                        st.rerun()
                    st.warning("Could not find that memory to adjust.")

    with st.container(border=True):
        st.subheader("Conflict Resolution")
        st.caption("Memories where Dad holds conflicting beliefs. Pick which one is true to resolve the contradiction.")
        _contradictions = bot.consolidated_contradictions(limit=6)
        if not _contradictions:
            st.caption("No contradictions detected. Dad's memory is consistent.")
        else:
            for _ci, _pair in enumerate(_contradictions):
                _left = str(_pair.get("left") or _pair.get("a") or "")
                _right = str(_pair.get("right") or _pair.get("b") or "")
                if not _left or not _right:
                    continue
                with st.container(border=True):
                    st.markdown(f"**Conflict #{_ci + 1}**")
                    _rc1, _rc2 = st.columns(2)
                    _rc1.info(f"🅐 {_left}")
                    _rc2.warning(f"🅑 {_right}")
                    _resolution = st.radio(
                        "Which is correct?",
                        options=["Keep A", "Keep B", "Both could be true", "Dismiss both"],
                        horizontal=True,
                        key=f"conflict-{_ci}-{hash(_left)}",
                        label_visibility="collapsed",
                    )
                    if st.button("Resolve", key=f"resolve-btn-{_ci}-{hash(_left)}", use_container_width=False):
                        if _resolution == "Keep A":
                            bot.resolve_consolidated_contradiction(_left, _right, keep="left", reason="user_review")
                        elif _resolution == "Keep B":
                            bot.resolve_consolidated_contradiction(_left, _right, keep="right", reason="user_review")
                        elif _resolution == "Both could be true":
                            bot.resolve_consolidated_contradiction(_left, _right, keep="both", reason="user_review")
                        else:
                            bot.resolve_consolidated_contradiction(_left, _right, keep="none", reason="user_review")
                        st.success("Resolved. Dad will update his understanding.")
                        st.rerun()

    with st.container(border=True):
        st.subheader("Living Dad snapshot")
        st.caption("Long-term signals shaping Dad's current replies.")
        _snap_cols = st.columns(2)
        with _snap_cols[0]:
            _shifts = [entry.get("trait", "") for entry in living["persona_shifts"] if entry.get("trait")]
            st.markdown("**Persona shifts**")
            if _shifts:
                for _t in _shifts:
                    st.markdown(f"- {_t}")
            else:
                st.caption("None yet.")
            _wisdom = [entry.get("summary", "") for entry in living["wisdom"] if entry.get("summary")]
            st.markdown("**Wisdom learned**")
            if _wisdom:
                for _w in _wisdom:
                    st.markdown(f"- {_w}")
            else:
                st.caption("None yet.")
        with _snap_cols[1]:
            _patterns = [entry.get("pattern", entry.get("summary", "")) for entry in living["patterns"] if entry.get("pattern") or entry.get("summary")]
            st.markdown("**Life patterns**")
            if _patterns:
                for _p in _patterns:
                    st.markdown(f"- {_p}")
            else:
                st.caption("None yet.")
            _queued = [entry.get("message", "") for entry in living["proactive_queue"] if entry.get("message")]
            st.markdown("**Queued proactive messages**")
            if _queued:
                for _q in _queued:
                    st.markdown(f"- {_q}")
            else:
                st.caption("None queued.")

    with st.container(border=True):
        st.subheader("📓 Dad's Journal")
        st.caption("Weekly reflections from Dad's point of view, written as short journal pages.")
        _journal_key = "dad_journal_entries"
        _journal_entries = list(bot.MEMORY_STORE.get(_journal_key) or []) if isinstance(bot.MEMORY_STORE, dict) else []
        _this_week = datetime.now().strftime("%Y-W%W")
        _existing_this_week = next((e for e in _journal_entries if e.get("week") == _this_week), None)
        _journal_col1, _journal_col2 = st.columns([3, 1])
        with _journal_col2:
            _force_regen = st.button("Write this week's entry", use_container_width=True)
        if _force_regen or (_existing_this_week is None and len(bot.session_archive()) > 0):
            _recent_sessions = bot.session_archive()[-5:]
            _session_text = "\n\n".join(
                f"[{s.get('started_at', '')[:10]}] Turns: {s.get('turn_count', 0)}, Mood: {s.get('peak_mood', 'neutral')}"
                for s in _recent_sessions
            )
            _journal_prompt = (
                f"You are Dad, writing your personal journal. "
                f"Summarize the past week of conversations with your son Tony in 2-3 warm, reflective paragraphs. "
                f"Mention what you noticed, what you felt proud of, and any worries. "
                f"Here is the session data:\n{_session_text}"
            )
            with st.spinner("Dad is writing in his journal..."):
                try:
                    _journal_resp = bot.call_ollama_chat(
                        [{"role": "user", "content": _journal_prompt}],
                        purpose="journal",
                    )
                    _journal_text = bot.extract_ollama_message_content(_journal_resp) or "(Dad hasn't written this week's entry yet.)"
                    _new_entry = {
                        "week": _this_week,
                        "written_at": datetime.now().isoformat(timespec="seconds"),
                        "text": str(_journal_text or "").strip(),
                    }
                    _journal_entries = [e for e in _journal_entries if e.get("week") != _this_week]
                    _journal_entries.append(_new_entry)
                    _journal_entries = _journal_entries[-12:]
                    bot.mutate_memory_store(**{_journal_key: _journal_entries}, save=True)
                    _existing_this_week = _new_entry
                    st.rerun()
                except Exception as _je:
                    st.warning(f"Could not generate journal entry: {_je}")
        if _journal_entries:
            for _jentry in reversed(_journal_entries[-8:]):
                _jwk = str(_jentry.get("week") or "")
                _jdate = str(_jentry.get("written_at") or "")[:10]
                with st.expander(f"Week {_jwk}  ({_jdate})", expanded=_jwk == _this_week):
                    st.markdown(_jentry.get("text", "_(empty)_"))
        else:
            st.caption("No journal entries yet. Start chatting and click 'Write this week's entry' to generate one.")

    with st.container(border=True):
        st.subheader("Knowledge web")
        st.caption("Visual map of what Dad has learned from long-term memory and relationship patterns.")
        graph = bot.memory.memory_graph_snapshot() or {}
        nodes = list(graph.get("nodes", []))[:24]
        edges = list(graph.get("edges", []))[:36]
        if not nodes:
            st.caption("No graph nodes yet. Keep chatting and run periodic synthesis to build this map.")
        else:
            dot_lines = ["digraph DadKnowledge {", "rankdir=LR;", "node [shape=ellipse, style=filled, fillcolor=lightgoldenrod1, color=gray40];"]
            for node in nodes:
                node_id = str(node.get("id") or f"n-{hash(str(node))}").replace('"', "'")
                label = str(node.get("label") or node_id).replace('"', "'")
                node_type = str(node.get("type") or "topic").lower()
                fill = "lightgoldenrod1"
                if node_type == "mood":
                    fill = "lightblue"
                elif node_type == "category":
                    fill = "honeydew"
                dot_lines.append(f'"{node_id}" [label="{label}", fillcolor="{fill}"];')
            for edge in edges:
                source = str(edge.get("source") or "").replace('"', "'")
                target = str(edge.get("target") or "").replace('"', "'")
                if not source or not target:
                    continue
                weight = int(edge.get("weight", 1) or 1)
                dot_lines.append(f'"{source}" -> "{target}" [label="{weight}"];')
            dot_lines.append("}")
            try:
                st.graphviz_chart("\n".join(dot_lines), use_container_width=True)
            except Exception as exc:
                st.warning(f"Graph rendering unavailable: {exc}")

    with st.container(border=True):
        st.subheader("Memory search")
        st.caption("Inspect what memories Dad can retrieve for a topic before asking for advice.")
        query = st.text_input("Search memories", value="", placeholder="e.g., work stress, sleep, routines")
        strategy = st.selectbox(
            "Retrieval strategy",
            options=["auto", "hybrid", "graph_heavy", "semantic_heavy", "consolidated_heavy"],
            index=0,
        )
        if st.button("Search", use_container_width=True, disabled=not str(query or "").strip()):
            retrieval = bot.retrieve_context(str(query).strip(), strategy=strategy, limit=6)
            st.caption(f"Strategy used: **{retrieval.get('strategy', 'hybrid')}**")
            for item in retrieval.get("bundle", []):
                bundle_type = str(item.get("type") or "unknown")
                score = float(item.get("score", 0.0) or 0.0)
                payload = item.get("payload", {})
                st.markdown(f"**{bundle_type.title()}** (score={score:.2f})")
                if isinstance(payload, dict):
                    if payload.get("summary"):
                        st.caption(str(payload.get("summary")))
                    elif payload.get("compressed_summary"):
                        st.caption(str(payload.get("compressed_summary")))
                    else:
                        st.caption(json.dumps(payload, indent=2)[:280])
                else:
                    st.caption(str(payload))
