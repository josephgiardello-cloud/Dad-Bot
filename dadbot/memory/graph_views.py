from __future__ import annotations

import copy
import logging
from typing import Any


logger = logging.getLogger(__name__)


def build_graph_projection(manager, turn_context=None):
    temporal = manager._require_turn_temporal(turn_context)
    enforce_temporal_window = turn_context is not None
    cache_key = (
        int(getattr(manager._bot, "_memory_graph_generation", 0) or 0),
        str(getattr(temporal, "wall_time", "") if enforce_temporal_window else ""),
        str(getattr(temporal, "wall_date", "") if enforce_temporal_window else ""),
    )
    cached = manager._projection_cache.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    node_map = {}
    edge_map = {}
    manager._project_consolidated_memories(
        node_map,
        edge_map,
        turn_context=turn_context,
    )
    manager._project_archive_sessions(node_map, edge_map, turn_context=turn_context)
    manager._project_persona_traits(node_map, edge_map, turn_context=turn_context)
    manager._project_life_patterns(node_map, edge_map, turn_context=turn_context)
    current_time_str = str(getattr(temporal, "wall_time", "")) if enforce_temporal_window else ""
    nodes = sorted(
        node_map.values(),
        key=lambda item: (item["node_type"], item["label"], item["node_key"]),
    )
    node_keys = {item.get("node_key") for item in nodes}
    edges = sorted(
        (
            e
            for e in edge_map.values()
            if (
                manager.is_edge_valid(e, current_time_str)
                if current_time_str
                else manager.is_edge_valid(e, str(e.get("updated_at") or ""))
            )
            and e.get("source_key") in node_keys
            and e.get("target_key") in node_keys
        ),
        key=lambda item: (
            item["source_key"],
            item["relation_type"],
            item["target_key"],
        ),
    )
    for edge in edges:
        if not edge.get("valid_from"):
            edge["valid_from"] = edge.get("updated_at") or current_time_str
    updated_at = None
    for item in [*nodes, *edges]:
        value = item.get("updated_at")
        if value and (updated_at is None or str(value) > str(updated_at)):
            updated_at = value
    snapshot = {"nodes": nodes, "edges": edges, "updated_at": updated_at}
    manager._projection_cache[cache_key] = copy.deepcopy(snapshot)
    return snapshot


def preview_memory_graph(manager, snapshot=None):
    graph = snapshot or graph_snapshot(manager)
    nodes_by_key = {node["node_key"]: node for node in graph.get("nodes", [])}
    semantic_weights = {}
    semantic_types = {}
    source_neighbors = {}

    for edge in graph.get("edges", []):
        source = nodes_by_key.get(edge.get("source_key"))
        target = nodes_by_key.get(edge.get("target_key"))
        if source is None or target is None:
            continue
        if source.get("node_type") not in manager.GRAPH_SOURCE_NODE_TYPES:
            continue
        if target.get("node_type") == "contradiction":
            continue
        label = str(target.get("label") or "").strip().lower()
        if not label:
            continue
        semantic_weights[label] = semantic_weights.get(label, 0.0) + float(
            edge.get("weight", 0.0) or 0.0,
        )
        semantic_types[label] = target.get("node_type")
        source_neighbors.setdefault(source["node_key"], []).append(label)

    edge_weights = {}
    for labels in source_neighbors.values():
        unique = []
        for label in labels:
            if label not in unique:
                unique.append(label)
        for index, left in enumerate(unique):
            for right in unique[index + 1 :]:
                edge_key = tuple(sorted((left, right)))
                edge_weights[edge_key] = edge_weights.get(edge_key, 0.0) + 1.0

    preview_nodes = []
    for label, weight in sorted(
        semantic_weights.items(),
        key=lambda item: (-item[1], item[0]),
    )[:18]:
        node_type = semantic_types.get(label, "topic")
        preview_type = node_type if node_type in {"topic", "category", "mood"} else "topic"
        preview_nodes.append(
            {
                "id": f"{preview_type}:{label}",
                "label": label,
                "type": preview_type,
                "weight": max(1, int(round(weight))),
            },
        )

    preview_edges = [
        {"source": left, "target": right, "weight": max(1, int(round(weight)))}
        for (left, right), weight in sorted(
            edge_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )[:18]
    ]
    return {
        "nodes": preview_nodes,
        "edges": preview_edges,
        "updated_at": graph.get("updated_at"),
    }


def sync_graph_store(manager, turn_context=None):
    runtime = getattr(manager, "_bot", None)
    commit_active = bool(getattr(runtime, "_graph_commit_active", False))
    if turn_context is not None:
        manager._require_turn_temporal(turn_context)
        active_stage = (
            str(
                (getattr(turn_context, "state", None) or {}).get(
                    "_active_graph_stage",
                )
                or "",
            )
            .strip()
            .lower()
        )
        if not commit_active or active_stage not in {"save", ""}:
            raise RuntimeError(
                "Graph sync violation: graph writes are only allowed at SaveNode commit boundary "
                f"(active_stage={active_stage!r}, commit_active={commit_active!r}).",
            )
    snapshot = build_graph_projection(manager, turn_context=turn_context)
    try:
        manager.ensure_graph_store()
        manager._graph_store_backend.replace_graph(
            snapshot.get("nodes", []),
            snapshot.get("edges", []),
        )
    except Exception as exc:
        logger.warning("Graph store sync failed: %s", exc)
    return snapshot


def graph_snapshot(manager):
    try:
        manager.ensure_graph_store()
        snapshot = manager._graph_store_backend.fetch_graph()
        if snapshot.get("nodes") or snapshot.get("edges"):
            if not snapshot.get("updated_at"):
                updated_at = None
                for item in [
                    *list(snapshot.get("nodes") or []),
                    *list(snapshot.get("edges") or []),
                ]:
                    if not isinstance(item, dict):
                        continue
                    value = item.get("updated_at")
                    if value and (updated_at is None or str(value) > str(updated_at)):
                        updated_at = value
                snapshot = dict(snapshot)
                snapshot["updated_at"] = updated_at
            return snapshot
    except Exception as exc:
        logger.warning(
            "Graph store fetch failed, using in-memory projection: %s",
            exc,
        )
    try:
        return build_graph_projection(manager, turn_context=None)
    except Exception as exc:
        logger.warning("In-memory graph projection failed: %s", exc)
        return {"nodes": [], "edges": [], "updated_at": None}


def _score_graph_node(
    manager,
    node,
    adjacency,
    nodes,
    query_tokens,
    query_category,
    query_mood,
    mood_trend,
    recent_topics,
):
    attributes = dict(node.get("attributes") or {})
    summary = manager.graph_source_summary(node)
    text = " ".join(
        part
        for part in [
            node.get("label", ""),
            node.get("content", ""),
            summary,
            " ".join(attributes.get("supporting_summaries", [])),
            " ".join(attributes.get("contradictions", [])),
        ]
        if part
    )
    source_tokens = manager._bot.significant_tokens(text)
    overlap = len(query_tokens & source_tokens)
    if query_category != "general" and str(node.get("category") or "general").strip().lower() == query_category:
        overlap += 2
    if query_mood != "neutral" and manager._bot.normalize_mood(node.get("mood")) == query_mood:
        overlap += 1
    if mood_trend != "neutral" and manager._bot.normalize_mood(node.get("mood")) == mood_trend:
        overlap += 0.5
    matched_labels = []
    for edge in adjacency.get(node["node_key"], []):
        neighbor_key = (
            edge.get("target_key") if edge.get("source_key") == node["node_key"] else edge.get("source_key")
        )
        neighbor = nodes.get(neighbor_key)
        if (
            neighbor is None
            or neighbor.get("node_type") in manager.GRAPH_SOURCE_NODE_TYPES
            or neighbor.get("node_type") == "contradiction"
        ):
            continue
        label = str(neighbor.get("label") or "").strip().lower()
        if not label:
            continue
        if (
            label in query_tokens
            or (query_category != "general" and label == query_category)
            or label in recent_topics
        ):
            matched_labels.append(label)
            overlap += 1.1
        elif query_mood != "neutral" and label == query_mood:
            matched_labels.append(label)
            overlap += 0.8
    if overlap <= 0 and query_tokens:
        if not any(token in text.lower() for token in query_tokens):
            return None
        overlap = 0.75
    freshness = manager._bot.memory_freshness_weight(node.get("updated_at"))
    confidence = max(
        0.4,
        min(1.15, float(node.get("confidence", 0.6) or 0.6) + 0.15),
    )
    contradictions = len(attributes.get("contradictions", []))
    contradiction_penalty = max(0.5, 1.0 - contradictions * 0.12)
    score = (
        overlap
        * freshness
        * confidence
        * contradiction_penalty
        * manager._RETRIEVAL_SOURCE_TYPE_WEIGHTS.get(node.get("node_type"), 1.0)
    )
    if score <= 0.15:
        return None
    return round(score, 4), sorted(set(matched_labels))


def _rank_graph_summary_nodes(manager, nodes, adjacency):
    ranked = []
    for node in nodes.values():
        if node.get("node_type") in manager.GRAPH_SOURCE_NODE_TYPES or node.get("node_type") == "contradiction":
            continue
        source_neighbors = []
        companion_labels = []
        total_weight = 0.0
        for edge in adjacency.get(node["node_key"], []):
            neighbor_key = (
                edge.get("target_key") if edge.get("source_key") == node["node_key"] else edge.get("source_key")
            )
            neighbor = nodes.get(neighbor_key)
            if neighbor is None:
                continue
            if neighbor.get("node_type") in manager.GRAPH_SOURCE_NODE_TYPES:
                source_neighbors.append(neighbor)
                total_weight += float(edge.get("weight", 0.0) or 0.0) * max(
                    0.4,
                    float(neighbor.get("confidence", 0.0) or 0.0),
                )
                for peer_edge in adjacency.get(neighbor["node_key"], []):
                    peer_key = (
                        peer_edge.get("target_key")
                        if peer_edge.get("source_key") == neighbor["node_key"]
                        else peer_edge.get("source_key")
                    )
                    peer = nodes.get(peer_key)
                    if (
                        peer is None
                        or peer.get("node_key") == node["node_key"]
                        or peer.get("node_type") in manager.GRAPH_SOURCE_NODE_TYPES
                        or peer.get("node_type") == "contradiction"
                    ):
                        continue
                    companion_labels.append(
                        str(peer.get("label") or "").strip().lower(),
                    )
        if not source_neighbors:
            continue
        source_types = sorted(
            {neighbor.get("node_type") for neighbor in source_neighbors if neighbor.get("node_type")},
        )
        ranked.append(
            (round(total_weight, 4), node, source_types, companion_labels),
        )
    return ranked


def _format_graph_summary_lines(manager, ranked, limit):
    source_labels = {
        "archive_session": "archived chats",
        "consolidated_memory": "consolidated insights",
        "persona_trait": "dad traits",
        "life_pattern": "life patterns",
    }
    lines = []
    for _score, node, source_types, companion_labels in sorted(
        ranked,
        key=lambda item: (item[0], item[1].get("label", "")),
        reverse=True,
    )[: max(1, int(limit or 3))]:
        sources_text = manager._bot.natural_list(
            [source_labels.get(name, name.replace("_", " ")) for name in source_types],
        )
        companion_unique = []
        for label in companion_labels:
            if label and label not in companion_unique:
                companion_unique.append(label)
        line = f"- {str(node.get('label', '')).title()} is reinforced across {sources_text}"
        if companion_unique:
            line += f", often alongside {manager._bot.natural_list([label.title() for label in companion_unique[:2]])}"
        line += "."
        lines.append(line)
    return lines


def graph_retrieval_for_input(manager, query, limit=3):
    snapshot = graph_snapshot(manager)
    nodes = {node["node_key"]: node for node in snapshot.get("nodes", [])}
    if not nodes:
        return None

    adjacency = {}
    for edge in snapshot.get("edges", []):
        adjacency.setdefault(edge.get("source_key"), []).append(edge)
        adjacency.setdefault(edge.get("target_key"), []).append(edge)

    query_tokens = manager._bot.significant_tokens(query)
    query_category = manager._bot.infer_memory_category(query)
    query_mood = manager._bot.normalize_mood(query)
    recent_topics = set(manager._bot.recent_memory_topics(limit=4))
    mood_trend = manager._bot.current_memory_mood_trend()

    ranked = []
    for node in nodes.values():
        if node.get("node_type") not in manager.GRAPH_SOURCE_NODE_TYPES:
            continue
        result = _score_graph_node(
            manager,
            node,
            adjacency,
            nodes,
            query_tokens,
            query_category,
            query_mood,
            mood_trend,
            recent_topics,
        )
        if result is None:
            continue
        score, matched_labels = result
        ranked.append((score, node, matched_labels))

    if not ranked:
        return None

    selected = sorted(
        ranked,
        key=lambda item: (
            item[0],
            item[1].get("updated_at", ""),
            item[1].get("label", ""),
        ),
        reverse=True,
    )[: max(1, int(limit or 3))]
    return {
        "compressed_summary": manager._graph_prompt_compressor.compress_neighborhood(
            query,
            snapshot.get("nodes", []),
            snapshot.get("edges", []),
            max_tokens=manager._bot.runtime_config.graph_context_token_budget,
        ),
        "summary_lines": manager.graph_source_lines(selected),
        "supporting_evidence": [
            {
                "source_type": node.get("node_type"),
                "label": node.get("label", ""),
                "summary": manager.graph_source_summary(node),
                "category": str(node.get("category") or "general").strip().lower() or "general",
                "mood": manager._bot.normalize_mood(node.get("mood")),
                "confidence": round(float(node.get("confidence", 0.0) or 0.0), 2),
                "updated_at": node.get("updated_at"),
                "score": score,
                "matched_nodes": matched_labels,
                "contradictions": list(
                    (node.get("attributes") or {}).get("contradictions", []),
                )[:2],
            }
            for score, node, matched_labels in selected
        ],
        "updated_at": snapshot.get("updated_at"),
    }


def build_graph_summary_context(manager, limit=3):
    snapshot = graph_snapshot(manager)
    nodes = {node["node_key"]: node for node in snapshot.get("nodes", [])}
    if not nodes:
        return None

    compressed = manager._graph_prompt_compressor.compress_neighborhood(
        "long-term relationship state with Tony",
        snapshot.get("nodes", []),
        snapshot.get("edges", []),
        max_tokens=min(260, manager._bot.runtime_config.graph_context_token_budget),
    )
    if compressed:
        return "Relationship graph summary:\n" + compressed

    adjacency = {}
    for edge in snapshot.get("edges", []):
        adjacency.setdefault(edge.get("source_key"), []).append(edge)
        adjacency.setdefault(edge.get("target_key"), []).append(edge)

    ranked = _rank_graph_summary_nodes(manager, nodes, adjacency)
    if not ranked:
        return None

    lines = _format_graph_summary_lines(manager, ranked, limit)
    if not lines:
        return None
    return "Relationship graph summary:\n" + "\n".join(lines)