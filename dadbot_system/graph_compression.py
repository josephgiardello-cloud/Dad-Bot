from collections import deque


class GraphPromptCompressor:
    def __init__(self, bot, max_tokens=800, max_hops=2, max_edges=18, max_nodes=16):
        self.bot = bot
        self.max_tokens = max(1, int(max_tokens or 800))
        self.max_hops = max(1, int(max_hops or 2))
        self.max_edges = max(4, int(max_edges or 18))
        self.max_nodes = max(4, int(max_nodes or 16))

    def compress_neighborhood(self, query: str, nodes: list, edges: list, max_tokens: int = None) -> str:
        if not nodes and not edges:
            return ""

        budget = max(1, int(max_tokens or self.max_tokens))
        subgraph = self._extract_relevant_subgraph(nodes, edges, query)
        raw_text = self._subgraph_to_text(subgraph)
        if not raw_text:
            return ""

        if self.bot.estimate_token_count(raw_text) <= budget:
            return raw_text

        prompt = f"""
Summarize the following relationship graph evidence in under {budget} tokens.
Focus on patterns, emotions, contradictions, and advice-relevant insights for Tony.
Be concise, specific, and useful for Dad's next reply.
Do not invent any facts that are not present in the evidence.
Return plain text only.

Query: {query}

Graph evidence:
{raw_text}

Compressed summary:
""".strip()
        try:
            response = self.bot.call_ollama_chat(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                purpose="graph_compression",
            )
            compressed = str(response["message"]["content"] or "").strip()
            if compressed:
                if self.bot.estimate_token_count(compressed) <= budget:
                    return compressed
                return self._truncate_to_budget(compressed, budget)
        except Exception:
            pass
        return self._truncate_to_budget(raw_text, budget)

    def _extract_relevant_subgraph(self, nodes, edges, query):
        node_map = {node.get("node_key"): node for node in nodes if isinstance(node, dict) and node.get("node_key")}
        if not node_map:
            return {"nodes": [], "edges": []}

        adjacency = {}
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source_key = edge.get("source_key")
            target_key = edge.get("target_key")
            if source_key not in node_map or target_key not in node_map:
                continue
            adjacency.setdefault(source_key, []).append(edge)
            adjacency.setdefault(target_key, []).append(edge)

        query_tokens = self.bot.significant_tokens(query)
        query_category = self.bot.infer_memory_category(query)
        query_mood = self.bot.normalize_mood(query)

        scored_nodes = []
        for node in node_map.values():
            score = self._node_match_score(node, query_tokens, query_category, query_mood)
            if score > 0:
                scored_nodes.append((score, node.get("node_key")))

        if not scored_nodes:
            fallback_nodes = sorted(
                node_map.values(),
                key=lambda item: (str(item.get("updated_at") or ""), str(item.get("label") or "")),
                reverse=True,
            )[: min(3, len(node_map))]
            scored_nodes = [(1.0, node.get("node_key")) for node in fallback_nodes if node.get("node_key")]

        seeds = [
            node_key
            for _score, node_key in sorted(scored_nodes, key=lambda item: item[0], reverse=True)[:3]
            if node_key
        ]
        visited_nodes = set(seeds)
        visited_edges = []
        seen_edge_keys = set()
        queue = deque((seed, 0) for seed in seeds)

        while queue and len(visited_nodes) < self.max_nodes and len(visited_edges) < self.max_edges:
            node_key, depth = queue.popleft()
            if depth >= self.max_hops:
                continue

            ranked_edges = sorted(
                adjacency.get(node_key, []),
                key=lambda edge: self._edge_priority(edge, node_map, query_tokens, query_category, query_mood),
                reverse=True,
            )
            for edge in ranked_edges:
                edge_key = edge.get("edge_key")
                if edge_key in seen_edge_keys:
                    continue
                other_key = edge.get("target_key") if edge.get("source_key") == node_key else edge.get("source_key")
                if other_key not in node_map:
                    continue
                seen_edge_keys.add(edge_key)
                visited_edges.append(edge)
                if other_key not in visited_nodes and len(visited_nodes) < self.max_nodes:
                    visited_nodes.add(other_key)
                    queue.append((other_key, depth + 1))
                if len(visited_edges) >= self.max_edges:
                    break

        subgraph_nodes = [node_map[node_key] for node_key in visited_nodes if node_key in node_map]
        return {
            "nodes": sorted(
                subgraph_nodes,
                key=lambda item: (item.get("node_type", ""), item.get("label", ""), item.get("node_key", "")),
            ),
            "edges": visited_edges,
        }

    def _node_match_score(self, node, query_tokens, query_category, query_mood):
        label = str(node.get("label") or "").lower()
        content = str(node.get("content") or "").lower()
        text = f"{label} {content}".strip()
        score = len(query_tokens & self.bot.significant_tokens(text))
        if query_category != "general" and str(node.get("category") or "general").strip().lower() == query_category:
            score += 2
        if query_mood != "neutral" and self.bot.normalize_mood(node.get("mood")) == query_mood:
            score += 1
        if not score and query_tokens and any(token in text for token in query_tokens):
            score = 0.75
        return float(score)

    def _edge_priority(self, edge, node_map, query_tokens, query_category, query_mood):
        source = node_map.get(edge.get("source_key"))
        target = node_map.get(edge.get("target_key"))
        source_score = self._node_match_score(source or {}, query_tokens, query_category, query_mood)
        target_score = self._node_match_score(target or {}, query_tokens, query_category, query_mood)
        return (
            float(edge.get("weight", 0.0) or 0.0)
            + source_score
            + target_score
            + float(edge.get("confidence", 0.0) or 0.0)
        )

    def _subgraph_to_text(self, subgraph):
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        if not nodes and not edges:
            return ""

        node_map = {node.get("node_key"): node for node in nodes if node.get("node_key")}
        source_nodes = [
            node
            for node in nodes
            if str(node.get("source_type") or "").strip()
            or node.get("node_type") in {"consolidated_memory", "archive_session", "persona_trait", "life_pattern"}
        ]
        if not source_nodes:
            source_nodes = nodes[:]

        lines = []
        for source in source_nodes:
            summary = self._source_summary(source)
            if not summary:
                continue
            source_type = str(source.get("node_type") or source.get("source_type") or "evidence").replace("_", " ")
            lines.append(f"- [{source_type}] {summary}")
            related = []
            for edge in edges:
                if edge.get("source_key") == source.get("node_key"):
                    neighbor = node_map.get(edge.get("target_key"))
                elif edge.get("target_key") == source.get("node_key"):
                    neighbor = node_map.get(edge.get("source_key"))
                else:
                    continue
                if neighbor is None:
                    continue
                neighbor_label = str(neighbor.get("label") or "").strip()
                if not neighbor_label:
                    continue
                relation = str(edge.get("relation_type") or "related_to").replace("_", " ")
                related.append(f"{relation} {neighbor_label}")
            if related:
                deduped = []
                for item in related:
                    if item not in deduped:
                        deduped.append(item)
                lines.append(f"  links: {'; '.join(deduped[:4])}")
        return "\n".join(lines)

    @staticmethod
    def _source_summary(node):
        attributes = dict(node.get("attributes") or {})
        return str(
            attributes.get("summary") or attributes.get("reason") or node.get("content") or node.get("label") or ""
        ).strip()

    def _truncate_to_budget(self, text, token_budget):
        if not text:
            return ""
        target_chars = max(1, int(token_budget) * max(1, int(getattr(self.bot, "APPROX_CHARS_PER_TOKEN", 4))))
        if len(text) <= target_chars:
            return text
        return text[: max(0, target_chars - 3)].rstrip() + "..."
