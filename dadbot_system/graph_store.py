import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path


class GraphStoreBackend:
    name = "unknown"

    def ensure_storage(self):
        raise NotImplementedError

    def replace_graph(self, nodes, edges):
        raise NotImplementedError

    def fetch_graph(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def stats(self):
        raise NotImplementedError


class SQLiteGraphStore(GraphStoreBackend):
    name = "sqlite"

    def __init__(self, bot, db_path):
        self.bot = bot
        self.db_path = Path(db_path)

    def with_connection(self, operation, write=False):
        last_error = None
        for attempt in range(4):
            try:
                io_lock = getattr(self.bot, "_io_lock", None)
                if io_lock is None:
                    with closing(sqlite3.connect(self.db_path, timeout=5)) as connection:
                        with connection:
                            connection.execute("PRAGMA busy_timeout = 5000")
                            if write:
                                connection.execute("PRAGMA journal_mode=WAL")
                            result = operation(connection)
                        if write:
                            # Revert to DELETE mode before close to avoid WAL
                            # checkpoint stalls on Windows when io_lock is held.
                            connection.execute("PRAGMA journal_mode=DELETE")
                        return result

                with io_lock:
                    with closing(sqlite3.connect(self.db_path, timeout=5)) as connection:
                        with connection:
                            connection.execute("PRAGMA busy_timeout = 5000")
                            if write:
                                connection.execute("PRAGMA journal_mode=WAL")
                            result = operation(connection)
                        if write:
                            # Revert to DELETE mode before close to avoid WAL
                            # checkpoint stalls on Windows when io_lock is held.
                            connection.execute("PRAGMA journal_mode=DELETE")
                        return result
            except sqlite3.OperationalError as exc:
                last_error = exc
                if "locked" not in str(exc).lower() or attempt == 3:
                    raise
                time.sleep(0.1 * (attempt + 1))
        raise last_error

    def ensure_storage(self):
        self.with_connection(
            lambda connection: connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    node_key TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    source_type TEXT,
                    source_id TEXT,
                    content TEXT,
                    category TEXT,
                    mood TEXT,
                    confidence REAL,
                    updated_at TEXT,
                    attributes_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_key TEXT PRIMARY KEY,
                    source_key TEXT NOT NULL,
                    target_key TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL,
                    confidence REAL,
                    updated_at TEXT,
                    evidence_json TEXT NOT NULL,
                    attributes_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_source_type ON graph_nodes(source_type);
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_updated_at ON graph_nodes(updated_at);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_key);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_key);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_relation_type ON graph_edges(relation_type);
                """
            ),
            write=True,
        )

    def replace_graph(self, nodes, edges):
        self.ensure_storage()

        def operation(connection):
            connection.execute("DELETE FROM graph_edges")
            connection.execute("DELETE FROM graph_nodes")
            connection.executemany(
                """
                INSERT INTO graph_nodes (
                    node_key,
                    node_type,
                    label,
                    source_type,
                    source_id,
                    content,
                    category,
                    mood,
                    confidence,
                    updated_at,
                    attributes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["node_key"],
                        row["node_type"],
                        row["label"],
                        row.get("source_type"),
                        row.get("source_id"),
                        row.get("content", ""),
                        row.get("category"),
                        row.get("mood"),
                        float(row.get("confidence", 0.0)),
                        row.get("updated_at"),
                        json.dumps(row.get("attributes", {}), ensure_ascii=True, sort_keys=True),
                    )
                    for row in nodes
                ],
            )
            connection.executemany(
                """
                INSERT INTO graph_edges (
                    edge_key,
                    source_key,
                    target_key,
                    relation_type,
                    weight,
                    confidence,
                    updated_at,
                    evidence_json,
                    attributes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["edge_key"],
                        row["source_key"],
                        row["target_key"],
                        row["relation_type"],
                        float(row.get("weight", 0.0)),
                        float(row.get("confidence", 0.0)),
                        row.get("updated_at"),
                        json.dumps(row.get("evidence", []), ensure_ascii=True, sort_keys=True),
                        json.dumps(row.get("attributes", {}), ensure_ascii=True, sort_keys=True),
                    )
                    for row in edges
                ],
            )

        self.with_connection(operation, write=True)

    @staticmethod
    def _parse_json(value, fallback):
        try:
            return json.loads(value)
        except Exception:
            return fallback

    def fetch_graph(self):
        if not self.db_path.exists():
            return {"nodes": [], "edges": [], "updated_at": None}

        def operation(connection):
            node_rows = connection.execute(
                """
                SELECT node_key, node_type, label, source_type, source_id, content, category, mood, confidence, updated_at, attributes_json
                FROM graph_nodes
                """
            ).fetchall()
            edge_rows = connection.execute(
                """
                SELECT edge_key, source_key, target_key, relation_type, weight, confidence, updated_at, evidence_json, attributes_json
                FROM graph_edges
                """
            ).fetchall()
            return node_rows, edge_rows

        node_rows, edge_rows = self.with_connection(operation)
        nodes = [
            {
                "node_key": row[0],
                "node_type": row[1],
                "label": row[2],
                "source_type": row[3],
                "source_id": row[4],
                "content": row[5] or "",
                "category": row[6],
                "mood": row[7],
                "confidence": float(row[8] or 0.0),
                "updated_at": row[9],
                "attributes": self._parse_json(row[10], {}),
            }
            for row in node_rows
        ]
        edges = [
            {
                "edge_key": row[0],
                "source_key": row[1],
                "target_key": row[2],
                "relation_type": row[3],
                "weight": float(row[4] or 0.0),
                "confidence": float(row[5] or 0.0),
                "updated_at": row[6],
                "evidence": self._parse_json(row[7], []),
                "attributes": self._parse_json(row[8], {}),
            }
            for row in edge_rows
        ]
        updated_at = None
        for item in [*nodes, *edges]:
            value = item.get("updated_at")
            if value and (updated_at is None or str(value) > str(updated_at)):
                updated_at = value
        return {"nodes": nodes, "edges": edges, "updated_at": updated_at}

    def clear(self):
        if not self.db_path.exists():
            return
        try:
            self.with_connection(lambda connection: connection.executescript("DELETE FROM graph_edges; DELETE FROM graph_nodes;"), write=True)
        except Exception:
            try:
                self.db_path.unlink()
            except Exception:
                pass

    def stats(self):
        if not self.db_path.exists():
            return {"node_count": 0, "edge_count": 0}

        def operation(connection):
            node_row = connection.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()
            edge_row = connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()
            return {
                "node_count": int(node_row[0]) if node_row else 0,
                "edge_count": int(edge_row[0]) if edge_row else 0,
            }

        return self.with_connection(operation)


class PostgresGraphStore(GraphStoreBackend):
    name = "postgres"

    def __init__(self, dsn, table_prefix="dadbot_graph"):
        self.dsn = dsn
        self.table_prefix = str(table_prefix or "dadbot_graph").strip() or "dadbot_graph"
        self.nodes_table = f"{self.table_prefix}_nodes"
        self.edges_table = f"{self.table_prefix}_edges"

    def with_connection(self, operation):
        import psycopg

        with psycopg.connect(self.dsn, autocommit=True, connect_timeout=5) as connection:
            return operation(connection)

    def ensure_storage(self):
        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.nodes_table} (
                        node_key TEXT PRIMARY KEY,
                        node_type TEXT NOT NULL,
                        label TEXT NOT NULL,
                        source_type TEXT,
                        source_id TEXT,
                        content TEXT,
                        category TEXT,
                        mood TEXT,
                        confidence DOUBLE PRECISION,
                        updated_at TEXT,
                        attributes_json JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.edges_table} (
                        edge_key TEXT PRIMARY KEY,
                        source_key TEXT NOT NULL,
                        target_key TEXT NOT NULL,
                        relation_type TEXT NOT NULL,
                        weight DOUBLE PRECISION,
                        confidence DOUBLE PRECISION,
                        updated_at TEXT,
                        evidence_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        attributes_json JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.nodes_table}_type ON {self.nodes_table}(node_type)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.nodes_table}_source_type ON {self.nodes_table}(source_type)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.nodes_table}_updated_at ON {self.nodes_table}(updated_at)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.edges_table}_source ON {self.edges_table}(source_key)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.edges_table}_target ON {self.edges_table}(target_key)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.edges_table}_relation_type ON {self.edges_table}(relation_type)")

        self.with_connection(operation)

    def replace_graph(self, nodes, edges):
        self.ensure_storage()

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(f"TRUNCATE {self.edges_table}")
                cursor.execute(f"TRUNCATE {self.nodes_table}")
                cursor.executemany(
                    f"""
                    INSERT INTO {self.nodes_table} (
                        node_key,
                        node_type,
                        label,
                        source_type,
                        source_id,
                        content,
                        category,
                        mood,
                        confidence,
                        updated_at,
                        attributes_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    [
                        (
                            row["node_key"],
                            row["node_type"],
                            row["label"],
                            row.get("source_type"),
                            row.get("source_id"),
                            row.get("content", ""),
                            row.get("category"),
                            row.get("mood"),
                            float(row.get("confidence", 0.0)),
                            row.get("updated_at"),
                            json.dumps(row.get("attributes", {}), ensure_ascii=True, sort_keys=True),
                        )
                        for row in nodes
                    ],
                )
                cursor.executemany(
                    f"""
                    INSERT INTO {self.edges_table} (
                        edge_key,
                        source_key,
                        target_key,
                        relation_type,
                        weight,
                        confidence,
                        updated_at,
                        evidence_json,
                        attributes_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    """,
                    [
                        (
                            row["edge_key"],
                            row["source_key"],
                            row["target_key"],
                            row["relation_type"],
                            float(row.get("weight", 0.0)),
                            float(row.get("confidence", 0.0)),
                            row.get("updated_at"),
                            json.dumps(row.get("evidence", []), ensure_ascii=True, sort_keys=True),
                            json.dumps(row.get("attributes", {}), ensure_ascii=True, sort_keys=True),
                        )
                        for row in edges
                    ],
                )

        self.with_connection(operation)

    @staticmethod
    def _parse_json(value, fallback):
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return fallback

    def fetch_graph(self):
        self.ensure_storage()

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT node_key, node_type, label, source_type, source_id, content, category, mood, confidence, updated_at, attributes_json
                    FROM {self.nodes_table}
                    """
                )
                node_rows = cursor.fetchall()
                cursor.execute(
                    f"""
                    SELECT edge_key, source_key, target_key, relation_type, weight, confidence, updated_at, evidence_json, attributes_json
                    FROM {self.edges_table}
                    """
                )
                edge_rows = cursor.fetchall()
                return node_rows, edge_rows

        node_rows, edge_rows = self.with_connection(operation)
        nodes = [
            {
                "node_key": row[0],
                "node_type": row[1],
                "label": row[2],
                "source_type": row[3],
                "source_id": row[4],
                "content": row[5] or "",
                "category": row[6],
                "mood": row[7],
                "confidence": float(row[8] or 0.0),
                "updated_at": row[9],
                "attributes": self._parse_json(row[10], {}),
            }
            for row in node_rows
        ]
        edges = [
            {
                "edge_key": row[0],
                "source_key": row[1],
                "target_key": row[2],
                "relation_type": row[3],
                "weight": float(row[4] or 0.0),
                "confidence": float(row[5] or 0.0),
                "updated_at": row[6],
                "evidence": self._parse_json(row[7], []),
                "attributes": self._parse_json(row[8], {}),
            }
            for row in edge_rows
        ]
        updated_at = None
        for item in [*nodes, *edges]:
            value = item.get("updated_at")
            if value and (updated_at is None or str(value) > str(updated_at)):
                updated_at = value
        return {"nodes": nodes, "edges": edges, "updated_at": updated_at}

    def clear(self):
        self.ensure_storage()

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(f"TRUNCATE {self.edges_table}")
                cursor.execute(f"TRUNCATE {self.nodes_table}")

        self.with_connection(operation)

    def stats(self):
        self.ensure_storage()

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {self.nodes_table}")
                node_row = cursor.fetchone()
                cursor.execute(f"SELECT COUNT(*) FROM {self.edges_table}")
                edge_row = cursor.fetchone()
                return {
                    "node_count": int(node_row[0]) if node_row else 0,
                    "edge_count": int(edge_row[0]) if edge_row else 0,
                }

        return self.with_connection(operation)