from langchain.memory.entity import BaseEntityStore
from langchain.pydantic_v1 import Field
from typing import Any, Optional

class CorrectedSQLiteEntityStore(BaseEntityStore):
    """SQLite-backed Entity store"""

    session_id: str = "default"
    table_name: str = "memory_store"

    def __init__(
        self,
        conn: Any = Field(None),
        session_id: str = "default",
        db_file: str = "entities.db",
        table_name: str = "memory_store",
        *args: Any,
        **kwargs: Any,
    ):
        try:
            import sqlite3
        except ImportError:
            raise ImportError(
                "Could not import sqlite3 python package. "
                "Please install it with `pip install sqlite3`."
            )
        super().__init__(*args, **kwargs)

        self.__dict__["conn"] = sqlite3.connect(db_file)
        self.session_id = session_id
        self.table_name = table_name
        self._create_table_if_not_exists()

    @property
    def full_table_name(self) -> str:
        return f"{self.table_name}_{self.session_id}"

    def _create_table_if_not_exists(self) -> None:
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.full_table_name} (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        with self.conn:
            self.conn.execute(create_table_query)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        query = f"""
            SELECT value
            FROM {self.full_table_name}
            WHERE key = ?
        """
        cursor = self.conn.execute(query, (key,))
        result = cursor.fetchone()
        if result is not None:
            value = result[0]
            return value
        return default

    def set(self, key: str, value: Optional[str]) -> None:
        if not value:
            return self.delete(key)
        query = f"""
            INSERT OR REPLACE INTO {self.full_table_name} (key, value)
            VALUES (?, ?)
        """
        with self.conn:
            self.conn.execute(query, (key, value))

    def delete(self, key: str) -> None:
        query = f"""
            DELETE FROM {self.full_table_name}
            WHERE key = ?
        """
        with self.conn:
            self.conn.execute(query, (key,))

    def exists(self, key: str) -> bool:
        query = f"""
            SELECT 1
            FROM {self.full_table_name}
            WHERE key = ?
            LIMIT 1
        """
        cursor = self.conn.execute(query, (key,))
        result = cursor.fetchone()
        return result is not None

    def clear(self) -> None:
        query = f"""
            DELETE FROM {self.full_table_name}
        """
        with self.conn:
            self.conn.execute(query)
