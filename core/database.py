# =============================================================================
# DATABASE - Connection Pool e Gerenciamento de Conexões
# =============================================================================
# Centraliza conexões SQLite com pooling e extensão sqlite-vec
# =============================================================================

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue

import apsw
import sqlite_vec

from .config import get_config

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Pool de conexões SQLite com suporte a sqlite-vec.

    Gerencia conexões reutilizáveis para evitar overhead de criação/destruição.
    Thread-safe via Queue.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        pool_size: int = 5,
        use_apsw: bool = True,
        load_vec_extension: bool = True,
    ):
        """
        Inicializa o pool de conexões.

        Args:
            db_path: Caminho do banco. Se None, usa config centralizada.
            pool_size: Número máximo de conexões no pool.
            use_apsw: Se True, usa APSW (necessário para sqlite-vec). Se False, usa sqlite3.
            load_vec_extension: Se True, carrega extensão sqlite-vec.
        """
        config = get_config()
        self.db_path = str(db_path) if db_path else str(config.rag_db_path)
        self.pool_size = pool_size
        self.use_apsw = use_apsw
        self.load_vec_extension = load_vec_extension
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created = 0

        # Garantir que o diretório existe
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _create_connection(self) -> apsw.Connection | sqlite3.Connection:
        """Cria uma nova conexão com as configurações apropriadas."""
        if self.use_apsw:
            conn = apsw.Connection(self.db_path)
            if self.load_vec_extension:
                conn.enableloadextension(True)
                conn.loadextension(sqlite_vec.loadable_path())
                conn.enableloadextension(False)
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row

        return conn

    def get_connection(self) -> apsw.Connection | sqlite3.Connection:
        """Obtém uma conexão do pool ou cria uma nova se necessário."""
        try:
            conn = self._pool.get_nowait()
            # Verificar se conexão ainda é válida
            try:
                if self.use_apsw:
                    conn.cursor().execute("SELECT 1")
                else:
                    conn.execute("SELECT 1")
                return conn
            except Exception:
                # Conexão inválida, criar nova
                pass
        except Empty:
            pass

        # Criar nova conexão se pool vazio ou conexão inválida
        with self._lock:
            if self._created < self.pool_size:
                self._created += 1
                return self._create_connection()

        # Pool cheio, esperar por conexão
        return self._pool.get(timeout=30)

    def return_connection(self, conn: apsw.Connection | sqlite3.Connection) -> None:
        """Retorna uma conexão ao pool."""
        try:
            self._pool.put_nowait(conn)
        except Exception:
            # Pool cheio, fechar conexão
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._created = max(0, self._created - 1)

    @contextmanager
    def connection(self):
        """Context manager para obter e retornar conexão automaticamente.

        Exemplo:
            with pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM documentos")
        """
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)

    def close_all(self) -> None:
        """Fecha todas as conexões no pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
            except Exception:
                pass
        with self._lock:
            self._created = 0


# Instâncias globais (singleton pattern)
_rag_pool: ConnectionPool | None = None
_audit_pool: ConnectionPool | None = None


def get_rag_pool() -> ConnectionPool:
    """Obtém o pool de conexões do RAG database."""
    global _rag_pool
    if _rag_pool is None:
        config = get_config()
        _rag_pool = ConnectionPool(
            db_path=config.rag_db_path,
            pool_size=5,
            use_apsw=True,
            load_vec_extension=True,
        )
    return _rag_pool


def get_audit_pool() -> ConnectionPool:
    """Obtém o pool de conexões do audit database."""
    global _audit_pool
    if _audit_pool is None:
        config = get_config()
        _audit_pool = ConnectionPool(
            db_path=config.audit_db_path,
            pool_size=3,
            use_apsw=False,  # Audit não precisa de sqlite-vec
            load_vec_extension=False,
        )
    return _audit_pool


def reset_pools() -> None:
    """Reseta todos os pools (útil para testes)."""
    global _rag_pool, _audit_pool
    if _rag_pool:
        _rag_pool.close_all()
        _rag_pool = None
    if _audit_pool:
        _audit_pool.close_all()
        _audit_pool = None
