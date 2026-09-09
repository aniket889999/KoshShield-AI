import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Add apps/api/src to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import koshshield.models  # noqa: F401, E402
from koshshield.config import get_settings  # noqa: E402
from koshshield.database import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def resolve_database_url(url: str) -> str:
    if url.startswith("sqlite:///"):
        path_str = url[len("sqlite:///") :]
        if path_str != ":memory:" and not path_str.startswith("/"):
            repo_root = Path(__file__).resolve().parents[2]
            abs_path = (repo_root / path_str).resolve()
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{abs_path}"
    return url


database_url = config.get_main_option("sqlalchemy.url")
if not database_url:
    settings = get_settings()
    database_url = settings.database_url
database_url = resolve_database_url(database_url)
config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url") or get_settings().database_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = config.get_main_option("sqlalchemy.url") or get_settings().database_url
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = url
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        from koshshield.database.migration import IncompatibleSchemaError, inspect_schema_state

        state, detail = inspect_schema_state(connection)
        if state == "incompatible":
            raise IncompatibleSchemaError(
                f"Cannot migrate database safely: {detail}. "
                "Manual inspection required; aborting without modifying schema."
            )
        if state == "legacy_pre_alembic":
            from sqlalchemy import Column, String, Table

            version_table = Table(
                "alembic_version",
                target_metadata,
                Column("version_num", String(32), primary_key=True, nullable=False),
            )
            version_table.create(connection, checkfirst=True)
            connection.execute(version_table.insert().values(version_num="0001_initial_schema"))
            connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()
        connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
