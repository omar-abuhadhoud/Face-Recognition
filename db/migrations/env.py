from logging.config import fileConfig

from alembic import context

from db.models import Base
from db.session import engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata   # lets `alembic revision --autogenerate` compare against the models


def run_migrations_offline():
    """Write the SQL to stdout instead of running it (alembic upgrade head --sql)."""
    context.configure(url=engine.url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run the migrations against the database."""
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
