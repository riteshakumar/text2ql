"""Parsed query validation shared by generation, execution and evaluation.

This is a statement/shape guard, not a database authorization boundary. Database
roles and caller policy hooks must still restrict data and side-effecting UDFs.
"""
from __future__ import annotations

from typing import Any

from graphql import OperationType, get_operation_ast, parse, validate
from graphql.language import FieldNode, FragmentDefinitionNode, FragmentSpreadNode, InlineFragmentNode
from sqlglot import exp, parse as parse_sql
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from text2ql.types import ValidationError

SUPPORTED_SQL_DIALECTS = {"sqlite", "postgres", "mysql", "tsql"}


def sql_dialect(value: str | None) -> str:
    dialect = {"postgresql": "postgres", "sqlserver": "tsql", "mssql": "tsql"}.get(value or "sqlite", value or "sqlite")
    if dialect not in SUPPORTED_SQL_DIALECTS:
        raise ValueError(f"Unsupported SQL dialect '{value}'. Choose sqlite, postgres, mysql or tsql.")
    return dialect


def sqlalchemy_sql(tree: exp.Query, dialect: str) -> str:
    """Keep named binds in SQLAlchemy's :name syntax across SQL dialects."""
    from sqlglot.dialects import Dialect
    selected = Dialect.get_or_raise(dialect)

    class NamedBindGenerator(selected.generator_class):
        def placeholder_sql(self, expression: exp.Placeholder) -> str:
            return f":{expression.name}" if expression.this else super().placeholder_sql(expression)

    return NamedBindGenerator(dialect=selected).generate(tree)


def validate_sql(query: str, config: Any = None, *, dialect: str = "sqlite") -> exp.Query:
    """Require a single read query, then validate known tables and columns."""
    dialect = sql_dialect(dialect)
    try:
        statements = [item for item in parse_sql(query, read=dialect) if item is not None]
    except (ParseError, ValueError) as exc:
        raise ValidationError("Invalid SQL", [str(exc)]) from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValidationError("Only one SQL read query is permitted", ["Expected SELECT, WITH … SELECT or a read-only set operation"])
    tree = statements[0]
    forbidden = (exp.DML, exp.DDL, exp.Command, exp.Into, exp.Lock)
    if any(isinstance(node, forbidden) for node in tree.walk()):
        raise ValidationError("SQL contains a prohibited operation", ["Writes, commands, SELECT INTO and locking reads are not allowed"])
    if not list(tree.find_all(exp.Select)):
        raise ValidationError("Expected a SELECT query", ["The statement has no SELECT"])
    if config is not None and config.entities:
        known = {name.lower(): name for name in config.entities}
        for scope in traverse_scope(tree):
            for source in scope.sources.values():
                if isinstance(source, exp.Table):
                    # A qualified database/schema is meaningful, never silently
                    # strip it and grant access based on the leaf table name.
                    name = ".".join(part.name for part in source.parts)
                    if name.lower() not in known:
                        raise ValidationError("Unknown SQL table", [name])
        schema = {
            name: {col: "UNKNOWN" for col in config.fields_by_entity.get(name, config.fields)}
            for name in config.entities
        }
        if schema and all(schema.values()) and all("." not in name for name in schema):
            try:
                qualify(tree.copy(), dialect=dialect, schema=schema, identify=False,
                        validate_qualify_columns=True, allow_partial_qualification=False)
            except OptimizeError as exc:
                raise ValidationError("Unknown or ambiguous SQL column", [str(exc)]) from exc
    return tree


def validate_graphql(query: str, config: Any = None, *, operation_name: str | None = None) -> Any:
    """Preserve the complete document and reject mutations/subscriptions."""
    try:
        document = parse(query)
    except Exception as exc:
        raise ValidationError("Invalid GraphQL", [str(exc)]) from exc
    from graphql.language import OperationDefinitionNode

    operations = [node for node in document.definitions if isinstance(node, OperationDefinitionNode)]
    if not operations or any(node.operation != OperationType.QUERY for node in operations):
        raise ValidationError("Only GraphQL queries are permitted", ["Mutation and subscription operations are prohibited"])
    operation = get_operation_ast(document, operation_name)
    if operation is None:
        raise ValidationError("Ambiguous GraphQL operation", ["Select an operation by name"])
    actual_schema = getattr(config, "graphql_schema", None)
    if actual_schema is not None:
        errors = validate(actual_schema, document)
        if errors:
            raise ValidationError("GraphQL schema validation failed", [error.message for error in errors])
    elif config is not None and config.entities:
        fragments = {node.name.value: node for node in document.definitions if isinstance(node, FragmentDefinitionNode)}

        def arguments(node: Any, allowed: set[str]) -> None:
            if not allowed:
                return
            for argument in node.arguments:
                name = argument.name.value
                base = name
                for suffix in ("_gte", "_lte", "_nin", "_gt", "_lt", "_ne", "_in"):
                    if base.endswith(suffix):
                        base = base[:-len(suffix)]
                        break
                if name not in allowed and base not in allowed and name not in {"and", "or", "not", "distinct", "having"}:
                    raise ValidationError("Unknown GraphQL argument", [name])

        def roots(selection: Any, visiting: frozenset[str] = frozenset(), entity: str | None = None,
                  fields: set[str] | None = None) -> None:
            for node in selection.selections:
                if isinstance(node, FieldNode):
                    name = node.name.value
                    if name == "__typename":
                        continue
                    if entity is None:
                        if name not in config.entities:
                            raise ValidationError("Unknown GraphQL root field", [name])
                        arguments(node, set(config.args_by_entity.get(name, [])))
                        if node.selection_set is not None:
                            roots(node.selection_set, visiting, name,
                                  set(config.fields_by_entity.get(name, config.fields)))
                        continue
                    relation = config.relations_by_entity.get(entity, {}).get(name)
                    if relation is not None:
                        arguments(node, set(relation.args))
                        if node.selection_set is None:
                            raise ValidationError("Missing GraphQL relation selection", [name])
                        roots(node.selection_set, visiting, relation.target,
                              set(relation.fields or config.fields_by_entity.get(relation.target, [])))
                    elif name in {"count", "sum", "avg", "min", "max"}:
                        arguments(node, {"field"})
                    elif fields and name not in fields:
                        raise ValidationError("Unknown GraphQL field", [f"{entity}.{name}"])
                    elif node.selection_set is not None:
                        raise ValidationError("Unknown GraphQL nested selection", [f"{entity}.{name}"])
                elif isinstance(node, InlineFragmentNode):
                    roots(node.selection_set, visiting, entity, fields)
                elif isinstance(node, FragmentSpreadNode):
                    name = node.name.value
                    if name in visiting or name not in fragments:
                        raise ValidationError("Invalid GraphQL fragment", [name])
                    roots(fragments[name].selection_set, visiting | {name}, entity, fields)
        roots(operation.selection_set)
    return document
