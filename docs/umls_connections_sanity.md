# UMLS Connections Sanity Queries

Use these after running `KG_PIPELINE_PHASE=umls_connections` with
`KG_UMLS_CONNECTIONS_WRITE_NEO4J=true`.

## Duplicate Logical Edges

```cypher
MATCH ()-[r]->()
WHERE type(r) STARTS WITH 'UMLS_'
  AND r.provenance = 'umls_connections'
WITH r.edge_key AS edge_key, count(r) AS n, collect(DISTINCT type(r)) AS types
WHERE edge_key IS NULL OR trim(toString(edge_key)) = '' OR n > 1
RETURN edge_key, n, types
ORDER BY n DESC, edge_key;
```

## Endpoint CUI Consistency

```cypher
MATCH (source:Concept)-[r]->(target:Concept)
WHERE type(r) STARTS WITH 'UMLS_'
  AND r.provenance = 'umls_connections'
WITH source, r, target, properties(source) AS sp, properties(target) AS tp, properties(r) AS rp
WHERE toUpper(coalesce(toString(sp['umls_cui']), '')) <>
      toUpper(coalesce(toString(rp['source_cui']), ''))
   OR toUpper(coalesce(toString(tp['umls_cui']), '')) <>
      toUpper(coalesce(toString(rp['target_cui']), ''))
RETURN type(r) AS relationship_type,
       rp['edge_key'] AS edge_key,
       source.name AS source_concept,
       target.name AS target_concept,
       sp['umls_cui'] AS source_node_cui,
       rp['source_cui'] AS relationship_source_cui,
       tp['umls_cui'] AS target_node_cui,
       rp['target_cui'] AS relationship_target_cui
ORDER BY relationship_type, edge_key;
```

## Counts By Relationship Type

```cypher
MATCH ()-[r]->()
WHERE type(r) STARTS WITH 'UMLS_'
  AND r.provenance = 'umls_connections'
RETURN type(r) AS relationship_type, count(r) AS n
ORDER BY relationship_type;
```

## First-Extension Local Type Audit

Compatible first-extension materializations must have
`local_type_compatible=true`. Type-incompatible candidates remain in the review
exports, but are skipped during materialization.

```cypher
MATCH (source:Concept)-[r]->(target:Concept)
WHERE type(r) IN [
  'UMLS_HAS_DEFINITIONAL_MANIFESTATION',
  'UMLS_DEFINITIONAL_MANIFESTATION_OF',
  'UMLS_USES_DEVICE',
  'UMLS_DEVICE_USED_BY',
  'UMLS_HAS_DIRECT_DEVICE',
  'UMLS_DIRECT_DEVICE_OF',
  'UMLS_HAS_MEASURED_COMPONENT',
  'UMLS_MEASURED_COMPONENT_OF'
]
  AND r.provenance = 'umls_connections'
  AND coalesce(r.local_type_compatible, false) <> true
RETURN type(r) AS relationship_type,
       r.edge_key AS edge_key,
       r.relation_name AS relation_name,
       source.name AS source_concept,
       source.canonical_type AS source_type,
       target.name AS target_concept,
       target.canonical_type AS target_type,
       r.local_type_compatibility_reason AS reason
ORDER BY relationship_type, edge_key;
```

## Idempotence Check

Run the same `umls_connections` command twice with identical config, then compare:

```cypher
MATCH ()-[r]->()
WHERE type(r) STARTS WITH 'UMLS_'
  AND r.provenance = 'umls_connections'
RETURN count(r) AS materialized_umls_relationships,
       count(DISTINCT r.edge_key) AS distinct_edge_keys;
```

The two counts should remain equal, and the result should not increase on the
second run unless the input graph or UMLS relation cache changed.
