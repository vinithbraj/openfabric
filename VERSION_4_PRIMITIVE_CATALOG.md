# Version 4 Deterministic Primitive Catalog

This file defines the target primitive catalog for the deterministic-first architecture for `sql_runner` and `slurm_runner`.

The design goal is:

- deterministic primitives are the ground truth
- the LLM selects from a known primitive catalog instead of inventing routines first
- the LLM also proposes a fallback SQL query or Slurm command
- the agent always runs the deterministic primitive first
- fallback is used only when the deterministic primitive cannot answer cleanly

## Primitive Contract

Every primitive should eventually have the same structural contract:

- `primitive_id`: stable unique ID
- `family`: high-level grouping
- `intent_tags`: routing and retrieval hints
- `summary`: one-line description
- `required_params`: strict required arguments
- `optional_params`: strict optional arguments
- `read_only`: whether the primitive is side-effect free
- `deterministic_executor`: code path that does not rely on free-form LLM generation
- `validator`: schema-aware or gateway-aware parameter validation
- `result_shape`: count, table, scalar, boolean, summary, path, control_ack
- `success_condition`: exact rule for "answer generated"
- `fallback_kind`: `sql`, `slurm_command`, or `none`
- `fallback_allowed`: whether fallback can run if deterministic path fails

## LLM Selection Contract

For both SQL and Slurm, the agent-local LLM should eventually return:

- `primitive_id`
- `selection_reason`
- `parameters`
- `confidence`
- `fallback`
- `fallback.reason`
- `fallback.sql` or `fallback.command` plus `fallback.args`

The agent should reject any primitive ID not present in the catalog.

## SQL Catalog

The SQL catalog should be exhaustive for common read-only database work. The deterministic layer should prefer schema-backed validation and query templates over model-authored SQL.

### SQL Schema Discovery

- `sql.schema.get_dialect`: identify the active SQL dialect
- `sql.schema.list_schemas`: list all schemas
- `sql.schema.list_tables`: list all tables
- `sql.schema.list_views`: list all views
- `sql.schema.list_materialized_views`: list all materialized views
- `sql.schema.list_tables_in_schema`: list all tables in one schema
- `sql.schema.list_views_in_schema`: list all views in one schema
- `sql.schema.describe_database`: return schemas, tables, columns, and relationships
- `sql.schema.describe_schema`: return a focused schema summary for one schema
- `sql.schema.describe_table`: return a focused schema summary for one table
- `sql.schema.list_columns`: list all columns for all tables
- `sql.schema.list_columns_in_table`: list all columns in one table
- `sql.schema.find_tables_by_keyword`: fuzzy-match tables by name token
- `sql.schema.find_columns_by_keyword`: fuzzy-match columns by name token
- `sql.schema.list_primary_keys`: list primary keys
- `sql.schema.list_foreign_keys`: list foreign keys
- `sql.schema.list_unique_constraints`: list unique constraints
- `sql.schema.list_indexes`: list indexes
- `sql.schema.list_nullable_columns`: list nullable columns
- `sql.schema.list_non_nullable_columns`: list non-nullable columns
- `sql.schema.relationship_graph`: return table-to-table foreign key graph
- `sql.schema.join_path_one_hop`: find a direct join path between two tables
- `sql.schema.join_path_two_hop`: find a two-hop join path between two tables
- `sql.schema.identifier_catalog`: return the canonical identifier catalog

### SQL Table Inspection

- `sql.table.exists`: check whether a table exists
- `sql.table.row_count`: count rows in a table
- `sql.table.sample_rows`: sample rows from a table
- `sql.table.latest_rows`: return latest rows using a validated timestamp column
- `sql.table.oldest_rows`: return oldest rows using a validated timestamp column
- `sql.table.distinct_values`: list distinct values from one column
- `sql.table.distinct_count`: count distinct values in one column
- `sql.table.null_count`: count nulls in one column
- `sql.table.non_null_count`: count non-nulls in one column
- `sql.table.min_value`: return min for one column
- `sql.table.max_value`: return max for one column
- `sql.table.min_max`: return min and max for one column
- `sql.table.numeric_summary`: count, min, max, avg, sum for one numeric column
- `sql.table.value_exists`: check if a value exists in one column
- `sql.table.duplicate_keys`: find duplicate values for a key column
- `sql.table.top_n_rows`: top N rows ordered by one validated column
- `sql.table.bottom_n_rows`: bottom N rows ordered by one validated column

### SQL Row Retrieval

- `sql.rows.select_all`: return rows from one table with limit
- `sql.rows.select_columns`: return selected validated columns from one table
- `sql.rows.where_equals`: filter one table by exact equality
- `sql.rows.where_not_equals`: filter one table by exact inequality
- `sql.rows.where_in`: filter one table by inclusion list
- `sql.rows.where_not_in`: filter one table by exclusion list
- `sql.rows.where_like`: filter one table by pattern match
- `sql.rows.where_ilike`: case-insensitive pattern match where supported
- `sql.rows.where_prefix`: filter by string prefix
- `sql.rows.where_suffix`: filter by string suffix
- `sql.rows.where_contains`: filter by substring containment
- `sql.rows.where_null`: filter where column is null
- `sql.rows.where_not_null`: filter where column is not null
- `sql.rows.where_between`: filter by numeric or temporal range
- `sql.rows.where_gt`: filter greater than
- `sql.rows.where_gte`: filter greater than or equal
- `sql.rows.where_lt`: filter less than
- `sql.rows.where_lte`: filter less than or equal
- `sql.rows.where_boolean`: filter by boolean column
- `sql.rows.sorted`: return rows sorted by one validated column
- `sql.rows.paginated`: deterministic page retrieval with limit and offset

### SQL Scalar Aggregates

- `sql.agg.count_all`: count all rows
- `sql.agg.count_where`: count rows matching a simple predicate
- `sql.agg.count_distinct`: count distinct values
- `sql.agg.sum`: sum one numeric column
- `sql.agg.avg`: average one numeric column
- `sql.agg.min`: minimum one comparable column
- `sql.agg.max`: maximum one comparable column
- `sql.agg.range`: max minus min for one numeric column
- `sql.agg.boolean_any`: whether any matching row exists
- `sql.agg.boolean_all`: whether all matching rows satisfy a predicate

### SQL Grouped Aggregates

- `sql.group.count_by_column`: grouped counts by one column
- `sql.group.count_distinct_by_column`: grouped distinct counts
- `sql.group.sum_by_column`: grouped sums
- `sql.group.avg_by_column`: grouped averages
- `sql.group.min_by_column`: grouped minimums
- `sql.group.max_by_column`: grouped maximums
- `sql.group.count_by_two_columns`: grouped counts by two columns
- `sql.group.sum_by_two_columns`: grouped sums by two columns
- `sql.group.top_groups_by_count`: top groups ordered by count
- `sql.group.top_groups_by_sum`: top groups ordered by sum
- `sql.group.filter_having_count_gt`: groups whose count exceeds threshold
- `sql.group.filter_having_sum_gt`: groups whose sum exceeds threshold
- `sql.group.filter_having_avg_gt`: groups whose average exceeds threshold

### SQL Temporal Primitives

- `sql.time.latest_timestamp`: latest value from a timestamp column
- `sql.time.earliest_timestamp`: earliest value from a timestamp column
- `sql.time.count_in_window`: count rows in a time window
- `sql.time.sum_in_window`: sum values in a time window
- `sql.time.avg_in_window`: average values in a time window
- `sql.time.rows_in_window`: list rows in a time window
- `sql.time.count_by_day`: grouped daily counts
- `sql.time.count_by_week`: grouped weekly counts
- `sql.time.count_by_month`: grouped monthly counts
- `sql.time.sum_by_day`: grouped daily sums
- `sql.time.sum_by_week`: grouped weekly sums
- `sql.time.sum_by_month`: grouped monthly sums
- `sql.time.latest_n_rows`: latest N records by timestamp
- `sql.time.period_over_period_count`: compare two time windows by count
- `sql.time.period_over_period_sum`: compare two time windows by sum

### SQL Join Primitives

These should use only validated join paths from schema foreign keys.

- `sql.join.count_related_rows_one_hop`: count related rows across one join
- `sql.join.list_related_rows_one_hop`: list related rows across one join
- `sql.join.exists_related_rows_one_hop`: boolean existence across one join
- `sql.join.sum_related_values_one_hop`: sum related numeric values across one join
- `sql.join.avg_related_values_one_hop`: average related numeric values across one join
- `sql.join.group_related_count_one_hop`: grouped count across one join
- `sql.join.group_related_sum_one_hop`: grouped sum across one join
- `sql.join.find_missing_relations_one_hop`: left-join anti-match for orphan detection
- `sql.join.count_related_rows_two_hop`: count related rows across two joins
- `sql.join.list_related_rows_two_hop`: list related rows across two joins
- `sql.join.group_related_count_two_hop`: grouped count across two joins

### SQL Data Quality Primitives

- `sql.quality.table_exists`: verify a table exists
- `sql.quality.column_exists`: verify a column exists
- `sql.quality.value_exists`: verify a specific value exists
- `sql.quality.null_ratio`: compute null percentage for one column
- `sql.quality.blank_ratio`: compute blank-string ratio for one text column
- `sql.quality.duplicate_ratio`: compute duplicate ratio for a key candidate
- `sql.quality.invalid_range_count`: count values outside a range
- `sql.quality.orphan_foreign_keys`: find rows whose foreign key has no parent
- `sql.quality.unique_value_profile`: distinct count plus top value frequency
- `sql.quality.constant_column_check`: detect columns with one value only

### SQL Comparison Primitives

- `sql.compare.count_between_tables`: compare table row counts
- `sql.compare.distinct_between_tables`: compare distinct value counts
- `sql.compare.missing_keys_between_tables`: keys present in A but not B
- `sql.compare.overlap_keys_between_tables`: keys present in both
- `sql.compare.top_categories_between_windows`: compare top grouped categories over two windows
- `sql.compare.metric_between_windows`: compare one aggregate across two windows

### SQL Export and Artifact Primitives

- `sql.export.schemas_json`: export schema list as JSON
- `sql.export.tables_json`: export table list as JSON
- `sql.export.columns_json`: export column list as JSON
- `sql.export.rows_json`: export result rows as JSON
- `sql.export.rows_csv`: export result rows as CSV
- `sql.export.scalar_text`: export scalar answer as text
- `sql.export.summary_text`: export reduced summary as text

### SQL Fallback-Only Categories

These should remain in the catalog as valid intent classes but should often route to fallback SQL unless and until deterministic coverage is implemented:

- multi-join analytical ranking beyond two validated hops
- window functions
- complex nested subqueries
- free-form case expressions
- dialect-specific JSON operators
- recursive CTEs
- regex-heavy matching
- percentile and median where dialect support is inconsistent
- arbitrary expression synthesis

## Slurm Catalog

The Slurm catalog should be exhaustive for cluster inspection, queue inspection, accounting, and safe control operations through the gateway.

### Slurm Cluster and Partition Discovery

- `slurm.cluster.health`: gateway and cluster availability check
- `slurm.cluster.list_partitions`: list partitions
- `slurm.cluster.partition_summary`: partition state summary
- `slurm.cluster.partition_details`: detailed partition records
- `slurm.cluster.partition_exists`: check whether partition exists
- `slurm.cluster.default_partition`: return the default partition
- `slurm.cluster.list_reservations`: list reservations
- `slurm.cluster.list_qos`: list QoS names if available
- `slurm.cluster.scheduler_diagnostics`: scheduler diagnostics summary
- `slurm.cluster.fairshare_summary`: fairshare summary
- `slurm.cluster.priority_summary`: scheduler priority summary

### Slurm Node Primitives

- `slurm.nodes.list_all`: list all nodes
- `slurm.nodes.count_all`: count all nodes
- `slurm.nodes.state_summary`: count nodes by state
- `slurm.nodes.list_by_state`: list nodes in one state
- `slurm.nodes.count_by_state`: count nodes in one state
- `slurm.nodes.list_in_partition`: list nodes in one partition
- `slurm.nodes.count_in_partition`: count nodes in one partition
- `slurm.nodes.partition_state_summary`: node state summary within one partition
- `slurm.nodes.gpu_inventory`: list GPU-capable nodes and GPU metadata
- `slurm.nodes.idle_inventory`: list idle nodes
- `slurm.nodes.down_inventory`: list down nodes
- `slurm.nodes.drain_inventory`: list drained nodes
- `slurm.nodes.mixed_inventory`: list mixed-state nodes
- `slurm.nodes.allocated_inventory`: list allocated nodes
- `slurm.nodes.node_details`: show one node in detail
- `slurm.nodes.reason_summary`: summarize node reasons for unavailable states

### Slurm Active Queue Primitives

- `slurm.jobs.list_active_all`: list active jobs across all users
- `slurm.jobs.list_active_by_user`: list active jobs for one user
- `slurm.jobs.list_active_by_partition`: list active jobs for one partition
- `slurm.jobs.list_active_by_state`: list active jobs for one state
- `slurm.jobs.list_pending`: list pending jobs
- `slurm.jobs.list_running`: list running jobs
- `slurm.jobs.list_non_running`: list active jobs not in RUNNING state
- `slurm.jobs.count_active_all`: count active jobs
- `slurm.jobs.count_by_user`: count active jobs for one user
- `slurm.jobs.count_by_partition`: count active jobs for one partition
- `slurm.jobs.count_by_state`: count jobs by one state
- `slurm.jobs.count_non_running`: count jobs not running
- `slurm.jobs.pending_reason_breakdown`: summarize pending reasons
- `slurm.jobs.queue_depth_by_user`: grouped queue counts by user
- `slurm.jobs.queue_depth_by_partition`: grouped queue counts by partition
- `slurm.jobs.running_time_summary`: summary of elapsed time for running jobs
- `slurm.jobs.list_long_running`: list longest-running active jobs
- `slurm.jobs.job_details`: show one job in detail
- `slurm.jobs.job_steps`: show live job-step statistics
- `slurm.jobs.job_node_allocation`: show nodes assigned to one job

### Slurm Historical Accounting Primitives

- `slurm.acct.list_completed`: list completed jobs
- `slurm.acct.list_failed`: list failed jobs
- `slurm.acct.list_cancelled`: list cancelled jobs
- `slurm.acct.list_timeout`: list timed-out jobs
- `slurm.acct.list_by_state`: list jobs by one historical state
- `slurm.acct.list_by_user`: list historical jobs for one user
- `slurm.acct.list_by_partition`: list historical jobs for one partition
- `slurm.acct.list_in_window`: list jobs in a time window
- `slurm.acct.count_completed`: count completed jobs
- `slurm.acct.count_failed`: count failed jobs
- `slurm.acct.count_cancelled`: count cancelled jobs
- `slurm.acct.count_timeout`: count timed-out jobs
- `slurm.acct.count_by_state`: count historical jobs by state
- `slurm.acct.count_by_user`: count historical jobs for one user
- `slurm.acct.count_by_partition`: count historical jobs for one partition
- `slurm.acct.elapsed_summary`: total, average, longest, shortest elapsed time
- `slurm.acct.elapsed_summary_by_partition`: elapsed summary scoped to partition
- `slurm.acct.elapsed_summary_by_user`: elapsed summary scoped to user
- `slurm.acct.longest_jobs`: list longest historical jobs
- `slurm.acct.shortest_jobs`: list shortest historical jobs
- `slurm.acct.success_rate`: successful jobs versus failed jobs
- `slurm.acct.failure_rate`: failed jobs versus completed jobs

### Slurm Resource Usage and Share Primitives

- `slurm.usage.cpu_summary`: CPU allocation or usage summary where available
- `slurm.usage.memory_summary`: memory summary where available
- `slurm.usage.gpu_summary`: GPU usage summary where available
- `slurm.usage.user_share`: fairshare for one user
- `slurm.usage.account_share`: fairshare for one account
- `slurm.usage.account_usage_summary`: usage summary by account
- `slurm.usage.user_usage_summary`: usage summary by user

### Slurm User, Account, and Association Primitives

- `slurm.assoc.list_users`: list Slurm accounting users
- `slurm.assoc.list_accounts`: list Slurm accounts
- `slurm.assoc.user_account_mapping`: map users to accounts
- `slurm.assoc.user_qos_mapping`: map users to QoS assignments
- `slurm.assoc.account_qos_mapping`: map accounts to QoS assignments
- `slurm.assoc.user_exists`: check whether accounting user exists
- `slurm.assoc.account_exists`: check whether accounting account exists

### Slurm Control Primitives

These are deterministic but not read-only.

- `slurm.control.cancel_job`: cancel one job
- `slurm.control.cancel_jobs_by_user`: cancel jobs for one user
- `slurm.control.hold_job`: hold one job
- `slurm.control.release_job`: release one job
- `slurm.control.requeue_job`: requeue one job
- `slurm.control.suspend_job`: suspend one job
- `slurm.control.resume_job`: resume one job
- `slurm.control.show_job`: show one job with `scontrol`
- `slurm.control.show_node`: show one node with `scontrol`
- `slurm.control.show_partition`: show one partition with `scontrol`

### Slurm Export and Artifact Primitives

- `slurm.export.node_inventory_json`: export node list as JSON
- `slurm.export.node_state_summary_text`: export node state summary as text
- `slurm.export.partition_summary_json`: export partition summary as JSON
- `slurm.export.active_jobs_json`: export active jobs as JSON
- `slurm.export.accounting_jobs_json`: export accounting jobs as JSON
- `slurm.export.elapsed_summary_text`: export elapsed summary as text

### Slurm Fallback-Only Categories

These should remain cataloged as valid intent classes but may need fallback command generation until deterministic coverage exists:

- advanced `sacctmgr` mutations
- multi-step control workflows with approval logic
- reservation creation or updates
- arbitrary `scontrol update` operations
- complex `sreport` reporting with multiple dimensions
- custom `sacct --format` requests with uncommon fields
- cluster-specific features not visible in gateway capabilities

## Primitive Coverage Philosophy

The deterministic catalog should be broad enough that the common path is:

- schema inspection
- table listing
- table description
- simple row retrieval
- single-table counts and aggregates
- validated one-hop and two-hop joins
- node inventory
- node state summaries
- active job counts and listings
- historical job counts and elapsed summaries
- safe job control

The fallback path should exist for:

- requests that require arbitrary SQL synthesis
- requests that require arbitrary Slurm command synthesis
- requests that reference unsupported fields, unsupported dialect features, or unsupported cluster features

## Recommended Version 4 Execution Rule

For both `sql_runner` and `slurm_runner`, `query_from_request` should become:

1. Gather deterministic context.
2. Ask the LLM to select one primitive from this catalog and also propose a fallback.
3. Validate primitive ID and parameters strictly.
4. Execute the deterministic primitive first.
5. If a valid answer is produced, stop there.
6. If no valid answer is produced, execute the fallback SQL or Slurm command.
7. Return both the deterministic plan and whether fallback was required in the result payload.

## Recommended Version 4 Metadata to Advertise

Each deterministic agent should advertise:

- `execution_model`: `deterministic_first_with_llm_fallback`
- `deterministic_catalog_version`
- `deterministic_catalog_families`
- `deterministic_catalog_size`
- `fallback_policy`
- `methods` that reflect the major primitive families

This lets the planner and any downstream LLM reason from known, bounded capabilities instead of inventing routines from scratch.
