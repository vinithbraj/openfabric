# Mermaid Diagrams

These diagrams reflect the current OpenFABRIC architecture. For the full design narrative, see [System Design](./SYSTEM_DESIGN.md).

## End-To-End Request Flow

```mermaid
flowchart TD
    A[CLI / API / OpenWebUI Request] --> B[ExecutionEngine]
    B --> C[Task Extraction + Runtime Context]
    C --> D[TaskPlanner]
    D --> E[LLMActionPlanner]
    E --> F[Structured Action JSON]
    F --> G[Canonicalize Actions]
    G --> H[Validate Dataflow / Tools / Safety / Shape]
    H --> I[ExecutionPlan]
    I --> J[PlanExecutor]
    J --> K[ToolRegistry]
    K --> L[Registered Tools]
    L --> M[StepLog History]
    M --> N[Final Presentation Boundary]
    N --> O[Response Renderer]
    O --> P[Markdown / Artifact Link]
```

## LLM Planning Boundary

```mermaid
flowchart LR
    A[User Goal] --> B[Planner Prompt + Tool Manifests]
    B --> C[LLM Action Planner]
    C --> D[Action JSON]
    D --> E[Canonicalizer]
    E --> F{Deterministic Validators}
    F -->|Valid| G[ExecutionPlan]
    F -->|Repairable| H[Compact Repair Facts]
    H --> C
    F -->|Unsafe / Invalid| I[Safe Failure]
```

The LLM proposes structured actions. It does not get authority to bypass validators, execute raw commands, ignore dataflow contracts, or format raw result payloads for final output.

## Deterministic Validation Boundary

```mermaid
flowchart TD
    A[Canonical Action Plan] --> B[Tool Registry Check]
    A --> C[Tool Output Contract Check]
    A --> D[Dataflow Reference Check]
    A --> E[Temporal Canonicalization]
    A --> F[Semantic Obligations]
    A --> G[Domain Safety]
    G --> H[SQL AST / Schema Validation]
    G --> I[Shell Safety]
    G --> J[Filesystem Roots]
    G --> K[SLURM Read-Only Safety]
    B --> L[Validated ExecutionPlan]
    C --> L
    D --> L
    E --> L
    F --> L
    H --> L
    I --> L
    J --> L
    K --> L
```

## Final Output Boundary

```mermaid
flowchart TD
    A[Tool Step Logs] --> B[OutputEnvelope]
    A --> C[Capability Presenter]
    B --> D[Result Shape Validation]
    C --> D
    D --> E{List/Table over Threshold?}
    E -->|Yes| F[Auto-Artifact CSV/TXT]
    E -->|No| G[Inline Markdown]
    F --> H[Rows Written + File Link]
    G --> I[Response Renderer]
    H --> I
    I --> J[Stats + DAG + Safe Details]
    J --> K[OpenWebUI/User Markdown]
```

User mode never displays raw JSON. Large list/table outputs are artifacted by display-row count, while scalar and grouped-count prompts validate the structured primary result instead of counting numbers in rendered Markdown.

## Auto-Artifact Decision Path

```mermaid
flowchart TD
    A[Final Step History] --> B[Infer Goal Output Contract]
    B --> C{Scalar / Status / File?}
    C -->|Yes| D[No Auto Artifact]
    C -->|No| E[Find Table/List Envelope]
    E --> F{Presentation Rows > Threshold?}
    F -->|No| G[Render Inline]
    F -->|Yes| H[Write Safe Random File]
    H --> I[Return Count + Clickable Link]
```

## SQL Request Path

```mermaid
flowchart TD
    A[SQL-Related Prompt] --> B[LLMActionPlanner]
    B --> C[sql.query / sql.schema / sql.validate Action]
    C --> D[SQL Catalog + Concept Resolver]
    D --> E[SQL AST Validation]
    E --> F[Read-Only Safety]
    F --> G{Execute Query?}
    G -->|Validate Only| H[SQL Validation Markdown]
    G -->|Execute| I[SQL Worker]
    I --> J[Rows / Scalar / Schema Result]
    J --> K[Presentation + Output Shape]
```

## SLURM Request Path

```mermaid
flowchart TD
    A[SLURM Prompt] --> B[LLMActionPlanner]
    B --> C[slurm.* Action]
    C --> D[Temporal + Semantic Canonicalization]
    D --> E[SLURM Argument Safety]
    E --> F[Gateway / Fixture Execution]
    F --> G[Structured Queue / Accounting / Node Result]
    G --> H[SLURM Presenter]
    H --> I[Markdown or Artifact]
```

SLURM prompts use native read-only `slurm.*` tools. They are not translated into arbitrary shell commands.

## Compatibility Capability-Pack Flow

Capability packs remain for helper code, fixtures, evals, and compatibility tests. They are not the default natural-language request path.

```mermaid
flowchart LR
    A[Compatibility Caller / Test] --> B[Capability Registry]
    B --> C[Pack classify]
    C --> D{Matched?}
    D -->|No| E[Unmatched Compatibility Result]
    D -->|Yes| F[Typed Intent]
    F --> G[Pack compile]
    G --> H[ExecutionPlan Fragment]
```

## Worker Lifecycle

```mermaid
flowchart TD
    A[Request Starts] --> B[Register RunHandle]
    B --> C[CancellationToken]
    C --> D[Plan + Execute Steps]
    D --> E{Child Worker Needed?}
    E -->|Yes| F[ManagedProcess]
    F --> G[Drain Queue Before Join]
    E -->|No| H[Normal Step Result]
    G --> I[Unregister Process]
    H --> J[Unregister RunHandle]
    I --> J
    K[Client Disconnect / Shutdown] --> L[Cancel Active Runs]
    L --> M[Terminate / Kill Child Processes]
```
