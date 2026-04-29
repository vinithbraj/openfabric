# Mermaid Diagrams

## End-to-End Request Flow

```mermaid
flowchart TD
    A[CLI or API Request] --> B[ExecutionEngine]
    B --> C[TaskPlanner]
    C --> D[LLMActionPlanner]
    D --> E[ActionPlan JSON]
    E --> F[Canonicalizer]
    F --> G[Validator]
    G --> H[ExecutionPlan]
    H --> I[PlanExecutor]
    I --> J[Tools]
    J --> K[RuntimeValidator]
    K --> L[runtime.return or Final Output Summary]
    L --> M[Session State / Final Output]
```

## Capability-Pack Lifecycle

Capability packs are retained for compatibility helpers and tests. They are
not the default natural-language planning route.

```mermaid
flowchart LR
    A[Goal Text] --> B[Pack classify]
    B --> C{Matched?}
    C -- No --> D[Next Pack]
    C -- Yes --> E[IntentResult]
    E --> F[Pack compile]
    F --> G[CompiledIntentPlan]
    G --> H[ExecutionPlan]
```

## `runtime.return` and Output Shaping

```mermaid
flowchart TD
    A[Tool Result or Referenced Value] --> B[OutputContract]
    B --> C[normalize_output]
    C --> D[Normalized Value]
    D --> E[render_output]
    E --> F[Final User Output String]
```

## Evaluation Flow

```mermaid
flowchart TD
    A[Checked-in Eval Pack JSON] --> B[evaluate_capability_packs.py]
    B --> C[Fixture Workspace]
    B --> D[Runtime Execution]
    D --> E[Final Output + Metrics]
    E --> F[Strict and Semantic Checks]
    F --> G[Per-Pack Thresholds]
    G --> H[Capability Eval Report]
```

## SLURM Fuzzy Prompt to Typed Intent

```mermaid
flowchart TD
    A[Fuzzy SLURM Prompt] --> B[Deterministic SlurmCapabilityPack classify]
    B --> C{Matched?}
    C -- Yes --> D[Typed SLURM Intent]
    C -- No --> E{LLM Intent Extraction Enabled?}
    E -- No --> F[Raw Planner Fallback]
    E -- Yes --> G[LLMIntentExtractor]
    G --> H[Validated SLURM Intent JSON]
    H --> I[SLURM Safety Validation]
    I --> J[SlurmCapabilityPack compile]
    J --> K[slurm.* Tool + runtime.return]
```
