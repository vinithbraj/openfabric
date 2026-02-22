ASDL is a control layer for multi-agent systems. It provides a blueprint language for defining system topology, a contract system for typed communication, a runtime orchestration engine for routing events, and a transport abstraction layer for connecting to agents across different execution environments.

ASDL is not the intelligence itself. The intelligence lives inside the agents — language models, databases, retrieval systems, segmentation services, or any other computational component. ASDL governs how those intelligent components are composed into a coherent system.

Layers

At the bottom is the behavior layer. This includes the agents themselves: planners, retrievers, synthesizers, LLMs, databases, or any service that performs actual work. These components implement logic and produce results.

Above that is the orchestration layer — the ASDL runtime. It routes events between agents, enforces data contracts, determines which agents subscribe to which events, and connects the system together. It acts as the traffic controller of the architecture.

At the top is the blueprint layer — the ASDL specification. This declarative definition describes the topology of the system, the contracts that define data shape, the event graph that governs communication, and the execution bindings that tell the runtime how to reach each agent. It is the wiring diagram of the system.

Together, these layers separate structure from behavior and allow intelligent components to be composed, validated, and orchestrated without hard-coding orchestration logic into application code.