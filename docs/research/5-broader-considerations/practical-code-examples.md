# Practical Code Examples: LLM Agent Architectures

This document provides practical, runnable code examples for implementing various LLM agent architectures discussed in the main research document.

## Table of Contents

1. [ReAct Pattern Implementation](#react-pattern-implementation)
1. [Plan-and-Execute Agent](#plan-and-execute-agent)
1. [Tree of Thoughts Implementation](#tree-of-thoughts-implementation)
1. [Memory Systems](#memory-systems)
1. [Multi-Agent Systems](#multi-agent-systems)
1. [Data Analysis Agent](#data-analysis-agent)

## ReAct Pattern Implementation

### Basic ReAct Agent with LangChain

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import pandas as pd

# Define a custom tool for data analysis
class DataAnalysisInput(BaseModel):
    query: str = Field(description="SQL-like query for data analysis")
    dataframe_name: str = Field(description="Name of the dataframe to analyze")

class DataAnalysisTool(BaseTool):
    name = "data_analysis"
    description = "Analyze data using pandas operations"
    args_schema: Type[BaseModel] = DataAnalysisInput

    def _run(self, query: str, dataframe_name: str) -> str:
        # In real implementation, maintain a registry of dataframes
        # For demo, using a sample dataframe
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'sales': [100, 150, 200, 180, 220, 250, 300, 280, 320, 350],
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })

        try:
            # Simple query parser (in practice, use proper SQL parser)
            if "sum" in query.lower():
                result = f"Total sales: {df['sales'].sum()}"
            elif "mean" in query.lower():
                result = f"Average sales: {df['sales'].mean():.2f}"
            elif "group" in query.lower():
                result = df.groupby('category')['sales'].sum().to_string()
            else:
                result = df.head().to_string()
            return result
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

# ReAct prompt template
react_prompt = PromptTemplate.from_template("""
You are a data analysis agent. Answer the following questions as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action as a JSON object
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}
""")

# Create the ReAct agent
def create_data_analysis_agent():
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    tools = [
        DataAnalysisTool(),
        Tool(
            name="calculator",
            func=lambda x: str(eval(x)),
            description="Useful for mathematical calculations"
        )
    ]

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

# Usage example
if __name__ == "__main__":
    agent = create_data_analysis_agent()

    # Example queries
    queries = [
        "What is the total sales in the dataset?",
        "Calculate the average sales and then multiply by 12 for annual projection",
        "Show me sales by category"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.invoke({"input": query})
        print(f"Answer: {result['output']}")
```

### StateAct-style Agent with State Tracking

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json

class AgentState(Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETE = "complete"

@dataclass
class StateMemory:
    """Tracks agent state across interactions"""
    current_state: AgentState = AgentState.PLANNING
    goal: str = ""
    completed_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        return f"""
Current State: {self.current_state.value}
Main Goal: {self.goal}
Completed Steps: {json.dumps(self.completed_steps, indent=2)}
Pending Steps: {json.dumps(self.pending_steps, indent=2)}
Context: {json.dumps(self.context, indent=2)}
"""

class StateActAgent:
    def __init__(self, llm):
        self.llm = llm
        self.memory = StateMemory()

    def self_prompt(self, user_input: str) -> str:
        """Generate self-prompt to maintain focus on goal"""
        return f"""
Remember your main goal: {self.memory.goal}

Current progress:
{self.memory.to_prompt_context()}

User input: {user_input}

Based on your current state and progress, what should you do next?
Stay focused on the main goal and avoid deviating from it.
"""

    def plan(self, task: str) -> List[str]:
        """Create a plan for the given task"""
        self.memory.goal = task
        self.memory.current_state = AgentState.PLANNING

        planning_prompt = f"""
Create a step-by-step plan to accomplish this task: {task}

Return a JSON list of steps. Each step should be a clear, actionable item.
Example: ["Step 1: Load the data", "Step 2: Clean missing values", ...]
"""

        response = self.llm.invoke(planning_prompt)
        steps = json.loads(response.content)
        self.memory.pending_steps = steps
        return steps

    def execute_step(self, step: str) -> Dict[str, Any]:
        """Execute a single step and track state"""
        self.memory.current_state = AgentState.EXECUTING

        # Self-prompt to maintain focus
        focused_prompt = self.self_prompt(f"Execute this step: {step}")

        execution_prompt = f"""
{focused_prompt}

Execute the following step and return the result:
Step: {step}

Return a JSON object with:
- "success": boolean
- "result": string description of what was done
- "data": any relevant data produced
"""

        response = self.llm.invoke(execution_prompt)
        result = json.loads(response.content)

        if result["success"]:
            self.memory.completed_steps.append(step)
            self.memory.pending_steps.remove(step)

        return result

    def evaluate_progress(self) -> bool:
        """Evaluate if the goal has been achieved"""
        self.memory.current_state = AgentState.EVALUATING

        eval_prompt = f"""
{self.memory.to_prompt_context()}

Has the main goal been achieved? Consider:
1. Are all planned steps completed?
2. Does the current state satisfy the original goal?
3. Is any additional work needed?

Return JSON: {{"complete": boolean, "reason": "explanation"}}
"""

        response = self.llm.invoke(eval_prompt)
        result = json.loads(response.content)

        if result["complete"]:
            self.memory.current_state = AgentState.COMPLETE

        return result["complete"]
```

## Plan-and-Execute Agent

### LangGraph Implementation

```python
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph, END
from langchain_openai import ChatOpenAI
import operator

# Define the state structure
class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[tuple], operator.add]
    current_step: int
    final_answer: str

# Planner node
def planner(state: PlanExecuteState):
    llm = ChatOpenAI(model="gpt-4")

    prompt = f"""
    Create a detailed plan to solve this task: {state['input']}

    Return a numbered list of steps that need to be executed.
    Each step should be specific and actionable.
    """

    response = llm.invoke(prompt)

    # Parse steps from response
    steps = []
    for line in response.content.split('\n'):
        if line.strip() and any(line.startswith(f"{i}.") for i in range(1, 10)):
            steps.append(line.strip())

    return {"plan": steps, "current_step": 0}

# Executor node
def executor(state: PlanExecuteState):
    llm = ChatOpenAI(model="gpt-4")

    current_step = state['plan'][state['current_step']]

    # Context from past steps
    context = "\n".join([f"Step {i+1} result: {result}"
                        for i, (step, result) in enumerate(state['past_steps'])])

    prompt = f"""
    Execute this step: {current_step}

    Context from previous steps:
    {context}

    Provide a concrete result for this step.
    """

    response = llm.invoke(prompt)

    return {
        "past_steps": [(current_step, response.content)],
        "current_step": state['current_step'] + 1
    }

# Replanner node (for dynamic adjustment)
def replanner(state: PlanExecuteState):
    llm = ChatOpenAI(model="gpt-4")

    # Get completed steps
    completed = "\n".join([f"{step} -> Result: {result}"
                          for step, result in state['past_steps']])

    prompt = f"""
    Original task: {state['input']}

    Original plan: {state['plan']}

    Completed steps:
    {completed}

    Based on the results so far, do we need to adjust the plan?
    If yes, provide an updated plan for the remaining steps.
    If no, return "CONTINUE".
    """

    response = llm.invoke(prompt)

    if response.content.strip() != "CONTINUE":
        # Parse new steps
        new_steps = []
        for line in response.content.split('\n'):
            if line.strip() and any(line.startswith(f"{i}.") for i in range(1, 10)):
                new_steps.append(line.strip())

        # Merge with original plan
        executed_steps = state['past_steps']
        remaining_original = state['plan'][state['current_step']:]

        return {"plan": state['plan'][:state['current_step']] + new_steps}

    return {}

# Solver node (generates final answer)
def solver(state: PlanExecuteState):
    llm = ChatOpenAI(model="gpt-4")

    # Compile all results
    results = "\n".join([f"{step} -> {result}"
                        for step, result in state['past_steps']])

    prompt = f"""
    Task: {state['input']}

    Execution results:
    {results}

    Based on these results, provide the final answer to the original task.
    """

    response = llm.invoke(prompt)

    return {"final_answer": response.content}

# Build the graph
def create_plan_execute_agent():
    workflow = StateGraph(PlanExecuteState)

    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("replanner", replanner)
    workflow.add_node("solver", solver)

    # Define the flow
    workflow.set_entry_point("planner")

    # From planner to executor
    workflow.add_edge("planner", "executor")

    # Conditional edge from executor
    def should_continue(state):
        if state['current_step'] >= len(state['plan']):
            return "solver"
        else:
            return "replanner"

    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "solver": "solver",
            "replanner": "replanner"
        }
    )

    # From replanner back to executor
    workflow.add_edge("replanner", "executor")

    # Solver goes to END
    workflow.add_edge("solver", END)

    return workflow.compile()

# Usage
if __name__ == "__main__":
    agent = create_plan_execute_agent()

    # Example task
    result = agent.invoke({
        "input": "Analyze sales data to find the best performing product category and create a report with recommendations",
        "plan": [],
        "past_steps": [],
        "current_step": 0,
        "final_answer": ""
    })

    print("Final Answer:", result['final_answer'])
```

## Tree of Thoughts Implementation

### Simple ToT for Data Analysis

```python
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

@dataclass
class Thought:
    """Represents a single thought/reasoning step"""
    content: str
    score: float = 0.0
    children: List['Thought'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

class ThoughtEvaluator(ABC):
    """Abstract base for thought evaluation"""

    @abstractmethod
    def evaluate(self, thought: Thought, context: str) -> float:
        pass

class LLMThoughtEvaluator(ThoughtEvaluator):
    """Evaluate thoughts using an LLM"""

    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, thought: Thought, context: str) -> float:
        prompt = f"""
        Context: {context}
        Thought: {thought.content}

        Rate this thought's quality for solving the problem (0-10):
        - Relevance to the problem
        - Logical correctness
        - Progress toward solution

        Return only a number between 0 and 10.
        """

        response = self.llm.invoke(prompt)
        try:
            score = float(response.content.strip())
            return min(max(score / 10.0, 0.0), 1.0)
        except:
            return 0.5

class TreeOfThoughts:
    def __init__(self, llm, evaluator: ThoughtEvaluator,
                 breadth: int = 3, depth: int = 3):
        self.llm = llm
        self.evaluator = evaluator
        self.breadth = breadth  # Number of thoughts to generate at each step
        self.depth = depth      # Maximum depth of the tree

    def generate_thoughts(self, problem: str, parent_thought: Optional[Thought] = None) -> List[Thought]:
        """Generate multiple thoughts for a given state"""

        context = problem
        if parent_thought:
            context += f"\nCurrent approach: {parent_thought.content}"

        prompt = f"""
        Problem: {context}

        Generate {self.breadth} different approaches or next steps to solve this problem.
        Each approach should be distinct and explore different angles.

        Format each approach as:
        Approach N: [description]
        """

        response = self.llm.invoke(prompt)

        thoughts = []
        for line in response.content.split('\n'):
            if line.strip() and 'Approach' in line:
                content = line.split(':', 1)[1].strip() if ':' in line else line
                thought = Thought(content=content)
                thought.score = self.evaluator.evaluate(thought, problem)
                thoughts.append(thought)

        return thoughts[:self.breadth]

    def search_bfs(self, problem: str) -> Tuple[Thought, List[Thought]]:
        """Breadth-first search through thought tree"""
        root = Thought(content="Initial state", score=0.0)

        # Queue of (thought, depth) tuples
        queue = [(root, 0)]
        all_thoughts = [root]
        best_thought = root

        while queue:
            current_thought, current_depth = queue.pop(0)

            if current_depth >= self.depth:
                continue

            # Generate child thoughts
            children = self.generate_thoughts(problem, current_thought)
            current_thought.children = children
            all_thoughts.extend(children)

            # Update best thought
            for child in children:
                if child.score > best_thought.score:
                    best_thought = child

                # Add to queue for further exploration
                queue.append((child, current_depth + 1))

        return best_thought, all_thoughts

    def search_beam(self, problem: str, beam_width: int = 2) -> Tuple[Thought, List[Thought]]:
        """Beam search through thought tree"""
        root = Thought(content="Initial state", score=0.0)

        current_level = [root]
        all_thoughts = [root]

        for depth in range(self.depth):
            next_level = []

            # Generate thoughts for each node in current level
            for thought in current_level:
                children = self.generate_thoughts(problem, thought)
                thought.children = children
                next_level.extend(children)
                all_thoughts.extend(children)

            # Keep only top beam_width thoughts
            next_level.sort(key=lambda t: t.score, reverse=True)
            current_level = next_level[:beam_width]

            if not current_level:
                break

        # Find best thought overall
        best_thought = max(all_thoughts, key=lambda t: t.score)
        return best_thought, all_thoughts

    def solve(self, problem: str, search_method: str = "bfs") -> str:
        """Solve a problem using tree of thoughts"""

        if search_method == "bfs":
            best_thought, all_thoughts = self.search_bfs(problem)
        elif search_method == "beam":
            best_thought, all_thoughts = self.search_beam(problem)
        else:
            raise ValueError(f"Unknown search method: {search_method}")

        # Generate final solution based on best path
        path = self._get_path_to_thought(best_thought, all_thoughts)

        solution_prompt = f"""
        Problem: {problem}

        Best reasoning path:
        {' -> '.join([t.content for t in path])}

        Based on this reasoning, provide the final solution.
        """

        response = self.llm.invoke(solution_prompt)
        return response.content

    def _get_path_to_thought(self, target: Thought, all_thoughts: List[Thought]) -> List[Thought]:
        """Reconstruct path from root to target thought"""
        # Simplified - in practice, maintain parent pointers
        path = [target]
        # This is a placeholder - implement proper path reconstruction
        return path

# Usage example for data analysis
class DataAnalysisToT:
    def __init__(self, llm):
        self.llm = llm
        evaluator = LLMThoughtEvaluator(llm)
        self.tot = TreeOfThoughts(llm, evaluator, breadth=3, depth=3)

    def analyze(self, data_description: str, analysis_goal: str) -> str:
        problem = f"""
        Data: {data_description}
        Goal: {analysis_goal}

        Find the best approach to analyze this data and achieve the goal.
        """

        solution = self.tot.solve(problem, search_method="beam")
        return solution

# Example usage
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4")
    analyzer = DataAnalysisToT(llm)

    result = analyzer.analyze(
        data_description="Sales data with columns: date, product, category, price, quantity, customer_segment",
        analysis_goal="Identify factors driving revenue growth and recommend optimization strategies"
    )

    print("Analysis Result:", result)
```

## Memory Systems

### Comprehensive Memory System Implementation

```python
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

@dataclass
class MemoryItem:
    """Base class for memory items"""
    id: str
    timestamp: datetime
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EpisodicMemory(MemoryItem):
    """Specific interaction or episode"""
    query: str
    response: str
    success: bool
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticMemory(MemoryItem):
    """General knowledge or pattern"""
    concept: str
    description: str
    confidence: float
    sources: List[str] = field(default_factory=list)

@dataclass
class ProceduralMemory(MemoryItem):
    """How-to knowledge"""
    task: str
    steps: List[str]
    conditions: Dict[str, Any]
    success_rate: float = 0.0

class VectorStore:
    """Simple in-memory vector store for semantic search"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.embeddings = []
        self.items = []

    def add(self, text: str, item: Any):
        embedding = self.embedding_model.embed_query(text)
        self.embeddings.append(embedding)
        self.items.append(item)

    def search(self, query: str, k: int = 5) -> List[Tuple[Any, float]]:
        if not self.embeddings:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = [(self.items[i], similarities[i]) for i in top_indices]
        return results

class ComprehensiveMemorySystem:
    def __init__(self, llm, embedding_model, max_episodes: int = 1000):
        self.llm = llm
        self.embedding_model = embedding_model
        self.max_episodes = max_episodes

        # Different memory stores
        self.episodic_store = VectorStore(embedding_model)
        self.semantic_store = VectorStore(embedding_model)
        self.procedural_store = VectorStore(embedding_model)

        # Working memory (current context)
        self.working_memory: List[Any] = []
        self.working_memory_limit = 10

    def store_episode(self, query: str, response: str, success: bool,
                     context: Optional[Dict] = None):
        """Store an episodic memory"""
        episode_id = hashlib.md5(f"{query}{datetime.now()}".encode()).hexdigest()[:8]

        episode = EpisodicMemory(
            id=episode_id,
            timestamp=datetime.now(),
            content=f"Q: {query}\nA: {response}",
            query=query,
            response=response,
            success=success,
            context=context or {}
        )

        # Add to episodic store
        self.episodic_store.add(f"{query} {response}", episode)

        # Update working memory
        self._update_working_memory(episode)

        # Extract semantic knowledge
        self._extract_semantic_knowledge(episode)

    def _extract_semantic_knowledge(self, episode: EpisodicMemory):
        """Extract general knowledge from episodes"""
        prompt = f"""
        From this interaction:
        Query: {episode.query}
        Response: {episode.response}
        Success: {episode.success}

        Extract any general knowledge or patterns that could be useful for future tasks.
        Return JSON: {{"concept": "...", "description": "...", "confidence": 0.0-1.0}}
        """

        response = self.llm.invoke(prompt)
        try:
            knowledge = json.loads(response.content)

            semantic = SemanticMemory(
                id=hashlib.md5(knowledge['concept'].encode()).hexdigest()[:8],
                timestamp=datetime.now(),
                content=knowledge['description'],
                concept=knowledge['concept'],
                description=knowledge['description'],
                confidence=knowledge['confidence'],
                sources=[episode.id]
            )

            self.semantic_store.add(
                f"{knowledge['concept']} {knowledge['description']}",
                semantic
            )
        except:
            pass

    def learn_procedure(self, task: str, episodes: List[EpisodicMemory]):
        """Learn a procedure from successful episodes"""
        successful_episodes = [e for e in episodes if e.success]

        if not successful_episodes:
            return

        prompt = f"""
        Task: {task}

        Successful approaches:
        {json.dumps([{"query": e.query, "response": e.response} for e in successful_episodes], indent=2)}

        Extract a general procedure for accomplishing this task.
        Return JSON: {{"steps": [...], "conditions": {{...}}}}
        """

        response = self.llm.invoke(prompt)
        try:
            procedure_data = json.loads(response.content)

            procedure = ProceduralMemory(
                id=hashlib.md5(task.encode()).hexdigest()[:8],
                timestamp=datetime.now(),
                content=task,
                task=task,
                steps=procedure_data['steps'],
                conditions=procedure_data['conditions'],
                success_rate=len(successful_episodes) / len(episodes)
            )

            self.procedural_store.add(task, procedure)
        except:
            pass

    def recall_relevant_memories(self, query: str, memory_types: List[str] = None) -> Dict[str, List[Any]]:
        """Recall relevant memories for a query"""
        if memory_types is None:
            memory_types = ['episodic', 'semantic', 'procedural']

        results = {}

        if 'episodic' in memory_types:
            episodic_results = self.episodic_store.search(query, k=3)
            results['episodic'] = [item for item, score in episodic_results if score > 0.5]

        if 'semantic' in memory_types:
            semantic_results = self.semantic_store.search(query, k=3)
            results['semantic'] = [item for item, score in semantic_results if score > 0.5]

        if 'procedural' in memory_types:
            procedural_results = self.procedural_store.search(query, k=2)
            results['procedural'] = [item for item, score in procedural_results if score > 0.6]

        return results

    def _update_working_memory(self, item: MemoryItem):
        """Update working memory with FIFO policy"""
        self.working_memory.append(item)
        if len(self.working_memory) > self.working_memory_limit:
            self.working_memory.pop(0)

    def get_context_prompt(self, query: str) -> str:
        """Generate a context prompt with relevant memories"""
        memories = self.recall_relevant_memories(query)

        context_parts = []

        # Add working memory
        if self.working_memory:
            recent = "\n".join([f"- {m.content[:100]}..." for m in self.working_memory[-3:]])
            context_parts.append(f"Recent context:\n{recent}")

        # Add episodic memories
        if memories.get('episodic'):
            episodes = "\n".join([f"- {e.query} -> {e.response[:100]}..."
                                for e in memories['episodic'][:2]])
            context_parts.append(f"\nSimilar past interactions:\n{episodes}")

        # Add semantic knowledge
        if memories.get('semantic'):
            knowledge = "\n".join([f"- {s.concept}: {s.description}"
                                 for s in memories['semantic']])
            context_parts.append(f"\nRelevant knowledge:\n{knowledge}")

        # Add procedural knowledge
        if memories.get('procedural'):
            procedures = "\n".join([f"- {p.task}: {', '.join(p.steps[:3])}..."
                                  for p in memories['procedural']])
            context_parts.append(f"\nRelevant procedures:\n{procedures}")

        return "\n".join(context_parts)

# Usage example
class MemoryAugmentedAgent:
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.memory = ComprehensiveMemorySystem(llm, embedding_model)

    def process_query(self, query: str) -> str:
        # Get relevant context from memory
        context = self.memory.get_context_prompt(query)

        # Generate response with context
        prompt = f"""
        {context}

        Current query: {query}

        Provide a response based on the context and your knowledge.
        """

        response = self.llm.invoke(prompt)

        # Store this interaction
        self.memory.store_episode(
            query=query,
            response=response.content,
            success=True,  # Would determine based on feedback
            context={"model": "gpt-4", "timestamp": str(datetime.now())}
        )

        return response.content
```

## Multi-Agent Systems

### AutoGen-style Multi-Agent System

```python
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
from enum import Enum

class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    HANDOFF = "handoff"
    TERMINATE = "terminate"

@dataclass
class Message:
    type: MessageType
    sender: str
    recipient: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

class Agent(ABC):
    """Base agent class"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.message_history: List[Message] = []

    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process a message and optionally return a response"""
        pass

    def add_to_history(self, message: Message):
        self.message_history.append(message)

class LLMAgent(Agent):
    """Agent powered by an LLM"""

    def __init__(self, name: str, description: str, llm, system_prompt: str):
        super().__init__(name, description)
        self.llm = llm
        self.system_prompt = system_prompt

    async def process_message(self, message: Message) -> Optional[Message]:
        self.add_to_history(message)

        # Build context from message history
        context = self._build_context()

        prompt = f"""
        {self.system_prompt}

        Message history:
        {context}

        Current message from {message.sender}:
        {message.content}

        Respond appropriately. If you need help from another agent, indicate who and why.
        """

        response = await asyncio.to_thread(self.llm.invoke, prompt)

        # Parse response to determine action
        if "HANDOFF:" in response.content:
            # Extract handoff target
            parts = response.content.split("HANDOFF:", 1)
            target = parts[1].split("\n")[0].strip()
            content = parts[1].split("\n", 1)[1] if "\n" in parts[1] else ""

            return Message(
                type=MessageType.HANDOFF,
                sender=self.name,
                recipient=target,
                content=content
            )
        elif "TERMINATE:" in response.content:
            result = response.content.replace("TERMINATE:", "").strip()
            return Message(
                type=MessageType.TERMINATE,
                sender=self.name,
                recipient="orchestrator",
                content=result
            )
        else:
            return Message(
                type=MessageType.RESULT,
                sender=self.name,
                recipient=message.sender,
                content=response.content
            )

    def _build_context(self) -> str:
        # Last 5 messages for context
        recent = self.message_history[-5:]
        return "\n".join([f"{m.sender} -> {m.recipient}: {m.content[:100]}..."
                         for m in recent])

class ToolAgent(Agent):
    """Agent that executes specific tools"""

    def __init__(self, name: str, description: str, tools: Dict[str, Callable]):
        super().__init__(name, description)
        self.tools = tools

    async def process_message(self, message: Message) -> Optional[Message]:
        self.add_to_history(message)

        # Parse tool request
        try:
            if isinstance(message.content, dict):
                tool_name = message.content.get('tool')
                args = message.content.get('args', {})

                if tool_name in self.tools:
                    result = await asyncio.to_thread(self.tools[tool_name], **args)
                    return Message(
                        type=MessageType.RESULT,
                        sender=self.name,
                        recipient=message.sender,
                        content={"result": result}
                    )

            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                recipient=message.sender,
                content="Invalid tool request"
            )
        except Exception as e:
            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                recipient=message.sender,
                content=f"Tool execution error: {str(e)}"
            )

class Orchestrator:
    """Orchestrates communication between agents"""

    def __init__(self, agents: List[Agent]):
        self.agents = {agent.name: agent for agent in agents}
        self.message_queue = asyncio.Queue()
        self.running = False

    async def send_message(self, message: Message):
        await self.message_queue.put(message)

    async def run(self):
        """Main orchestration loop"""
        self.running = True

        while self.running:
            try:
                # Get next message
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                # Route to recipient
                if message.recipient in self.agents:
                    agent = self.agents[message.recipient]
                    response = await agent.process_message(message)

                    if response:
                        if response.type == MessageType.TERMINATE:
                            self.running = False
                            return response.content
                        elif response.type == MessageType.HANDOFF:
                            # Re-route to new recipient
                            await self.send_message(response)
                        else:
                            # Send response back
                            await self.send_message(response)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Orchestrator error: {e}")

    async def solve_task(self, task: str, primary_agent: str = None) -> str:
        """Solve a task using the multi-agent system"""
        if primary_agent is None:
            primary_agent = list(self.agents.keys())[0]

        # Send initial task
        await self.send_message(Message(
            type=MessageType.TASK,
            sender="user",
            recipient=primary_agent,
            content=task
        ))

        # Run until completion
        result = await self.run()
        return result

# Example: Data Analysis Multi-Agent System
def create_data_analysis_team(llm):
    # Planning agent
    planner = LLMAgent(
        name="planner",
        description="Creates analysis plans",
        llm=llm,
        system_prompt="""You are a data analysis planning agent.
        Break down analysis tasks into steps and delegate to appropriate agents:
        - 'analyst' for data exploration and insights
        - 'statistician' for statistical analysis
        - 'visualizer' for creating charts
        When the task is complete, use TERMINATE: followed by the final result."""
    )

    # Analysis agent
    analyst = LLMAgent(
        name="analyst",
        description="Performs data exploration",
        llm=llm,
        system_prompt="""You are a data analyst. Explore data and provide insights.
        If you need statistical analysis, use HANDOFF: statistician
        If you need visualizations, use HANDOFF: visualizer"""
    )

    # Statistics agent
    statistician = LLMAgent(
        name="statistician",
        description="Performs statistical analysis",
        llm=llm,
        system_prompt="You are a statistician. Perform statistical tests and analysis."
    )

    # Visualization agent with tools
    import matplotlib.pyplot as plt
    import seaborn as sns

    def create_chart(chart_type: str, data: Dict[str, List], **kwargs):
        plt.figure(figsize=(10, 6))

        if chart_type == "bar":
            plt.bar(data['x'], data['y'])
        elif chart_type == "line":
            plt.plot(data['x'], data['y'])
        elif chart_type == "scatter":
            plt.scatter(data['x'], data['y'])

        plt.title(kwargs.get('title', 'Chart'))
        plt.xlabel(kwargs.get('xlabel', 'X'))
        plt.ylabel(kwargs.get('ylabel', 'Y'))

        # Save to file
        filename = f"chart_{chart_type}.png"
        plt.savefig(filename)
        plt.close()

        return filename

    visualizer = ToolAgent(
        name="visualizer",
        description="Creates data visualizations",
        tools={"create_chart": create_chart}
    )

    return Orchestrator([planner, analyst, statistician, visualizer])

# Usage
async def main():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4")
    team = create_data_analysis_team(llm)

    result = await team.solve_task(
        "Analyze sales data to identify trends and create visualizations",
        primary_agent="planner"
    )

    print("Analysis Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Data Analysis Agent

### Production-Ready Data Analysis Agent

````python
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json
from pathlib import Path
import logging
from functools import wraps
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_execution(func):
    """Decorator for safe execution with error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    return wrapper

@dataclass
class AnalysisRequest:
    dataset_path: str
    analysis_type: str
    parameters: Dict[str, Any]
    output_format: str = "report"

@dataclass
class AnalysisResult:
    success: bool
    data: Any
    insights: List[str]
    visualizations: List[str]
    metadata: Dict[str, Any]

class DataAnalysisAgent:
    """Production-ready data analysis agent"""

    def __init__(self, llm, working_dir: Path = Path("./analysis_output")):
        self.llm = llm
        self.working_dir = working_dir
        self.working_dir.mkdir(exist_ok=True)

        # Analysis capabilities
        self.analysis_functions = {
            "descriptive": self._descriptive_analysis,
            "correlation": self._correlation_analysis,
            "time_series": self._time_series_analysis,
            "segmentation": self._segmentation_analysis,
            "anomaly": self._anomaly_detection
        }

    @safe_execution
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Main analysis entry point"""
        logger.info(f"Starting analysis: {request.analysis_type}")

        # Load data
        df = self._load_data(request.dataset_path)

        # Validate data
        validation = self._validate_data(df)
        if not validation['valid']:
            return AnalysisResult(
                success=False,
                data=None,
                insights=[f"Data validation failed: {validation['reason']}"],
                visualizations=[],
                metadata=validation
            )

        # Perform analysis
        if request.analysis_type in self.analysis_functions:
            result = self.analysis_functions[request.analysis_type](
                df, request.parameters
            )
        else:
            result = self._custom_analysis(df, request)

        # Generate insights
        insights = self._generate_insights(df, result, request)

        # Create visualizations
        visualizations = self._create_visualizations(df, result, request)

        # Compile report
        if request.output_format == "report":
            self._generate_report(result, insights, visualizations, request)

        return AnalysisResult(
            success=True,
            data=result,
            insights=insights,
            visualizations=visualizations,
            metadata={
                "rows": len(df),
                "columns": len(df.columns),
                "analysis_type": request.analysis_type,
                "timestamp": str(datetime.now())
            }
        )

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load data with multiple format support"""
        path_obj = Path(path)

        if path_obj.suffix == '.csv':
            return pd.read_csv(path)
        elif path_obj.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif path_obj.suffix == '.json':
            return pd.read_json(path)
        elif path_obj.suffix == '.parquet':
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")

    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality"""
        issues = []

        # Check for empty dataframe
        if df.empty:
            return {"valid": False, "reason": "Empty dataframe"}

        # Check for missing values
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > 0.5]
        if not high_missing.empty:
            issues.append(f"High missing values in: {list(high_missing.index)}")

        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            issues.append("No numeric columns found for analysis")

        return {
            "valid": len(issues) == 0,
            "reason": "; ".join(issues) if issues else "Data is valid",
            "missing_summary": missing_pct.to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

    @safe_execution
    def _descriptive_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Perform descriptive statistical analysis"""
        numeric_df = df.select_dtypes(include=[np.number])

        analysis = {
            "summary_stats": numeric_df.describe().to_dict(),
            "correlations": numeric_df.corr().to_dict(),
            "skewness": numeric_df.skew().to_dict(),
            "kurtosis": numeric_df.kurtosis().to_dict()
        }

        # Add categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            analysis["categorical"] = {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            }

        return analysis

    @safe_execution
    def _correlation_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Analyze correlations in the data"""
        numeric_df = df.select_dtypes(include=[np.number])

        # Pearson correlation
        pearson_corr = numeric_df.corr(method='pearson')

        # Find strong correlations
        strong_corr = []
        corr_matrix = pearson_corr.values
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > params.get('threshold', 0.7):
                    strong_corr.append({
                        'var1': pearson_corr.index[i],
                        'var2': pearson_corr.columns[j],
                        'correlation': corr_matrix[i, j]
                    })

        return {
            "correlation_matrix": pearson_corr.to_dict(),
            "strong_correlations": strong_corr,
            "method": "pearson"
        }

    @safe_execution
    def _time_series_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Perform time series analysis"""
        date_col = params.get('date_column')
        value_col = params.get('value_column')

        if not date_col or not value_col:
            # Try to auto-detect
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) == 0:
                # Try to parse date columns
                for col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_col = col
                        break
                    except:
                        continue

        if date_col and value_col:
            df_ts = df.set_index(date_col).sort_index()

            # Basic time series stats
            analysis = {
                "trend": self._calculate_trend(df_ts[value_col]),
                "seasonality": self._detect_seasonality(df_ts[value_col]),
                "statistics": {
                    "mean": df_ts[value_col].mean(),
                    "std": df_ts[value_col].std(),
                    "min": df_ts[value_col].min(),
                    "max": df_ts[value_col].max()
                }
            }

            return analysis

        return {"error": "Could not identify time series columns"}

    def _calculate_trend(self, series: pd.Series) -> Dict:
        """Calculate trend in time series"""
        x = np.arange(len(series))
        y = series.values

        # Linear regression for trend
        coeffs = np.polyfit(x, y, 1)
        trend_line = np.poly1d(coeffs)

        return {
            "slope": coeffs[0],
            "intercept": coeffs[1],
            "direction": "increasing" if coeffs[0] > 0 else "decreasing"
        }

    def _detect_seasonality(self, series: pd.Series) -> Dict:
        """Detect seasonality patterns"""
        # Simplified seasonality detection
        if len(series) < 24:  # Need at least 2 years of monthly data
            return {"detected": False, "reason": "Insufficient data"}

        # Check for monthly patterns
        monthly_avg = series.groupby(series.index.month).mean()
        seasonality_strength = monthly_avg.std() / monthly_avg.mean()

        return {
            "detected": seasonality_strength > 0.1,
            "strength": seasonality_strength,
            "pattern": monthly_avg.to_dict()
        }

    @safe_execution
    def _segmentation_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Perform customer/data segmentation"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Select features for clustering
        features = params.get('features', df.select_dtypes(include=[np.number]).columns)
        n_clusters = params.get('n_clusters', 3)

        # Prepare data
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Analyze segments
        df_clustered = X.copy()
        df_clustered['cluster'] = clusters

        segment_profiles = {}
        for cluster_id in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            segment_profiles[f"segment_{cluster_id}"] = {
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(df_clustered) * 100,
                "characteristics": cluster_data.drop('cluster', axis=1).mean().to_dict()
            }

        return {
            "n_segments": n_clusters,
            "segment_profiles": segment_profiles,
            "inertia": kmeans.inertia_
        }

    @safe_execution
    def _anomaly_detection(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Detect anomalies in the data"""
        from sklearn.ensemble import IsolationForest

        # Select features
        features = params.get('features', df.select_dtypes(include=[np.number]).columns)
        contamination = params.get('contamination', 0.1)

        # Prepare data
        X = df[features].dropna()

        # Detect anomalies
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(X)

        # Analyze anomalies
        anomaly_indices = X.index[anomalies == -1]
        normal_indices = X.index[anomalies == 1]

        return {
            "n_anomalies": len(anomaly_indices),
            "anomaly_rate": len(anomaly_indices) / len(X) * 100,
            "anomaly_indices": anomaly_indices.tolist()[:100],  # Limit for large datasets
            "anomaly_characteristics": X.loc[anomaly_indices].describe().to_dict()
        }

    def _custom_analysis(self, df: pd.DataFrame, request: AnalysisRequest) -> Dict:
        """Perform custom analysis using LLM"""
        # Generate analysis code
        prompt = f"""
        Generate Python code to analyze this dataset:
        Columns: {list(df.columns)}
        Data types: {df.dtypes.to_dict()}
        Shape: {df.shape}

        Analysis request: {request.analysis_type}
        Parameters: {request.parameters}

        Return only executable Python code that uses the dataframe 'df' and returns a dictionary of results.
        """

        response = self.llm.invoke(prompt)

        # Execute generated code (with safety measures)
        try:
            # Create safe execution environment
            safe_globals = {
                'df': df,
                'pd': pd,
                'np': np,
                'json': json
            }

            exec(response.content, safe_globals)

            # Assuming the code defines a 'results' variable
            return safe_globals.get('results', {})
        except Exception as e:
            logger.error(f"Custom analysis execution error: {str(e)}")
            return {"error": str(e), "generated_code": response.content}

    def _generate_insights(self, df: pd.DataFrame, analysis_result: Dict,
                          request: AnalysisRequest) -> List[str]:
        """Generate insights using LLM"""
        prompt = f"""
        Based on this data analysis, provide key insights:

        Analysis type: {request.analysis_type}
        Results: {json.dumps(analysis_result, indent=2)[:2000]}  # Truncate for token limit

        Dataset info:
        - Shape: {df.shape}
        - Columns: {list(df.columns)}

        Provide 3-5 bullet point insights that are:
        1. Actionable
        2. Specific and quantified
        3. Business-relevant

        Format as a JSON list of strings.
        """

        response = self.llm.invoke(prompt)

        try:
            insights = json.loads(response.content)
            return insights if isinstance(insights, list) else []
        except:
            # Fallback: extract bullet points
            lines = response.content.split('\n')
            return [line.strip('- •*').strip() for line in lines
                   if line.strip() and any(line.startswith(p) for p in ['-', '•', '*'])]

    def _create_visualizations(self, df: pd.DataFrame, analysis_result: Dict,
                              request: AnalysisRequest) -> List[str]:
        """Create relevant visualizations"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        viz_paths = []

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # Create visualizations based on analysis type
        if request.analysis_type == "descriptive":
            # Distribution plots
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
            if len(numeric_cols) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.ravel()

                for i, col in enumerate(numeric_cols):
                    if i < 4:
                        df[col].hist(ax=axes[i], bins=30)
                        axes[i].set_title(f'Distribution of {col}')

                plt.tight_layout()
                path = self.working_dir / "distributions.png"
                plt.savefig(path)
                plt.close()
                viz_paths.append(str(path))

        elif request.analysis_type == "correlation":
            # Correlation heatmap
            if 'correlation_matrix' in analysis_result:
                plt.figure(figsize=(10, 8))
                corr_df = pd.DataFrame(analysis_result['correlation_matrix'])
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Matrix')

                path = self.working_dir / "correlation_heatmap.png"
                plt.savefig(path)
                plt.close()
                viz_paths.append(str(path))

        elif request.analysis_type == "time_series":
            # Time series plot
            date_col = request.parameters.get('date_column')
            value_col = request.parameters.get('value_column')

            if date_col and value_col and date_col in df.columns and value_col in df.columns:
                plt.figure(figsize=(12, 6))
                df.set_index(date_col)[value_col].plot()
                plt.title(f'{value_col} over Time')
                plt.xlabel(date_col)
                plt.ylabel(value_col)

                path = self.working_dir / "time_series.png"
                plt.savefig(path)
                plt.close()
                viz_paths.append(str(path))

        return viz_paths

    def _generate_report(self, analysis_result: Dict, insights: List[str],
                        visualizations: List[str], request: AnalysisRequest):
        """Generate a comprehensive report"""
        from datetime import datetime

        report = f"""
# Data Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type:** {request.analysis_type}

## Executive Summary

{' '.join(insights[:3])}

## Detailed Analysis Results

```json
{json.dumps(analysis_result, indent=2)[:5000]}
````

## Key Insights

"""
for i, insight in enumerate(insights, 1):
report += f"{i}. {insight}\\n"

```
    report += "\n## Visualizations\n\n"
    for viz_path in visualizations:
        report += f"![Visualization]({viz_path})\n\n"

    # Save report
    report_path = self.working_dir / f"report_{request.analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Report saved to: {report_path}")
```

# Usage example

if __name__ == "__main__":
from langchain_openai import ChatOpenAI

```
# Initialize agent
llm = ChatOpenAI(model="gpt-4")
agent = DataAnalysisAgent(llm)

# Create analysis request
request = AnalysisRequest(
    dataset_path="sales_data.csv",
    analysis_type="descriptive",
    parameters={},
    output_format="report"
)

# Perform analysis
result = agent.analyze(request)

print(f"Analysis completed: {result.success}")
print(f"Insights: {result.insights}")
print(f"Visualizations created: {len(result.visualizations)}")
```

```

## Summary

These practical implementations demonstrate:

1. **ReAct Pattern**: Basic implementation with data analysis tools and StateAct improvements
2. **Plan-and-Execute**: LangGraph-based architecture with dynamic replanning
3. **Tree of Thoughts**: Search-based reasoning with multiple evaluation methods
4. **Memory Systems**: Comprehensive episodic, semantic, and procedural memory
5. **Multi-Agent Systems**: AutoGen-style orchestration with specialized agents
6. **Data Analysis Agent**: Production-ready agent with multiple analysis capabilities

Each implementation includes error handling, logging, and extensibility points for real-world usage. The code is designed to be modular and can be adapted for specific use cases and requirements.
```
