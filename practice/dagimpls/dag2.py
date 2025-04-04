import uuid
import os
import asyncio
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

class NodeStatus(Enum):
    """Represents the current status of a QNode."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CACHED = "CACHED"

class CacheRegistry:
    """Global registry for managing cache instances."""
    _instances: Dict[str, 'Cache'] = {}
    
    @classmethod
    def register(cls, cache: 'Cache') -> None:
        cls._instances[cache.uuid] = cache
    
    @classmethod
    def get(cls, uuid: str) -> Optional['Cache']:
        return cls._instances.get(uuid)

@dataclass
class Cache:
    """Represents a cache with a name and path for storing data."""
    name: str
    path: str
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        os.makedirs(self.path, exist_ok=True)
        CacheRegistry.register(self)
    
    def store(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        pass
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache."""
        pass

@dataclass
class QNode:
    """Represents a node in a workflow DAG."""
    name: str
    func: Callable
    cache: Optional[Cache] = None
    dependencies: List['QNode'] = field(default_factory=list)
    outputs: List['QNode'] = field(default_factory=list)
    status: NodeStatus = field(default=NodeStatus.PENDING)
    last_run: Optional[datetime] = None
    error: Optional[Exception] = None
    
    def add_dependency(self, node: 'QNode') -> None:
        if node not in self.dependencies:
            self.dependencies.append(node)
            node.outputs.append(self)
    
    async def run(self, *args, **kwargs) -> Any:
        """Execute the node's function, checking cache first if available."""
        self.status = NodeStatus.RUNNING
        try:
            if self.cache:
                cache_key = self._generate_cache_key(*args, **kwargs)
                result = self.cache.retrieve(cache_key)
                if result is not None:
                    self.status = NodeStatus.CACHED
                    return result
            
            for dep in self.dependencies:
                await dep.run(*args, **kwargs)
            
            result = await self.func(*args, **kwargs)
            
            if self.cache:
                self.cache.store(cache_key, result)
            
            self.status = NodeStatus.COMPLETED
            self.last_run = datetime.now()
            return result
            
        except Exception as e:
            self.status = NodeStatus.FAILED
            self.error = e
            raise
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        return str(uuid.uuid4())

class QWorkflow:
    """Represents a DAG of QNodes forming a workflow."""
    def __init__(self, name: str):
        self.name = name
        self.nodes: List[QNode] = []
        self.start_nodes: List[QNode] = []
        
    def add_node(self, node: QNode) -> None:
        if node not in self.nodes:
            self.nodes.append(node)
            if not node.dependencies:
                self.start_nodes.append(node)
    
    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        results = {}
        for node in self.start_nodes:
            results[node.name] = await node.run(*args, **kwargs)
        return results
    
    def validate(self) -> bool:
        visited = set()
        temp = set()
        
        def has_cycle(node: QNode) -> bool:
            if node in temp:
                return True
            if node in visited:
                return False
            
            temp.add(node)
            for dep in node.outputs:
                if has_cycle(dep):
                    return True
            temp.remove(node)
            visited.add(node)
            return False
        
        return not any(has_cycle(node) for node in self.nodes)

class QConductor:
    """Scheduler and orchestrator for QWorkflow execution."""
    def __init__(self):
        self.workflows: Dict[str, QWorkflow] = {}
        self.running: Set[str] = set()
        self.schedule: Dict[str, timedelta] = {}
        self.logger = logging.getLogger(__name__)
        self._shutdown = False
        self._tasks: List[asyncio.Task] = []
    
    def register_workflow(self, workflow: QWorkflow, interval: Optional[timedelta] = None) -> None:
        """Register a workflow with an optional execution interval."""
        if not workflow.validate():
            raise ValueError(f"Workflow {workflow.name} contains cycles")
        
        self.workflows[workflow.name] = workflow
        if interval:
            self.schedule[workflow.name] = interval
    
    async def start(self) -> None:
        """Start the conductor and begin scheduling workflows."""
        self._shutdown = False
        self.logger.info("Starting QConductor")
        
        # Start scheduled workflows
        for workflow_name, interval in self.schedule.items():
            self._tasks.append(
                asyncio.create_task(self._run_scheduled_workflow(workflow_name, interval))
            )
    
    async def stop(self) -> None:
        """Stop the conductor and all running workflows."""
        self._shutdown = True
        self.logger.info("Stopping QConductor")
        
        # Cancel all running tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
    
    async def _run_scheduled_workflow(self, workflow_name: str, interval: timedelta) -> None:
        """Run a workflow on a schedule."""
        while not self._shutdown:
            try:
                if workflow_name not in self.running:
                    self.running.add(workflow_name)
                    workflow = self.workflows[workflow_name]
                    
                    self.logger.info(f"Starting scheduled execution of workflow: {workflow_name}")
                    await workflow.run()
                    self.logger.info(f"Completed scheduled execution of workflow: {workflow_name}")
                    
                    self.running.remove(workflow_name)
                
                await asyncio.sleep(interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in workflow {workflow_name}: {str(e)}")
                if workflow_name in self.running:
                    self.running.remove(workflow_name)
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def trigger_workflow(self, workflow_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Manually trigger a workflow execution."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        if workflow_name in self.running:
            raise RuntimeError(f"Workflow {workflow_name} is already running")
        
        try:
            self.running.add(workflow_name)
            workflow = self.workflows[workflow_name]
            self.logger.info(f"Manually triggering workflow: {workflow_name}")
            results = await workflow.run(*args, **kwargs)
            self.logger.info(f"Completed manual execution of workflow: {workflow_name}")
            return results
        finally:
            self.running.remove(workflow_name)
    
    def get_workflow_status(self, workflow_name: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        workflow = self.workflows[workflow_name]
        return {
            'name': workflow_name,
            'running': workflow_name in self.running,
            'scheduled': workflow_name in self.schedule,
            'interval': self.schedule.get(workflow_name),
            'nodes': {
                node.name: {
                    'status': node.status.value,
                    'last_run': node.last_run,
                    'error': str(node.error) if node.error else None
                }
                for node in workflow.nodes
            }
        }