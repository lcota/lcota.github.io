import uuid
import os
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time

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
    _lock = threading.Lock()
    
    @classmethod
    def register(cls, cache: 'Cache') -> None:
        with cls._lock:
            cls._instances[cache.uuid] = cache
    
    @classmethod
    def get(cls, uuid: str) -> Optional['Cache']:
        with cls._lock:
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
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_dependency(self, node: 'QNode') -> None:
        with self._lock:
            if node not in self.dependencies:
                self.dependencies.append(node)
                node.outputs.append(self)
    
    def run(self, *args, **kwargs) -> Any:
        """Execute the node's function, checking cache first if available."""
        with self._lock:
            self.status = NodeStatus.RUNNING
        
        try:
            if self.cache:
                cache_key = self._generate_cache_key(*args, **kwargs)
                result = self.cache.retrieve(cache_key)
                if result is not None:
                    with self._lock:
                        self.status = NodeStatus.CACHED
                    return result
            
            # Execute dependencies first
            for dep in self.dependencies:
                dep.run(*args, **kwargs)
            
            # Execute this node's function
            result = self.func(*args, **kwargs)
            
            if self.cache:
                self.cache.store(cache_key, result)
            
            with self._lock:
                self.status = NodeStatus.COMPLETED
                self.last_run = datetime.now()
            
            return result
            
        except Exception as e:
            with self._lock:
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
        self._lock = threading.Lock()
        
    def add_node(self, node: QNode) -> None:
        with self._lock:
            if node not in self.nodes:
                self.nodes.append(node)
                if not node.dependencies:
                    self.start_nodes.append(node)
    
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        results = {}
        for node in self.start_nodes:
            results[node.name] = node.run(*args, **kwargs)
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
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
    
    def register_workflow(self, workflow: QWorkflow, interval: Optional[timedelta] = None) -> None:
        """Register a workflow with an optional execution interval."""
        if not workflow.validate():
            raise ValueError(f"Workflow {workflow.name} contains cycles")
        
        with self._lock:
            self.workflows[workflow.name] = workflow
            if interval:
                self.schedule[workflow.name] = interval
    
    def start(self) -> None:
        """Start the conductor and begin scheduling workflows."""
        with self._lock:
            self._shutdown = False
        
        self.logger.info("Starting QConductor")
        
        # Start scheduled workflows
        for workflow_name, interval in self.schedule.items():
            thread = threading.Thread(
                target=self._run_scheduled_workflow,
                args=(workflow_name, interval),
                daemon=True
            )
            self._threads.append(thread)
            thread.start()
    
    def stop(self) -> None:
        """Stop the conductor and all running workflows."""
        self.logger.info("Stopping QConductor")
        
        with self._lock:
            self._shutdown = True
        
        # Wait for all threads to complete
        for thread in self._threads:
            thread.join()
        self._threads.clear()
    
    def _run_scheduled_workflow(self, workflow_name: str, interval: timedelta) -> None:
        """Run a workflow on a schedule."""
        while not self._shutdown:
            try:
                with self._lock:
                    if workflow_name not in self.running:
                        self.running.add(workflow_name)
                        workflow = self.workflows[workflow_name]
                
                try:
                    self.logger.info(f"Starting scheduled execution of workflow: {workflow_name}")
                    workflow.run()
                    self.logger.info(f"Completed scheduled execution of workflow: {workflow_name}")
                finally:
                    with self._lock:
                        self.running.remove(workflow_name)
                
                time.sleep(interval.total_seconds())
                
            except Exception as e:
                self.logger.error(f"Error in workflow {workflow_name}: {str(e)}")
                time.sleep(1)  # Brief pause before retrying
    
    def trigger_workflow(self, workflow_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Manually trigger a workflow execution."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        with self._lock:
            if workflow_name in self.running:
                raise RuntimeError(f"Workflow {workflow_name} is already running")
            self.running.add(workflow_name)
        
        try:
            workflow = self.workflows[workflow_name]
            self.logger.info(f"Manually triggering workflow: {workflow_name}")
            results = workflow.run(*args, **kwargs)
            self.logger.info(f"Completed manual execution of workflow: {workflow_name}")
            return results
        finally:
            with self._lock:
                self.running.remove(workflow_name)
    
    def get_workflow_status(self, workflow_name: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        workflow = self.workflows[workflow_name]
        with self._lock:
            running = workflow_name in self.running
            scheduled = workflow_name in self.schedule
            interval = self.schedule.get(workflow_name)
        
        return {
            'name': workflow_name,
            'running': running,
            'scheduled': scheduled,
            'interval': interval,
            'nodes': {
                node.name: {
                    'status': node.status.value,
                    'last_run': node.last_run,
                    'error': str(node.error) if node.error else None
                }
                for node in workflow.nodes
            }
        }