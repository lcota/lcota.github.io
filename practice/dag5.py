from datetime import datetime
from typing import Any, Dict, List, Optional, Iterable, TypeVar, Union
import ray
import pandas as pd
import json
from pathlib import Path
import os

T = TypeVar('T')

@ray.remote
class DistributedCache:
    """Distributed cache implementation using Ray."""
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = Path(path)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.path.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache data
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache data from disk."""
        cache_file = self.path / f"{self.name}_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
            except json.JSONDecodeError:
                self.cache = {}
    
    def _save_cache(self) -> None:
        """Save cache data to disk."""
        cache_file = self.path / f"{self.name}_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def store(self, key: str, value: Any) -> None:
        """Store a value with metadata in the cache."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'key': key,
            'value': value,
            'attributes': {}  # Extensible metadata dictionary
        }
        
        # Handle different types of values
        if isinstance(value, pd.DataFrame):
            # Store DataFrame as parquet
            data_file = self.path / f"{key}.parquet"
            value.to_parquet(data_file)
            metadata['value'] = str(data_file)
            metadata['type'] = 'dataframe'
        else:
            try:
                # Attempt to serialize the value
                json.dumps(value)
                metadata['type'] = 'json'
            except (TypeError, OverflowError):
                raise ValueError(f"Unsupported value type for caching: {type(value)}")
        
        self.cache[key] = metadata
        self._save_cache()
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache."""
        metadata = self.cache.get(key)
        if not metadata:
            return None
        
        if metadata['type'] == 'dataframe':
            return pd.read_parquet(metadata['value'])
        else:
            return metadata['value']
    
    def read(self, 
             sdate: Optional[datetime] = None,
             edate: Optional[datetime] = None,
             filter_attrs: Optional[List[Iterable[Any]]] = None) -> Dict[str, Any]:
        """
        Read cache entries within a date range and/or matching filter criteria.
        
        Args:
            sdate: Start date for filtering (inclusive)
            edate: End date for filtering (inclusive)
            filter_attrs: List of iterables containing filter criteria
                        Each iterable represents an OR condition
                        Multiple iterables represent AND conditions
        
        Returns:
            Dict of matching cache entries
        """
        results = {}
        
        for key, metadata in self.cache.items():
            timestamp = datetime.fromisoformat(metadata['timestamp'])
            
            # Date range filtering
            if sdate and timestamp < sdate:
                continue
            if edate and timestamp > edate:
                continue
            
            # Attribute filtering
            if filter_attrs:
                matches_all = True
                for filter_set in filter_attrs:
                    # Convert single values to iterable
                    if not isinstance(filter_set, (list, tuple, set)):
                        filter_set = [filter_set]
                    
                    # Check if any value in the filter set matches
                    matches_any = False
                    for filter_value in filter_set:
                        if isinstance(filter_value, (str, int, float)):
                            # Direct value comparison
                            if metadata['value'] == filter_value:
                                matches_any = True
                                break
                        elif callable(filter_value):
                            # Function predicate
                            try:
                                if filter_value(metadata['value']):
                                    matches_any = True
                                    break
                            except Exception:
                                continue
                    
                    if not matches_any:
                        matches_all = False
                        break
                
                if not matches_all:
                    continue
            
            # If all filters pass, add to results
            results[key] = self.retrieve(key)
        
        return results

class Cache:
    """Local interface for distributed cache."""
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.uuid = str(uuid.uuid4())
        self._actor: Optional[ActorHandle] = None
    
    def initialize(self) -> None:
        """Initialize the distributed cache actor."""
        self._actor = DistributedCache.remote(self.name, self.path)
    
    def store(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        ray.get(self._actor.store.remote(key, value))
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache."""
        return ray.get(self._actor.retrieve.remote(key))
    
    def read(self,
            sdate: Optional[datetime] = None,
            edate: Optional[datetime] = None,
            filter_attrs: Optional[List[Iterable[Any]]] = None) -> Dict[str, Any]:
        """
        Read cache entries within a date range and/or matching filter criteria.
        
        Args:
            sdate: Start date for filtering (inclusive)
            edate: End date for filtering (inclusive)
            filter_attrs: List of iterables containing filter criteria
                        Each iterable represents an OR condition
                        Multiple iterables represent AND conditions
        
        Example:
            # Read all entries between two dates
            cache.read(
                sdate=datetime(2024, 1, 1),
                edate=datetime(2024, 12, 31)
            )
            
            # Read entries with specific values
            cache.read(filter_attrs=[[1, 2, 3]])  # Match any of these values
            
            # Read entries with multiple conditions
            cache.read(filter_attrs=[
                [1, 2, 3],           # Value must be 1, 2, or 3 AND
                [lambda x: x > 0]     # Value must be positive
            ])
            
            # Combine date range and filters
            cache.read(
                sdate=datetime(2024, 1, 1),
                filter_attrs=[[1, 2, 3]]
            )
        
        Returns:
            Dict of matching cache entries
        """
        return ray.get(self._actor.read.remote(sdate, edate, filter_attrs))

# Example usage:
if __name__ == "__main__":
    # Create cache
    cache = Cache("example_cache", "/tmp/cache")
    cache.initialize()
    
    # Store some values
    cache.store("key1", {"value": 1, "tags": ["a", "b"]})
    cache.store("key2", {"value": 2, "tags": ["b", "c"]})
    
    # Read with filters
    results = cache.read(
        sdate=datetime(2024, 1, 1),
        filter_attrs=[
            [1, 2],  # Match values 1 or 2
            [lambda x: "a" in x["tags"]]  # Must have tag "a"
        ]
    )import uuid
import os
import ray
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from ray.actor import ActorHandle
from ray.util.queue import Queue

class NodeStatus(Enum):
    """Represents the current status of a QNode."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CACHED = "CACHED"

class ClusterConfig:
    """Configuration for Ray cluster connection."""
    def __init__(self, 
                 cluster_type: str = "local",
                 address: Optional[str] = None,
                 namespace: str = "default",
                 runtime_env: Optional[Dict] = None):
        self.cluster_type = cluster_type
        self.address = address
        self.namespace = namespace
        self.runtime_env = runtime_env or {}

@ray.remote
class DistributedCache:
    """Distributed cache implementation using Ray."""
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.cache: Dict[str, Any] = {}
        os.makedirs(path, exist_ok=True)
    
    def store(self, key: str, value: Any) -> None:
        self.cache[key] = value
        
    def retrieve(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

class Cache:
    """Local interface for distributed cache."""
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.uuid = str(uuid.uuid4())
        self._actor: Optional[ActorHandle] = None
    
    def initialize(self) -> None:
        """Initialize the distributed cache actor."""
        self._actor = DistributedCache.remote(self.name, self.path)
    
    def store(self, key: str, value: Any) -> None:
        ray.get(self._actor.store.remote(key, value))
    
    def retrieve(self, key: str) -> Optional[Any]:
        return ray.get(self._actor.retrieve.remote(key))

@ray.remote
class DistributedQNode:
    """Distributed implementation of workflow node using Ray."""
    def __init__(self, name: str, func: Callable, cache: Optional[Cache] = None):
        self.name = name
        self.func = func
        self.cache = cache
        self.status = NodeStatus.PENDING
        self.last_run = None
        self.error = None
    
    def run(self, *args, **kwargs) -> Any:
        self.status = NodeStatus.RUNNING
        try:
            if self.cache:
                cache_key = str(uuid.uuid4())  # Simplified cache key generation
                result = self.cache.retrieve(cache_key)
                if result is not None:
                    self.status = NodeStatus.CACHED
                    return result
            
            result = self.func(*args, **kwargs)
            
            if self.cache:
                self.cache.store(cache_key, result)
            
            self.status = NodeStatus.COMPLETED
            self.last_run = datetime.now()
            return result
            
        except Exception as e:
            self.status = NodeStatus.FAILED
            self.error = str(e)
            raise

class QNode:
    """Local interface for distributed QNode."""
    def __init__(self, name: str, func: Callable, cache: Optional[Cache] = None):
        self.name = name
        self.func = func
        self.cache = cache
        self.dependencies: List['QNode'] = []
        self.outputs: List['QNode'] = []
        self._actor: Optional[ActorHandle] = None
    
    def initialize(self) -> None:
        """Initialize the distributed node actor."""
        self._actor = DistributedQNode.remote(self.name, self.func, self.cache)
    
    def add_dependency(self, node: 'QNode') -> None:
        if node not in self.dependencies:
            self.dependencies.append(node)
            node.outputs.append(self)

@ray.remote
class WorkflowExecutor:
    """Ray actor for executing workflows."""
    def __init__(self):
        self.running_workflows: Set[str] = set()
    
    def execute_workflow(self, nodes: List[ActorHandle], *args, **kwargs) -> Dict[str, Any]:
        """Execute a workflow by running its nodes in dependency order."""
        results = {}
        deps_completed = set()
        
        while len(deps_completed) < len(nodes):
            futures = []
            
            for node in nodes:
                node_ref = ray.get(node.get_info.remote())
                if (node_ref['name'] not in deps_completed and 
                    all(dep in deps_completed for dep in node_ref['dependencies'])):
                    futures.append(node.run.remote(*args, **kwargs))
            
            if futures:
                done_refs = ray.get(futures)
                for node_ref, result in zip(nodes, done_refs):
                    node_name = ray.get(node_ref.get_info.remote())['name']
                    results[node_name] = result
                    deps_completed.add(node_name)
        
        return results

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
    """Distributed scheduler and orchestrator using Ray."""
    def __init__(self, cluster_config: Optional[ClusterConfig] = None):
        self.cluster_config = cluster_config or ClusterConfig()
        self.workflows: Dict[str, QWorkflow] = {}
        self.schedule: Dict[str, timedelta] = {}
        self.logger = logging.getLogger(__name__)
        self._executor: Optional[ActorHandle] = None
        self._scheduler_queue: Optional[Queue] = None
    
    def initialize(self) -> None:
        """Initialize Ray and create required actors."""
        # Initialize Ray based on cluster configuration
        if not ray.is_initialized():
            if self.cluster_config.cluster_type == "local":
                ray.init(namespace=self.cluster_config.namespace,
                        runtime_env=self.cluster_config.runtime_env)
            else:
                ray.init(address=self.cluster_config.address,
                        namespace=self.cluster_config.namespace,
                        runtime_env=self.cluster_config.runtime_env)
        
        # Initialize executor and scheduler queue
        self._executor = WorkflowExecutor.remote()
        self._scheduler_queue = Queue()
        
        # Initialize all nodes and caches
        for workflow in self.workflows.values():
            for node in workflow.nodes:
                if node.cache:
                    node.cache.initialize()
                node.initialize()
    
    def register_workflow(self, workflow: QWorkflow, interval: Optional[timedelta] = None) -> None:
        """Register a workflow with an optional execution interval."""
        if not workflow.validate():
            raise ValueError(f"Workflow {workflow.name} contains cycles")
        
        self.workflows[workflow.name] = workflow
        if interval:
            self.schedule[workflow.name] = interval
    
    def start(self) -> None:
        """Start the conductor and begin scheduling workflows."""
        if not ray.is_initialized():
            self.initialize()
        
        self.logger.info("Starting QConductor")
        
        # Start scheduler actor for periodic workflows
        @ray.remote
        def scheduler_loop():
            while True:
                for workflow_name, interval in self.schedule.items():
                    workflow = self.workflows[workflow_name]
                    ray.get(self._executor.execute_workflow.remote(
                        [node._actor for node in workflow.nodes]
                    ))
                    ray.sleep(interval.total_seconds())
        
        ray.get(scheduler_loop.remote())
    
    def stop(self) -> None:
        """Stop the conductor and shut down Ray."""
        self.logger.info("Stopping QConductor")
        if ray.is_initialized():
            ray.shutdown()
    
    def trigger_workflow(self, workflow_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Manually trigger a workflow execution."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        workflow = self.workflows[workflow_name]
        return ray.get(self._executor.execute_workflow.remote(
            [node._actor for node in workflow.nodes],
            *args, **kwargs
        ))
    
    def get_workflow_status(self, workflow_name: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        workflow = self.workflows[workflow_name]
        return {
            'name': workflow_name,
            'scheduled': workflow_name in self.schedule,
            'interval': self.schedule.get(workflow_name),
            'nodes': {
                node.name: ray.get(node._actor.get_status.remote())
                for node in workflow.nodes
            }
        }

# Example usage:
if __name__ == "__main__":
    # Local cluster configuration
    local_config = ClusterConfig(
        cluster_type="local",
        namespace="example",
        runtime_env={"pip": ["numpy", "pandas"]}
    )
    
    # Remote cluster configuration
    remote_config = ClusterConfig(
        cluster_type="remote",
        address="ray://192.168.1.100:10001",
        namespace="production",
        runtime_env={"pip": ["numpy", "pandas"]}
    )
    
    # Create conductor with chosen configuration
    conductor = QConductor(local_config)  # or remote_config
    
    # Define a sample task
    def process_data(data):
        return data * 2
    
    # Create nodes and workflow
    cache = Cache("example_cache", "/tmp/cache")
    node1 = QNode("process", process_data, cache)
    
    workflow = QWorkflow("example_workflow")
    workflow.add_node(node1)
    
    # Register and run workflow
    conductor.register_workflow(workflow, interval=timedelta(minutes=5))
    conductor.initialize()
    conductor.start()
    print(results)  # Will return matching entries