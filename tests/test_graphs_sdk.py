import tempfile
import pytest
from unittest.mock import MagicMock
from aiknow import AIKnowClient
from aiknow_contracts.graph_models import GraphType

def test_graphs_sdk_register_from_yaml():
    yaml_content = """
    graph_id: test_yaml_flow
    version: 1.0.0
    entry_node_id: start
    nodes:
      start:
        node_id: start
        behavior: WAIT
        prompt: "Hello from YAML"
    edges: []
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp.flush()
        
        # Initialize client and mock transport/client post
        client = AIKnowClient(api_key="mock_key")
        client._client.post = MagicMock()
        mock_res = MagicMock()
        mock_res.status_code = 201
        mock_res.json.return_value = {
            "status": "success",
            "graph_id": "test_yaml_flow",
            "type": "conversation_graph"
        }
        client._client.post.return_value = mock_res
        
        res = client.graphs.register_from_yaml(tmp.name, tenant_id="test-tenant")
        assert res["status"] == "success"
        assert res["graph_id"] == "test_yaml_flow"
        
        # Assert client called the endpoint correctly
        client._client.post.assert_called_once()
        args, kwargs = client._client.post.call_args
        assert args[0] == "/graphs"
        assert kwargs["params"] == {
            "graph_type": GraphType.CONVERSATION_GRAPH.value,
            "tenant_id": "test-tenant"
        }


@pytest.mark.asyncio
async def test_async_graphs_sdk_register_from_yaml():
    yaml_content = """
    graph_id: test_yaml_flow_async
    version: 1.0.0
    entry_node_id: start
    nodes:
      start:
        node_id: start
        behavior: WAIT
        prompt: "Hello from Async YAML"
    edges: []
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp.flush()
        
        from aiknow import AsyncAIKnowClient
        from unittest.mock import AsyncMock
        
        client = AsyncAIKnowClient(api_key="mock_key")
        client._client.post = AsyncMock()
        mock_res = MagicMock()
        mock_res.status_code = 201
        mock_res.json.return_value = {
            "status": "success",
            "graph_id": "test_yaml_flow_async",
            "type": "conversation_graph"
        }
        client._client.post.return_value = mock_res
        
        res = await client.graphs.register_from_yaml(tmp.name, tenant_id="test-tenant")
        assert res["status"] == "success"
        assert res["graph_id"] == "test_yaml_flow_async"
        
        # Assert client called the endpoint correctly
        client._client.post.assert_called_once()
        args, kwargs = client._client.post.call_args
        assert args[0] == "/graphs"
        assert kwargs["params"] == {
            "graph_type": GraphType.CONVERSATION_GRAPH.value,
            "tenant_id": "test-tenant"
        }
