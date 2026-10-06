"""客户端取消：下游任务停止 + 唯一 cancelled 终态。"""
import pytest

from conftest import FakeProvider
from llm_gateway.core.models import LLMRequest, Message


@pytest.mark.asyncio
async def test_stream_cancel_records_single_cancelled_terminal(app_factory):
    provider = FakeProvider(outcomes=[{"delta": "hello"}, {"delta": "world"}])
    app = app_factory(provider=provider)
    service = app.state.llm_service

    request = LLMRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )
    generator = service.stream(request)
    first = await generator.__anext__()
    assert first == {"type": "content.delta", "delta": "hello"}

    # 模拟客户端断开：aclose → GeneratorExit 注入生成器
    await generator.aclose()

    # 唯一终态：只有一条 cancelled trace，无 success/failed
    traces = await app.state.ledger.recent()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.status == "cancelled"
    assert trace.error_code == "client_disconnected"
    assert trace.attempts == 1

    # 取消传播：下游（FakeProvider 流生成器）被显式关闭
    assert provider.stream_closed is True


@pytest.mark.asyncio
async def test_stream_normal_completion_is_single_success(app_factory):
    provider = FakeProvider()
    app = app_factory(provider=provider)
    service = app.state.llm_service

    request = LLMRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )
    events = [event async for event in service.stream(request)]
    assert events[-1] == {"type": "response.completed", "model": "test-model"}
    traces = await app.state.ledger.recent()
    assert len(traces) == 1
    assert traces[0].status == "success"
    assert provider.stream_closed is True
