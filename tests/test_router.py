from fastapi.testclient import TestClient
from app.thinking_router import app

client = TestClient(app)

def test_simple_query_routes_to_no_think():
    resp = client.post("/v1/route", json={"prompt": "Summarize this email in 2 bullet points."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["routed_mode"] == "REASONING_OFF"
    assert data["injected_system_prompt"] == "/no_think"
    assert data["upstream_payload"]["messages"][0]["content"] == "/no_think"

def test_math_query_routes_to_detailed_thinking():
    resp = client.post("/v1/route", json={"prompt": "Calculate the roofline ridge point and prove step by step why B=1 is bandwidth bound."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["routed_mode"] == "REASONING_ON"
    assert data["injected_system_prompt"] == "detailed thinking on"

def test_prometheus_metrics_increment():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert 'nemotron_router_requests_total{mode="reasoning_off"} 1' in text
    assert 'nemotron_router_requests_total{mode="reasoning_on"} 1' in text
    assert 'nemotron_router_estimated_tokens_saved_total 250' in text
