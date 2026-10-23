"""Existing search scenarios wait for the new explicit readiness contract."""
import time


def wait_for_search(client):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get("/api/bootstrap")
        assert response.status_code == 200, response.text
        state = response.json()["startup"]["index"]
        assert state["status"] != "error", state
        if state["status"] == "ready":
            return
        time.sleep(.01)
    raise AssertionError("Search startup did not finish")
