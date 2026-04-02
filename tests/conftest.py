# conftest.py
def pytest_collection_modifyitems(config, items):
    for item in items:
        # Only apply to parametrized test_shapes calls
        if hasattr(item, "callspec"):
            case = item.callspec.params.get("case", None)
            if case is None:
                continue

            # Attach markers before pytest selects tests
            for m in case.get("markers", []):
                item.add_marker(m)