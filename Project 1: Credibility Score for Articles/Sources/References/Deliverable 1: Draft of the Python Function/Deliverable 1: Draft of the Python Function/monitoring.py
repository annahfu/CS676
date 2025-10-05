from prometheus_fastapi_instrumentator import Instrumentator
def setup_instrumentation(app):
    Instrumentator().instrument(app).expose(app, endpoint='/metrics', include_in_schema=False)
