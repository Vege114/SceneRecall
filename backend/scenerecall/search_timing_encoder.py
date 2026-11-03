"""Search timing adapter that leaves the encoder's implementation identity intact.

The base module is fingerprinted as part of every persisted vector namespace.
Observability lives here so adding timings never invalidates actual embeddings.
"""
from contextvars import ContextVar
from time import perf_counter

from .search_timing import record_interval, timed, trace_span
from .visual_encoder import VisualEncoder


_preparation_span = ContextVar("encoder_timing_preparation_span", default=None)
_exchange_started = ContextVar("encoder_timing_exchange_started", default=None)
_worker_started = ContextVar("encoder_timing_worker_started", default=False)


class TimedVisualEncoder(VisualEncoder):
    @timed("encoder.identity.work")
    def identity(self):
        return super().identity()

    @timed("encoder.request.work")
    def _request(self, *args, **kwargs):
        exchange = _exchange_started.set(None)
        worker = _worker_started.set(False)
        try:
            with trace_span("encoder.prepare_and_wait") as preparation:
                token = _preparation_span.set(preparation)
                try:
                    return super()._request(*args, **kwargs)
                finally:
                    _preparation_span.reset(token)
        finally:
            _worker_started.reset(worker)
            _exchange_started.reset(exchange)

    def _start(self, config, identity):
        preparation = _preparation_span.get()
        if preparation is not None:
            preparation.__exit__(None, None, None)
        with trace_span("encoder.worker_start") as phase:
            previous = self._process
            process = super()._start(config, identity)
            new = process is not previous
            _worker_started.set(new)
            phase.set(worker_started=new)
        _exchange_started.set(perf_counter())
        return process

    def _vectors(self, value, count):
        # The base calls this after assigning this request's reply metadata and
        # before releasing its request lock. A concurrent call cannot overwrite it.
        started = _exchange_started.get()
        if started is not None:
            new = _worker_started.get()
            record_interval("encoder.worker_exchange", started, worker_started=new,
                            worker_model_load_ms=self.metadata.get("model_load_ms") if new else 0,
                            worker_inference_ms=self.metadata.get("inference_ms"), item_count=count)
        return super()._vectors(value, count)
