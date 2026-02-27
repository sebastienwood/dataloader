import os, time

def test_monitor():
    os.environ.setdefault("SLURM_JOB_ID", "test")
    from dino_loader.monitor.metrics import init_registry, get_registry

    init_registry(job_id="test", create=True, local_rank=0)
    reg = get_registry()

    reg.inc("loader_batches_yielded", 42)
    reg.inc("lustre_bytes_read", 1024 * 1024 * 500)   # 500 MB
    reg.set("shard_cache_utilization_pct", 67.3)
    reg.heartbeat()

    # Simulate a second rank reading
    reader = __import__("dino_loader.monitor.metrics", fromlist=["MetricsRegistry"])
    r2 = reader.MetricsRegistry(job_id="test", create=False, local_rank=0)
    m  = r2.read_all_ranks().ranks[0]
    assert m.loader_batches_yielded == 42, f"Got {m.loader_batches_yielded}"
    assert m.lustre_bytes_read == 1024*1024*500
    assert abs(m.shard_cache_utilization_pct - 67.3) < 0.01
    assert m.heartbeat_ts > 0
    r2.close()
    reg.unlink()