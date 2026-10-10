# Shared GPU FIFO scheduler

The two experiment owners acquire one exclusive GPU lease per complete trial. CPU threads are capped at 4 and the allocator memory fraction at 0.8. Runtime queue state and process logs are local only. See scheduler.py and policy.json for the interface.
