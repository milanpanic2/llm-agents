#t2.priority DESC,             -- optional: paid tiers cut ahead
WFQ_LONGEST_IDLE_CLAIM = """
WITH next_task AS (
    SELECT t1.id
    FROM {table_name} t1
    JOIN LATERAL (
        SELECT max(t2.started_at) AS longest_idle
        FROM {table_name} t2
        WHERE t2.context_id = t1.context_id
    ) lic ON true
    WHERE t1.status = 'pending'
    ORDER BY
        lic.longest_idle ASC NULLS FIRST,
        t1.created_at ASC
    FOR UPDATE OF t1 SKIP LOCKED
    LIMIT 1
)
UPDATE {table_name} t
SET status = 'running', started_at = now()
FROM next_task
WHERE t.id = next_task.id
RETURNING t.*;
"""


WFQ_FIFO_CLAIM = """
WITH next_task AS (
    SELECT t1.id
    FROM {table_name} t1
    WHERE t1.status = 'pending'
    ORDER BY t1.created_at ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE {table_name} t
SET status = 'running', started_at = now()
FROM next_task
WHERE t.id = next_task.id
RETURNING t.*;
"""
