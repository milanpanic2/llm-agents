#t2.priority DESC,             -- optional: paid tiers cut ahead
WFQ_LONGEST_IDLE_CLAIM = """
WITH longest_idle_context AS (
    SELECT context_id, max(started_at) AS last_done
    FROM transcription_tasks
    WHERE status = 'done'
    GROUP BY context_id
)
UPDATE {table_name} t
SET status = 'running', started_at = now()
WHERE t.id = (
    SELECT t2.id
    FROM {table_name} t2
    LEFT JOIN longest_idle_context lic ON lic.context_id = t2.context_id
    WHERE t2.status = 'pending'
    ORDER BY
        lic.last_done ASC NULLS FIRST,
        t2.created_at ASC
    FOR UPDATE OF t2 SKIP LOCKED
    LIMIT 1
)
RETURNING t.id, t.context_id, t.payload;"""


WFQ_FIFO_CLAIM = """
UPDATE {table_name} t
SET status = 'running', started_at = now()
WHERE t.id = (
    SELECT t2.id
    FROM {table_name} t2
    WHERE t2.status = 'pending'
    ORDER BY
        t2.created_at ASC
    FOR UPDATE OF t2 SKIP LOCKED
    LIMIT 1
)
RETURNING t.id, t.context_id, t.payload;"""
