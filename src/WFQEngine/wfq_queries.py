# WFQ_CLAIM_LONGEST_IDLE_COMMAND = """
# WITH inflight AS (
#     SELECT context_id, count(*) AS n
#     FROM transcription_tasks
#     WHERE status = 'running'
#     GROUP BY context_id
# ),
# longest_idle_context AS (
#     SELECT context_id, GREATEST(max(finished_at), max(started_at)) AS last_done
#     FROM transcription_tasks
#     WHERE status = 'done'
#     GROUP BY context_id
# ),
# done AS (
#     SELECT context_id, count(*) AS n
#     FROM transcription_tasks
#     WHERE status = 'done'
#     GROUP BY context_id
# )
# UPDATE transcription_tasks t
# SET status = 'running', started_at = now()
# WHERE t.id = (
#     SELECT t2.id
#     FROM transcription_tasks t2
#     LEFT JOIN longest_idle_context lic ON lic.context_id = t2.context_id
#     WHERE t2.status = 'pending'
#     ORDER BY
#         lic.last_done ASC NULLS FIRST,  -- last idle context queueing
#         t2.created_at ASC             -- then oldest job (aging / anti-starvation)
#     FOR UPDATE OF t2 SKIP LOCKED
#     LIMIT 1
# )
# RETURNING t.id, t.context_id, t.file_path;"""

#t2.priority DESC,             -- optional: paid tiers cut ahead
WFQ_LONGEST_IDLE_CLAIM = """
WITH longest_idle_context AS (
    SELECT context_id, GREATEST(max(finished_at), max(started_at)) AS last_done
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
        lic.last_done ASC NULLS FIRST,  -- last idle context queueing
        t2.created_at ASC             -- then oldest job (aging / anti-starvation)
    FOR UPDATE OF t2 SKIP LOCKED
    LIMIT 1
)
RETURNING t.id, t.context_id, t.payload;"""


FIFO_CLAIM = """"""
