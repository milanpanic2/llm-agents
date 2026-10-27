## Nvidia metrics helper app

A simple app used to expose prometheus formatted metrics from a local
`nvidia-smi` command results showing important GPU metrics inside.
It is used alongside EndpointSlice in kubernetes that exposes the app's
host and port inside kubernetes, so grafana can use it for showing the 
following metrics:

1. GPU compute utilization
2. VRAM used
3. VRAM total
4. Power draw
5. Power limit
6. Temperature