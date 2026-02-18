from insightface.app import FaceAnalysis

# Load lightweight recognition model
app = FaceAnalysis(
    name="buffalo_sc",          # small & fast
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# ctx_id = 0 → GPU, -1 → CPU
app.prepare(ctx_id=0, det_size=(112, 112))
