import subprocess, sys, os, json, pathlib

def test_pipeline_smoke():
    # Run minimal pipeline end-to-end
    subprocess.check_call([sys.executable, "-m", "src.data"])
    subprocess.check_call([sys.executable, "-m", "src.train"])
    subprocess.check_call([sys.executable, "-m", "src.evaluate"])
    # Check artifacts exist
    art = pathlib.Path("artifacts")
    assert (art/"models"/"model.joblib").exists()
    assert (art/"metrics"/"metrics.json").exists()
