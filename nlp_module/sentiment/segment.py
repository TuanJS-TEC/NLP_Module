"""Vietnamese word segmentation for PhoBERT via the VnCoreNLP jar (JRE is enough; py_vncorenlp needs a JDK)."""
import subprocess
import tempfile
from pathlib import Path

JAR_DIR = Path(__file__).resolve().parents[2] / "data" / "vncorenlp"
_END = "ZZZENDZZZ"


def segment(texts):
    """-> list of word-segmented strings (multi-syllable words joined by '_'), same order as input."""
    with tempfile.TemporaryDirectory() as d:
        fin, fout = Path(d, "in.txt"), Path(d, "out.txt")
        fin.write_text("".join(" ".join(t.split()) + f"\n{_END}\n" for t in texts), encoding="utf-8")
        subprocess.run(["java", "-Xmx2g", "-jar", str(JAR_DIR / "VnCoreNLP-1.2.jar"), "-fin", str(fin),
                        "-fout", str(fout), "-annotators", "wseg"], check=True, capture_output=True, cwd=JAR_DIR)
        res, cur = [], []
        for line in fout.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            if parts[1] == _END:
                res.append(" ".join(cur))
                cur = []
            else:
                cur.append(parts[1])
    assert len(res) == len(texts), (len(res), len(texts))
    return res
