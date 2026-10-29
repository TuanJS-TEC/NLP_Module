"""Sequential near-duplicate clustering. Processed in published_at order; a later article never changes an
earlier article's cluster, so a historical cutoff sees the same groups it would have seen live."""
import re
import uuid
from datetime import datetime, timedelta

THRESHOLD = 0.8                 # Jaccard on word 5-shingles: >= repost; below = different analysis of same event
WINDOW = timedelta(days=7)


def shingles(text: str, k=5) -> frozenset:
    w = re.findall(r"\w+", text.lower())
    return frozenset(hash(" ".join(w[i:i + k])) for i in range(max(1, len(w) - k + 1)))


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else datetime.max


def assign_clusters(c, threshold=THRESHOLD):
    """Assign duplicate_cluster_id / is_representative to versions that have none. Returns #assigned."""
    rows = c.execute("""SELECT document_version_id vid, title, normalized_text txt, published_at pub,
                        version_first_seen_at seen, duplicate_cluster_id cid FROM document_version
                        ORDER BY COALESCE(published_at, version_first_seen_at), version_first_seen_at""").fetchall()
    done, n = [], 0   # done: (time, shingles, cluster)
    for r in rows:
        t = _dt(r["pub"] or r["seen"])
        sh = shingles(f"{r['title']} {r['txt']}")
        if r["cid"]:
            done.append((t, sh, r["cid"]))
            continue
        done = [d for d in done if t - d[0] <= WINDOW]
        best = max(((jaccard(sh, d[1]), d[2]) for d in done), default=(0, None))
        cid, rep = (best[1], 0) if best[0] >= threshold else (r["vid"], 1)
        c.execute("UPDATE document_version SET duplicate_cluster_id=?, is_representative=? WHERE document_version_id=?",
                  (cid, rep, r["vid"]))
        done.append((t, sh, cid))
        n += 1
    c.commit()
    return n
