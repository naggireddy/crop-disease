"""
Agronomist Review Portal & Dashboard
Interactive web interface for agronomists to audit uncertain predictions,
inspect Grad-CAM lesion attributions, submit verified ground-truth labels,
and monitor operational model drift.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

dashboard_router = APIRouter()

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agronomist Review & Drift Monitoring Portal</title>
  <style>
    :root {
      --primary: #2e7d32;
      --primary-dark: #1b5e20;
      --accent: #ff9800;
      --bg: #f4f6f8;
      --card-bg: #ffffff;
      --text: #212121;
      --text-muted: #666666;
      --border: #e0e0e0;
      --danger: #d32f2f;
      --success: #388e3c;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding-bottom: 40px; }
    header { background: var(--primary-dark); color: white; padding: 1.2rem 2rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    header h1 { font-size: 1.4rem; font-weight: 600; display: flex; align-items: center; gap: 10px; }
    .badge { background: #81c784; color: #1b5e20; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    
    .container { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
    .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
    .kpi-card { background: var(--card-bg); padding: 1.2rem; border-radius: 8px; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .kpi-card h3 { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem; }
    .kpi-card .val { font-size: 1.8rem; font-weight: 700; color: var(--primary); }
    
    .section-title { font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }
    .refresh-btn { background: white; border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
    .refresh-btn:hover { background: #e8f5e9; }
    
    .queue-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 1.5rem; }
    .case-card { background: var(--card-bg); border-radius: 8px; border: 1px solid var(--border); overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.06); display: flex; flex-direction: column; }
    .case-media { display: flex; height: 180px; background: #eee; border-bottom: 1px solid var(--border); }
    .case-media img { width: 50%; height: 100%; object-fit: cover; }
    .case-media .img-label { position: absolute; background: rgba(0,0,0,0.6); color: white; font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; }
    
    .case-body { padding: 1rem; flex: 1; display: flex; flex-direction: column; gap: 0.6rem; }
    .case-title { font-size: 1rem; font-weight: 600; color: #d32f2f; }
    .meta-chip { display: inline-block; background: #e0f2f1; color: #004d40; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; margin-right: 4px; margin-bottom: 4px; }
    .conf-bar { background: #e0e0e0; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 4px; }
    .conf-fill { background: var(--accent); height: 100%; }
    
    .action-panel { background: #fafafa; border-top: 1px solid var(--border); padding: 0.8rem 1rem; }
    .btn-group { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
    .btn { flex: 1; padding: 8px; border: none; border-radius: 4px; font-size: 0.8rem; font-weight: 600; cursor: pointer; text-align: center; }
    .btn-correct { background: var(--success); color: white; }
    .btn-correct:hover { background: #2e7d32; }
    .btn-incorrect { background: var(--danger); color: white; }
    .btn-incorrect:hover { background: #b71c1c; }
    .btn-more { background: #757575; color: white; }
    .notes-input { width: 100%; padding: 6px; border: 1px solid var(--border); border-radius: 4px; font-size: 0.8rem; margin-bottom: 0.5rem; }
    
    .empty-state { text-align: center; padding: 3rem; background: white; border-radius: 8px; border: 1px dashed var(--border); color: var(--text-muted); }
  </style>
</head>
<body>
  <header>
    <h1>🌿 Agronomist Review & Drift Portal <span class="badge">20 Crops Active</span></h1>
    <div>
      <span style="font-size: 0.85rem; opacity: 0.9;">Model Version: <strong>v1.0.0</strong></span>
    </div>
  </header>

  <div class="container">
    <div class="kpi-row">
      <div class="kpi-card">
        <h3>Pending Review Queue</h3>
        <div class="val" id="kpi-queue-count">0</div>
      </div>
      <div class="kpi-card">
        <h3>Average Confidence</h3>
        <div class="val" id="kpi-avg-conf">--%</div>
      </div>
      <div class="kpi-card">
        <h3>OOD Rejection Rate</h3>
        <div class="val" id="kpi-rejection-rate">--%</div>
      </div>
      <div class="kpi-card">
        <h3>Retraining Trigger</h3>
        <div class="val" id="kpi-retrain-status" style="font-size: 1.2rem; color: #388e3c;">Optimal</div>
      </div>
    </div>

    <div class="section-title">
      <h2>Uncertain Predictions Pending Review</h2>
      <button class="refresh-btn" onclick="loadQueueAndStats()">🔄 Refresh Data</button>
    </div>

    <div id="queue-container" class="queue-grid">
      <div class="empty-state">Loading queue data from FastAPI backend...</div>
    </div>
  </div>

  <script>
    async function loadQueueAndStats() {
      try {
        // Load stats
        const statsRes = await fetch('/drift/stats');
        if (statsRes.ok) {
          const stats = await statsRes.json();
          document.getElementById('kpi-avg-conf').textContent = stats.mean_confidence ? `${(stats.mean_confidence * 100).toFixed(1)}%` : '--%';
          document.getElementById('kpi-rejection-rate').textContent = stats.uncertainty_rejection_rate ? `${(stats.uncertainty_rejection_rate * 100).toFixed(1)}%` : '0.0%';
          const retrainEl = document.getElementById('kpi-retrain-status');
          if (stats.retraining_recommended) {
            retrainEl.textContent = 'Action Required';
            retrainEl.style.color = '#d32f2f';
          } else {
            retrainEl.textContent = 'Nominal';
            retrainEl.style.color = '#388e3c';
          }
        }

        // Load queue
        const queueRes = await fetch('/review-queue');
        if (queueRes.ok) {
          const items = await queueRes.json();
          document.getElementById('kpi-queue-count').textContent = items.length;
          renderQueue(items);
        }
      } catch (e) {
        console.error("Failed to load dashboard data:", e);
      }
    }

    function renderQueue(items) {
      const container = document.getElementById('queue-container');
      if (!items || items.length === 0) {
        container.innerHTML = '<div class="empty-state" style="grid-column: 1/-1;">🎉 No uncertain predictions currently pending agronomist review! All predictions are above confidence threshold.</div>';
        return;
      }

      container.innerHTML = items.map(item => `
        <div class="case-card" id="card-${item.prediction_id}">
          <div class="case-media">
            <img src="${item.image_path || '/static/placeholder.jpg'}" alt="Leaf Image" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%2250%%22 x=%2250%%22 text-anchor=%22middle%22 font-size=%2214%22 fill=%22%23999%22>Leaf Image</text></svg>'">
            <img src="${item.gradcam_path || '/static/placeholder_gradcam.jpg'}" alt="Grad-CAM" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%2250%%22 x=%2250%%22 text-anchor=%22middle%22 font-size=%2214%22 fill=%22%23999%22>Grad-CAM</text></svg>'">
          </div>
          <div class="case-body">
            <div class="case-title">⚠️ ${item.candidate_prediction || 'Uncertain Disease'}</div>
            <div style="font-size: 0.85rem; color: #555;">Confidence: <strong>${((item.confidence || 0) * 100).toFixed(1)}%</strong></div>
            <div class="conf-bar"><div class="conf-fill" style="width: ${(item.confidence || 0) * 100}%;"></div></div>
            <div style="margin-top: 6px;">
              <span class="meta-chip">ID: ${item.prediction_id.substring(0, 8)}</span>
              <span class="meta-chip">Crop: ${item.candidate_crop || 'Unknown'}</span>
            </div>
          </div>
          <div class="action-panel">
            <input type="text" id="notes-${item.prediction_id}" class="notes-input" placeholder="Agronomist clinical notes...">
            <div class="btn-group">
              <button class="btn btn-correct" onclick="submitReview('${item.prediction_id}', 'correct')">✓ Verify</button>
              <button class="btn btn-incorrect" onclick="submitReview('${item.prediction_id}', 'incorrect')">✗ Correct</button>
              <button class="btn btn-more" onclick="submitReview('${item.prediction_id}', 'needs_more_data')">? Needs Data</button>
            </div>
          </div>
        </div>
      `).join('');
    }

    async function submitReview(predictionId, verdict) {
      const notes = document.getElementById(`notes-${predictionId}`).value;
      try {
        const res = await fetch('/review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prediction_id: predictionId,
            verdict: verdict,
            notes: notes,
            agronomist_id: "lead_agronomist"
          })
        });
        if (res.ok) {
          const card = document.getElementById(`card-${predictionId}`);
          if (card) card.remove();
          loadQueueAndStats();
        }
      } catch (e) {
        alert("Failed to submit review: " + e);
      }
    }

    // Auto-load on load
    window.addEventListener('DOMContentLoaded', loadQueueAndStats);
  </script>
</body>
</html>
"""

@dashboard_router.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Renders agronomist verification dashboard."""
    return DASHBOARD_HTML

