import logging
import os
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.db.db import get_client

logger = logging.getLogger(__name__)

app = FastAPI(title="Tokyo Cement Demand Forecasting API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cfg: dict = {}
_models_loaded = False


@app.on_event("startup")
async def startup():
    global _cfg, _models_loaded
    with open("config.yaml") as f:
        _cfg = yaml.safe_load(f)

    from src.model.predict import load_models
    try:
        load_models(_cfg)
        _models_loaded = True
        logger.info("[SERVE] Models loaded successfully")
    except Exception as e:
        logger.warning("[SERVE] Could not load models at startup (train first): %s", e)


# ── Helpers ───────────────────────────────────────────────────

def _resolve_depot(name: str) -> tuple[int, str]:
    sb = get_client()
    result = sb.table("tc_depots").select("depot_id,name").eq("name", name).maybe_single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Depot '{name}' not found")
    return result.data["depot_id"], result.data["name"]


def _get_recent_panel(depot_id: int, n_weeks: int = 52) -> pd.DataFrame:
    sb = get_client()
    result = sb.table("tc_demand_panel").select("*").eq("depot_id", depot_id).order(
        "week_start", desc=True
    ).limit(n_weeks).execute()
    df = pd.DataFrame(result.data)
    if df.empty:
        return df
    depot_result = sb.table("tc_depots").select("name").eq("depot_id", depot_id).single().execute()
    df["depot"] = depot_result.data["name"]
    return df.sort_values("week_start")


# ── GET /depots ───────────────────────────────────────────────

@app.get("/depots")
def get_depots():
    sb = get_client()
    result = sb.table("tc_depots").select(
        "depot_id,name,district,province,latitude,longitude"
    ).order("name").execute()
    return result.data


# ── POST /forecast ────────────────────────────────────────────

class ForecastRequest(BaseModel):
    depot: str
    as_of_date: date


@app.post("/forecast")
def create_forecast(req: ForecastRequest, background_tasks: BackgroundTasks):
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Run `python pipeline.py --mode train` first.")

    depot_id, depot_name = _resolve_depot(req.depot)
    recent = _get_recent_panel(depot_id, 52)
    if recent.empty:
        raise HTTPException(status_code=422, detail=f"No panel data for depot '{depot_name}'")

    from src.model.predict import forecast_depot
    forecasts = forecast_depot(depot_name, req.as_of_date, recent, _cfg)

    sb = get_client()
    for fc in forecasts:
        sb.table("tc_forecasts").upsert({
            "depot_id": depot_id,
            "as_of_date": req.as_of_date.isoformat(),
            "horizon_weeks": fc["horizon"],
            "forecast_week": fc["forecast_week"].isoformat(),
            "demand_forecast": fc["demand_tonnes"],
        }, on_conflict="depot_id,as_of_date,horizon_weeks").execute()

    background_tasks.add_task(_run_alert_evaluation, depot_id, forecasts)
    background_tasks.add_task(_run_po_generation, depot_id, forecasts, req.as_of_date)

    return {
        "depot": depot_name,
        "as_of_date": req.as_of_date,
        "forecasts": forecasts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── GET /forecasts/{depot} ────────────────────────────────────

@app.get("/forecasts/{depot}")
def get_forecasts(depot: str, as_of_date: Optional[date] = None):
    depot_id, depot_name = _resolve_depot(depot)
    sb = get_client()
    query = sb.table("tc_forecasts").select(
        "horizon_weeks,forecast_week,demand_forecast"
    ).eq("depot_id", depot_id)
    if as_of_date:
        query = query.eq("as_of_date", as_of_date.isoformat()).order("horizon_weeks")
    else:
        query = query.order("generated_at", desc=True).order("horizon_weeks").limit(6)
    result = query.execute()
    rows = [
        {"horizon": r["horizon_weeks"], "forecast_week": r["forecast_week"],
         "demand_tonnes": r["demand_forecast"]}
        for r in result.data
    ]
    return {"depot": depot_name, "forecasts": rows}


# ── POST /stock ───────────────────────────────────────────────

class StockRequest(BaseModel):
    depot: str
    week_start: date
    stock_tonnes: float
    reported_by: Optional[str] = None


@app.post("/stock")
def submit_stock(req: StockRequest, background_tasks: BackgroundTasks):
    depot_id, _ = _resolve_depot(req.depot)
    sb = get_client()
    result = sb.table("tc_stock_levels").upsert({
        "depot_id": depot_id,
        "week_start": req.week_start.isoformat(),
        "stock_tonnes": req.stock_tonnes,
        "reported_by": req.reported_by,
    }, on_conflict="depot_id,week_start").execute()
    stock_id = result.data[0]["id"] if result.data else None
    background_tasks.add_task(_run_alert_evaluation, depot_id, None)
    return {"status": "saved", "stock_id": stock_id}


# ── GET /stock/{depot} ────────────────────────────────────────

@app.get("/stock/{depot}")
def get_stock(depot: str):
    depot_id, depot_name = _resolve_depot(depot)
    sb = get_client()
    result = sb.table("tc_stock_levels").select(
        "week_start,stock_tonnes,reported_at"
    ).eq("depot_id", depot_id).order("week_start", desc=True).limit(12).execute()
    rows = result.data
    return {"depot": depot_name, "latest": rows[0] if rows else None, "history": rows}


# ── GET /purchase-orders/{depot} ─────────────────────────────

@app.get("/purchase-orders/{depot}")
def get_purchase_orders(depot: str, status: str = "pending"):
    depot_id, depot_name = _resolve_depot(depot)
    sb = get_client()
    query = sb.table("tc_purchase_orders").select(
        "id,week_start,recommended_qty,current_stock,forecast_demand,status,created_at"
    ).eq("depot_id", depot_id)
    if status != "all":
        query = query.eq("status", status)
    result = query.order("created_at", desc=True).execute()
    rows = result.data
    for r in rows:
        r["po_id"] = r.pop("id")
        r["depot"] = depot_name
    return rows


# ── PATCH /purchase-orders/{po_id} ───────────────────────────

class POPatch(BaseModel):
    status: str
    approved_by: Optional[str] = None


@app.patch("/purchase-orders/{po_id}")
def patch_purchase_order(po_id: int, req: POPatch):
    sb = get_client()
    result = sb.table("tc_purchase_orders").update({
        "status": req.status,
        "approved_by": req.approved_by,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", po_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Purchase order not found")
    return {"po_id": po_id, "status": req.status, "approved_at": datetime.now(timezone.utc).isoformat()}


# ── GET /alerts/{depot} ───────────────────────────────────────

@app.get("/alerts/{depot}")
def get_alerts(depot: str, resolved: bool = False):
    depot_id, depot_name = _resolve_depot(depot)
    sb = get_client()
    result = sb.table("tc_alerts").select(
        "id,alert_type,severity,message,created_at"
    ).eq("depot_id", depot_id).eq("resolved", resolved).order("created_at", desc=True).execute()
    rows = result.data
    for r in rows:
        r["alert_id"] = r.pop("id")
        r["depot"] = depot_name
    return rows


# ── PATCH /alerts/{alert_id}/resolve ─────────────────────────

@app.patch("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int):
    sb = get_client()
    result = sb.table("tc_alerts").update({
        "resolved": True,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", alert_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"alert_id": alert_id, "resolved": True, "resolved_at": datetime.now(timezone.utc).isoformat()}


# ── GET /dashboard/{depot} ────────────────────────────────────

@app.get("/dashboard/{depot}")
def get_dashboard(depot: str):
    depot_id, _ = _resolve_depot(depot)
    sb = get_client()

    depot_meta = sb.table("tc_depots").select("*").eq("depot_id", depot_id).single().execute().data
    stock_result = sb.table("tc_stock_levels").select(
        "week_start,stock_tonnes,reported_at"
    ).eq("depot_id", depot_id).order("week_start", desc=True).limit(1).execute()
    latest_stock = stock_result.data[0] if stock_result.data else None
    forecast_result = sb.table("tc_forecasts").select(
        "horizon_weeks,forecast_week,demand_forecast"
    ).eq("depot_id", depot_id).order("generated_at", desc=True).order("horizon_weeks").limit(6).execute()
    forecast = [{"horizon": r["horizon_weeks"], "forecast_week": r["forecast_week"],
                 "demand_tonnes": r["demand_forecast"]} for r in forecast_result.data]
    po_result = sb.table("tc_purchase_orders").select(
        "id,week_start,recommended_qty,current_stock,forecast_demand,status"
    ).eq("depot_id", depot_id).eq("status", "pending").execute()
    pending_pos = [dict(r, po_id=r.pop("id")) for r in po_result.data]
    alert_result = sb.table("tc_alerts").select(
        "id,alert_type,severity,message,created_at"
    ).eq("depot_id", depot_id).eq("resolved", False).execute()
    active_alerts = [dict(r, alert_id=r.pop("id")) for r in alert_result.data]

    return {
        "depot": depot_meta,
        "latest_stock": latest_stock,
        "forecast": forecast,
        "pending_pos": pending_pos,
        "active_alerts": active_alerts,
    }


# ── POST /sales ───────────────────────────────────────────────

class SalesSubmit(BaseModel):
    depot: str
    week_start: date
    sales_tonnes: float
    demand_tonnes: Optional[float] = None
    entered_by: Optional[str] = None
    notes: Optional[str] = None


@app.post("/sales")
def submit_sales(req: SalesSubmit, background_tasks: BackgroundTasks):
    depot_id, _ = _resolve_depot(req.depot)
    sb = get_client()
    result = sb.table("tc_sales_actuals").upsert({
        "depot_id": depot_id,
        "week_start": req.week_start.isoformat(),
        "sales_tonnes": req.sales_tonnes,
        "demand_tonnes": req.demand_tonnes,
        "notes": req.notes,
        "entered_by": req.entered_by,
    }, on_conflict="depot_id,week_start").execute()
    sales_id = result.data[0]["id"] if result.data else None

    sb.table("tc_demand_panel").update({
        "sales_tonnes": req.sales_tonnes,
        "demand_tonnes": req.demand_tonnes,
        "data_source": "actual",
    }).eq("depot_id", depot_id).eq("week_start", req.week_start.isoformat()).execute()

    background_tasks.add_task(_maybe_trigger_retrain, "auto", f"New sales for {req.depot} week {req.week_start}")
    return {"status": "saved", "sales_id": sales_id, "retrain_scheduled": True}


# ── PUT /sales/{depot}/{week_start} ──────────────────────────

class SalesUpdate(BaseModel):
    sales_tonnes: float
    demand_tonnes: Optional[float] = None
    updated_by: Optional[str] = None
    notes: Optional[str] = None


@app.put("/sales/{depot}/{week_start}")
def update_sales(depot: str, week_start: date, req: SalesUpdate, background_tasks: BackgroundTasks):
    depot_id, _ = _resolve_depot(depot)
    sb = get_client()
    result = sb.table("tc_sales_actuals").update({
        "sales_tonnes": req.sales_tonnes,
        "demand_tonnes": req.demand_tonnes,
        "notes": req.notes,
        "updated_by": req.updated_by,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("depot_id", depot_id).eq("week_start", week_start.isoformat()).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Sales record not found")
    sales_id = result.data[0]["id"]

    sb.table("tc_demand_panel").update({
        "sales_tonnes": req.sales_tonnes,
        "demand_tonnes": req.demand_tonnes,
        "data_source": "actual",
    }).eq("depot_id", depot_id).eq("week_start", week_start.isoformat()).execute()

    background_tasks.add_task(_maybe_trigger_retrain, "auto", f"Updated sales for {depot} week {week_start}")
    return {"status": "updated", "sales_id": sales_id, "retrain_scheduled": True}


# ── GET /sales/{depot} ────────────────────────────────────────

@app.get("/sales/{depot}")
def get_sales(depot: str, weeks: int = 12):
    depot_id, _ = _resolve_depot(depot)
    weeks = min(weeks, 52)
    sb = get_client()
    result = sb.table("tc_sales_actuals").select(
        "week_start,sales_tonnes,demand_tonnes,notes"
    ).eq("depot_id", depot_id).order("week_start", desc=True).limit(weeks).execute()
    return result.data


# ── POST /retrain ─────────────────────────────────────────────

class RetrainRequest(BaseModel):
    triggered_by: Optional[str] = "admin"


@app.post("/retrain")
async def trigger_retrain(req: RetrainRequest, background_tasks: BackgroundTasks):
    retrain_id = _create_retrain_log_row(req.triggered_by, "Manual trigger via API")
    background_tasks.add_task(_run_retrain, retrain_id)
    return {
        "status": "started",
        "retrain_id": retrain_id,
        "message": f"Retraining in progress. Check /retrain/status/{retrain_id} for updates.",
    }


# ── GET /retrain/status/{retrain_id} ─────────────────────────

@app.get("/retrain/status/{retrain_id}")
def get_retrain_status(retrain_id: int):
    sb = get_client()
    result = sb.table("tc_retrain_log").select("*").eq("id", retrain_id).maybe_single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Retrain log not found")
    return result.data


# ── GET /retrain/history ──────────────────────────────────────

@app.get("/retrain/history")
def get_retrain_history():
    sb = get_client()
    result = sb.table("tc_retrain_log").select(
        "id,triggered_at,mape_before,mape_after,promoted,status"
    ).order("triggered_at", desc=True).limit(10).execute()
    return [dict(r, retrain_id=r.pop("id")) for r in result.data]


# ── GET /plots/latest ─────────────────────────────────────────

@app.get("/plots/latest")
def get_latest_plots():
    sb = get_client()
    # Get latest completed retrain id
    log_result = sb.table("tc_retrain_log").select("id").eq(
        "status", "completed"
    ).order("id", desc=True).limit(1).execute()
    if not log_result.data:
        return []
    latest_id = log_result.data[0]["id"]
    result = sb.table("tc_model_plots").select("plot_type,image_data").eq(
        "retrain_id", latest_id
    ).is_("depot_id", "null").order("plot_type").execute()
    return result.data


# ── GET /plots/depot/{depot} ──────────────────────────────────

@app.get("/plots/depot/{depot}")
def get_depot_plot(depot: str):
    depot_id, depot_name = _resolve_depot(depot)
    sb = get_client()
    result = sb.table("tc_model_plots").select(
        "plot_type,image_data,retrain_id,created_at"
    ).eq("plot_type", "depot_forecast").eq("depot_id", depot_id).order(
        "retrain_id", desc=True
    ).limit(1).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail=f"No depot forecast plot found for '{depot_name}'")
    row = result.data[0]
    row["depot"] = depot_name
    return row


# ── GET /plots/{retrain_id} ───────────────────────────────────

@app.get("/plots/{retrain_id}")
def get_plots_for_run(retrain_id: int, plot_type: Optional[str] = None):
    sb = get_client()
    query = sb.table("tc_model_plots").select(
        "plot_type,depot_id,image_data,created_at"
    ).eq("retrain_id", retrain_id)
    if plot_type:
        query = query.eq("plot_type", plot_type)
    return query.order("plot_type").execute().data


# ── Internal: alert evaluation ────────────────────────────────

def _run_alert_evaluation(depot_id: int, forecasts: Optional[list]) -> None:
    try:
        sb = get_client()

        stock_result = sb.table("tc_stock_levels").select("stock_tonnes").eq(
            "depot_id", depot_id
        ).order("week_start", desc=True).limit(1).execute()
        current_stock = float(stock_result.data[0]["stock_tonnes"]) if stock_result.data else None

        if forecasts is None:
            fc_result = sb.table("tc_forecasts").select("demand_forecast").eq(
                "depot_id", depot_id
            ).order("generated_at", desc=True).order("horizon_weeks").limit(6).execute()
            forecasts = [{"demand_tonnes": r["demand_forecast"], "horizon": i + 1}
                         for i, r in enumerate(fc_result.data)]

        if not forecasts:
            return

        demand_2w = sum(f["demand_tonnes"] for f in forecasts[:2])
        demand_4w = sum(f["demand_tonnes"] for f in forecasts[:4])
        demand_6w = sum(f["demand_tonnes"] for f in forecasts[:6])
        demand_w1 = forecasts[0]["demand_tonnes"]

        panel_result = sb.table("tc_demand_panel").select("demand_tonnes").eq(
            "depot_id", depot_id
        ).order("week_start", desc=True).limit(4).execute()
        recent = [r["demand_tonnes"] for r in panel_result.data if r["demand_tonnes"] is not None]
        rolling_mean = float(sum(recent) / len(recent)) if recent else demand_w1

        alerts_to_create = []
        alert_cfg = _cfg.get("alerts", {})

        if current_stock is not None:
            threshold_crit = demand_2w * alert_cfg.get("low_stock_critical_buffer", 0.80)
            if current_stock < threshold_crit:
                alerts_to_create.append((
                    "low_stock", "critical",
                    f"Projected stockout in 2 weeks. Current stock {current_stock:.0f}t vs 2-week forecast {demand_2w:.0f}t."
                ))
            elif current_stock < demand_4w * alert_cfg.get("low_stock_warning_buffer", 0.90):
                alerts_to_create.append((
                    "low_stock", "warning",
                    f"Stock may run low within 4 weeks. Current stock {current_stock:.0f}t vs 4-week forecast {demand_4w:.0f}t."
                ))
            overstock_thresh = demand_6w * alert_cfg.get("overstock_multiplier", 1.50)
            if current_stock > overstock_thresh:
                alerts_to_create.append((
                    "overstock", "warning",
                    f"Excess stock detected. {current_stock:.0f}t held vs {demand_6w:.0f}t forecast over 6 weeks."
                ))

        spike_mult = alert_cfg.get("demand_spike_multiplier", 1.30)
        if demand_w1 > rolling_mean * spike_mult:
            pct = (demand_w1 / rolling_mean - 1) * 100 if rolling_mean else 0
            alerts_to_create.append((
                "demand_spike", "warning",
                f"Demand spike forecast: {demand_w1:.0f}t vs 4-week avg {rolling_mean:.0f}t (+{pct:.0f}%)."
            ))

        for alert_type, severity, message in alerts_to_create:
            existing = sb.table("tc_alerts").select("id").eq("depot_id", depot_id).eq(
                "alert_type", alert_type
            ).eq("resolved", False).execute()
            if existing.data:
                continue
            sb.table("tc_alerts").insert({
                "depot_id": depot_id,
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
            }).execute()

    except Exception as e:
        logger.warning("[SERVE] Alert evaluation failed for depot_id=%d: %s", depot_id, e)


# ── Internal: PO generation ───────────────────────────────────

def _run_po_generation(depot_id: int, forecasts: list, as_of_date: date) -> None:
    try:
        sb = get_client()
        stock_result = sb.table("tc_stock_levels").select("stock_tonnes").eq(
            "depot_id", depot_id
        ).order("week_start", desc=True).limit(1).execute()
        current_stock = float(stock_result.data[0]["stock_tonnes"]) if stock_result.data else 0.0

        if not forecasts:
            return

        forecast_w1 = forecasts[0]["demand_tonnes"]
        safety_pct = _cfg.get("purchase_orders", {}).get("safety_stock_pct", 0.25)
        safety_stock = forecast_w1 * safety_pct
        recommended_qty = max(0.0, forecast_w1 + safety_stock - current_stock)
        if recommended_qty <= 0:
            return

        from datetime import timedelta
        week_start = as_of_date + timedelta(weeks=1)
        sb.table("tc_purchase_orders").upsert({
            "depot_id": depot_id,
            "week_start": week_start.isoformat(),
            "recommended_qty": recommended_qty,
            "current_stock": current_stock,
            "forecast_demand": forecast_w1,
        }, on_conflict="depot_id,week_start").execute()

    except Exception as e:
        logger.warning("[SERVE] PO generation failed for depot_id=%d: %s", depot_id, e)


# ── Internal: retrain helpers ─────────────────────────────────

def _create_retrain_log_row(triggered_by: str, reason: str) -> int:
    sb = get_client()
    result = sb.table("tc_retrain_log").insert({
        "triggered_by": triggered_by,
        "trigger_reason": reason,
        "status": "pending",
    }).execute()
    return result.data[0]["id"]


def _maybe_trigger_retrain(triggered_by: str, reason: str) -> None:
    batch_size = _cfg.get("model", {}).get("retrain_batch_size", 5)
    sb = get_client()

    log_result = sb.table("tc_retrain_log").select("triggered_at").eq(
        "status", "completed"
    ).order("id", desc=True).limit(1).execute()
    last_retrain_at = log_result.data[0]["triggered_at"] if log_result.data else "2000-01-01"

    pending_result = sb.table("tc_sales_actuals").select("id", count="exact").gt(
        "entered_at", last_retrain_at
    ).execute()
    pending = pending_result.count or 0

    if pending >= batch_size:
        retrain_id = _create_retrain_log_row(triggered_by, reason)
        _run_retrain(retrain_id)
    else:
        _create_retrain_log_row(triggered_by, f"{reason} (pending, {pending}/{batch_size} rows)")
        logger.info("[SERVE] Retrain pending: %d/%d new rows", pending, batch_size)


def _run_retrain(retrain_id: int) -> None:
    from src.model.train import train_all_horizons
    from src.model.evaluate import run_evaluation
    from src.features.build_features import rebuild_lag_features_for_depots

    sb = get_client()
    sb.table("tc_retrain_log").update({"status": "running"}).eq("id", retrain_id).execute()

    try:
        log_result = sb.table("tc_retrain_log").select("mape_after").eq(
            "status", "completed"
        ).order("id", desc=True).limit(1).execute()
        mape_before = float(log_result.data[0]["mape_after"]) if log_result.data else None

        result = train_all_horizons(_cfg, retrain_id=retrain_id)
        df_full = rebuild_lag_features_for_depots([], _cfg)
        run_evaluation(result, df_full, retrain_id, _cfg)

        latest_result = sb.table("tc_demand_panel").select("week_start").order(
            "week_start", desc=True
        ).limit(1).execute()
        latest_week = latest_result.data[0]["week_start"] if latest_result.data else None

        sb.table("tc_retrain_log").update({
            "status": "completed",
            "mape_before": mape_before,
            "mape_after": result["overall_mape"],
            "promoted": result["promoted"],
            "training_data_up_to": latest_week,
        }).eq("id", retrain_id).execute()

        if result["promoted"]:
            global _models_loaded
            from src.model.predict import load_models
            try:
                load_models(_cfg)
                _models_loaded = True
                logger.info("[SERVE] Models reloaded after promotion")
            except Exception as e:
                logger.warning("[SERVE] Model reload failed: %s", e)

        logger.info("[SERVE] Retrain %d complete: MAPE %.2f%% (promoted=%s)",
                    retrain_id, result["overall_mape"], result["promoted"])

    except Exception as e:
        logger.error("[SERVE] Retrain %d failed: %s", retrain_id, e)
        sb.table("tc_retrain_log").update({
            "status": "failed",
            "error_message": str(e),
        }).eq("id", retrain_id).execute()
