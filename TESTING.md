# G. System Testing

## G.1 Overview

System testing for CemCast AI was conducted across three categories: functional testing to validate that each system feature produces correct outputs under valid and invalid conditions; user interface testing to validate usability and visual correctness of the frontend dashboard; and backend API and data pipeline testing to validate endpoint reliability, error handling, and background workflow correctness. Testing was conducted using Postman for API-level tests, manual browser-based end-to-end testing for user interface flows, and direct inspection of Supabase database state to verify background task outcomes.

---

## G.2 Functional Testing

**Objective:** To verify that each system function produces the expected output under both valid and invalid input conditions, and that business rules are correctly enforced.

**Tools:** Manual testing, Postman (API testing), Supabase dashboard (database state verification)

### Health and Depot

| TC ID | Feature | Input / Scenario | Expected Result | Pass/Fail |
|---|---|---|---|---|
| TC-DEP-01 | Health check | `GET /health` with models loaded | `{ "status": "ok", "models_loaded": true }` | Pass |
| TC-DEP-02 | List all depots | `GET /depots` | 200 OK; 24 depot records with `depot_id`, `name`, `district`, `province`, `latitude`, `longitude` | Pass |
| TC-DEP-03 | Invalid depot name | Any endpoint with an unrecognised depot name | 404 Not Found | Pass |

### Forecast Generation

| TC ID | Feature | Input / Scenario | Expected Result | Pass/Fail |
|---|---|---|---|---|
| TC-FCST-01 | Generate forecast | `POST /forecast` with valid depot and `as_of_date` | 200 OK; 6 forecast records returned; rows upserted to `tc_forecasts`; alert evaluation and PO generation triggered in background | Pass |
| TC-FCST-02 | Forecast — models not loaded | `POST /forecast` before training has been run | 503 Service Unavailable; `"Models not loaded"` message | Pass |
| TC-FCST-03 | Retrieve stored forecasts | `GET /forecasts/{depot}` | 200 OK; latest 6 horizon forecasts in ascending horizon order | Pass |

### Stock Levels

| TC ID | Feature | Input / Scenario | Expected Result | Pass/Fail |
|---|---|---|---|---|
| TC-STK-01 | Submit stock level | `POST /stock` with valid depot, `week_start`, and `stock_tonnes` | 200 OK; record saved to `tc_stock_levels`; alert evaluation triggered in background | Pass |
| TC-STK-02 | Retrieve stock history | `GET /stock/{depot}` | 200 OK; `latest` stock record and up to 12 weeks of `history`; `latest: null` if no records exist | Pass |

### Alert System

| TC ID | Feature | Input / Scenario | Expected Result | Pass/Fail |
|---|---|---|---|---|
| TC-ALR-01 | Low stock critical alert | Current stock < 80% of 2-week forecast demand after `POST /stock` or `POST /forecast` | `low_stock` alert with severity `critical` inserted into `tc_alerts` | Pass |
| TC-ALR-02 | Low stock warning alert | Current stock < 90% of 4-week forecast demand (above critical threshold) | `low_stock` alert with severity `warning` inserted | Pass |
| TC-ALR-03 | Retrieve active alerts | `GET /alerts/{depot}` | 200 OK; unresolved alerts with `alert_type`, `severity`, `message` | Pass |
| TC-ALR-04 | Resolve an alert | `PATCH /alerts/{alert_id}/resolve` | 200 OK; `resolved=true` and `resolved_at` set; alert no longer returned in active list | Pass |

### Purchase Orders

| TC ID | Feature | Input / Scenario | Expected Result | Pass/Fail |
|---|---|---|---|---|
| TC-PO-01 | Auto PO generation | `POST /forecast` where stock < next-week demand + 25% safety buffer | PO record created in `tc_purchase_orders` with status `pending` | Pass |
| TC-PO-02 | Retrieve pending POs | `GET /purchase-orders/{depot}` | 200 OK; pending orders with `recommended_qty`, `current_stock`, `forecast_demand` | Pass |
| TC-PO-03 | Approve a purchase order | `PATCH /purchase-orders/{po_id}` with `{ "status": "approved", "approved_by": "manager" }` | 200 OK; `approved_by` and `approved_at` updated in database | Pass |
| TC-PO-04 | Reject a purchase order | `PATCH /purchase-orders/{po_id}` with `{ "status": "rejected" }` | 200 OK; status set to `rejected` | Pass |

### Sales Actuals

| TC ID | Feature | Input / Scenario | Expected Result | Pass/Fail |
|---|---|---|---|---|
| TC-SALES-01 | Submit sales record | `POST /sales` with valid depot, `week_start`, and `sales_tonnes` | 200 OK; record saved to `tc_sales_actuals`; matching `tc_demand_panel` row updated to `data_source='actual'`; retrain check triggered | Pass |
| TC-SALES-02 | Retrieve sales history | `GET /sales/{depot}` | 200 OK; up to 12 most recent weeks (configurable via `?weeks=N`, max 52) | Pass |
| TC-SALES-03 | Auto-retrain trigger | 5th new sales record submitted since last completed retrain | Retrain automatically triggered; `tc_retrain_log` row created with `triggered_by='auto'` | Pass |

### Model Retraining

| TC ID | Feature | Input / Scenario | Expected Result | Pass/Fail |
|---|---|---|---|---|
| TC-RTRN-01 | Manual retrain trigger | `POST /retrain` | 200 OK; `retrain_id` returned; `tc_retrain_log` status changes to `running` | Pass |
| TC-RTRN-02 | Retrain status | `GET /retrain/status/{retrain_id}` after completion | `status=completed`; `mape_before`, `mape_after`, and `promoted` all populated | Pass |
| TC-RTRN-03 | Model promotion | Retrain MAPE equal or better than previous run | `promoted=true`; models reloaded into memory automatically | Pass |
| TC-RTRN-04 | Model not promoted | Retrain MAPE worse than previous run | `promoted=false`; existing production models unchanged | Pass |

### Dashboard

| TC ID | Feature | Input / Scenario | Expected Result | Pass/Fail |
|---|---|---|---|---|
| TC-DASH-01 | Full dashboard load | `GET /dashboard/{depot}` for depot with data | 200 OK; returns depot metadata, `latest_stock`, 6-week `forecast`, `pending_pos`, and `active_alerts` in one response | Pass |
| TC-DASH-02 | Dashboard — no data | `GET /dashboard/{depot}` for depot with no submissions | 200 OK; `latest_stock: null`, `forecast: []`, `pending_pos: []`, `active_alerts: []` | Pass |

---

## G.3 User Interface Testing

**Objective:** To validate that the frontend dashboard is visually correct, consistently structured, and usable by non-technical operational staff.

**Tools:** Manual browser testing (Chrome, Firefox), browser developer tools for responsive layout inspection

| TC ID | Description | Steps | Expected Result | Pass/Fail |
|---|---|---|---|---|
| TC-UI-01 | Dashboard loads for selected depot | Navigate to dashboard and select a depot | Latest stock, 6-week demand total, active alert count, and pending PO count displayed as summary cards | Pass |
| TC-UI-02 | 6-week forecast chart renders | View the forecast panel for a depot | Line chart with 6 data points; `horizon_weeks` on x-axis, `demand_tonnes` on y-axis; axes correctly labelled | Pass |
| TC-UI-03 | Alert severity colour coding | View active alerts panel | Critical alerts in red; warning alerts in amber | Pass |
| TC-UI-04 | Purchase order approval | Click Approve on a pending PO | PO status updates to Approved; row moves out of pending list | Pass |
| TC-UI-05 | Purchase order rejection | Click Reject on a pending PO | PO status updates to Rejected; row removed from pending list | Pass |
| TC-UI-06 | Resolve alert | Click Resolve on an active alert | Alert removed from active list | Pass |
| TC-UI-07 | Sales entry form | Submit a sales record via the form | Success notification shown; sales history table updates | Pass |
| TC-UI-08 | Stock entry form | Submit current stock level | Stock card updates; alert panel refreshes if new alerts were triggered | Pass |

---

## G.4 Backend API and Data Pipeline Testing

**Objective:** To validate that all core API endpoints return correct HTTP responses, handle invalid input gracefully, persist data correctly to Supabase, and trigger background workflows with the expected side effects.

**Tools:** Postman (API testing), Supabase Table Editor (database state verification after each test)

| TC ID | Endpoint | Test Scenario | Expected Response | Pass/Fail |
|---|---|---|---|---|
| TC-API-01 | `GET /health` | Service running, models loaded | `200 OK`; `{ "status": "ok", "models_loaded": true }` | Pass |
| TC-API-02 | `GET /depots` | Standard request | `200 OK`; array of 24 depot objects | Pass |
| TC-API-03 | `POST /forecast` | Valid body: `{ "depot": "Colombo", "as_of_date": "2025-06-01" }` | `200 OK`; `forecasts` array with 6 entries; each has `horizon`, `forecast_week`, `demand_tonnes` | Pass |
| TC-API-04 | `POST /forecast` | Models not loaded | `503 Service Unavailable` | Pass |
| TC-API-05 | `POST /forecast` | Invalid depot name | `404 Not Found` | Pass |
| TC-API-06 | `GET /forecasts/{depot}` | Valid depot with stored forecasts | `200 OK`; depot name and latest 6 forecast rows | Pass |
| TC-API-07 | `POST /stock` | Valid body: `{ "depot": "Galle", "week_start": "2025-06-02", "stock_tonnes": 450.0 }` | `200 OK`; `{ "status": "saved", "stock_id": <id> }`; row confirmed in `tc_stock_levels` | Pass |
| TC-API-08 | `GET /stock/{depot}` | Valid depot | `200 OK`; `latest` object and `history` array | Pass |
| TC-API-09 | `GET /alerts/{depot}` | Depot with active alerts | `200 OK`; unresolved alerts with `alert_type`, `severity`, `message` | Pass |
| TC-API-10 | `PATCH /alerts/{alert_id}/resolve` | Valid alert ID | `200 OK`; `resolved: true`, `resolved_at` set; confirmed in database | Pass |
| TC-API-11 | `GET /purchase-orders/{depot}` | Depot with pending POs | `200 OK`; list with `recommended_qty`, `current_stock`, `forecast_demand` | Pass |
| TC-API-12 | `PATCH /purchase-orders/{po_id}` | Approve: `{ "status": "approved", "approved_by": "ops_manager" }` | `200 OK`; `approved_at` set; database row confirms `status=approved` | Pass |
| TC-API-13 | `POST /sales` | Valid body: `{ "depot": "Kandy", "week_start": "2025-06-02", "sales_tonnes": 312.5 }` | `200 OK`; row in `tc_sales_actuals`; `tc_demand_panel` updated to `data_source='actual'` | Pass |
| TC-API-14 | `GET /sales/{depot}` | Default request | `200 OK`; up to 12 most recent sales records | Pass |
| TC-API-15 | `GET /dashboard/{depot}` | Depot with full data | `200 OK`; depot metadata, latest stock, 6-week forecast, pending POs, active alerts in one response | Pass |
| TC-API-16 | `POST /retrain` | Manual trigger | `200 OK`; `retrain_id` returned; `tc_retrain_log` row created | Pass |
| TC-API-17 | `GET /retrain/status/{retrain_id}` | Completed retrain | `200 OK`; `status=completed`, `mape_before`, `mape_after`, `promoted` all populated | Pass |
| TC-API-18 | Background alert evaluation | `POST /forecast` for depot with stock below critical threshold | `tc_alerts` row with `low_stock / critical` automatically inserted after forecast | Pass |
| TC-API-19 | Background PO generation | `POST /forecast` where stock below demand + safety buffer | `tc_purchase_orders` row with `status=pending` automatically upserted after forecast | Pass |
| TC-API-20 | Background auto-retrain | 5th sales record submitted since last retrain | `tc_retrain_log` row inserted with `triggered_by='auto'`; training runs without manual intervention | Pass |
