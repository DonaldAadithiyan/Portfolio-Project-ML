-- Tokyo Cement Demand Forecasting — Database Schema
-- Tables prefixed with tc_ to avoid conflicts with existing project tables.
-- Run once in the Supabase SQL editor: https://supabase.com/dashboard → SQL Editor

CREATE TABLE IF NOT EXISTS tc_depots (
    depot_id     SERIAL PRIMARY KEY,
    name         VARCHAR(100) NOT NULL UNIQUE,
    district     VARCHAR(100),
    province     VARCHAR(100),
    latitude     NUMERIC(9,6),
    longitude    NUMERIC(9,6),
    pop_weight   NUMERIC(6,5),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tc_demand_panel (
    id                        SERIAL PRIMARY KEY,
    depot_id                  INTEGER REFERENCES tc_depots(depot_id),
    week_start                DATE NOT NULL,
    demand_tonnes             NUMERIC(10,2),
    sales_tonnes              NUMERIC(10,2),
    production_tonnes         NUMERIC(10,2),
    precip_sum                NUMERIC(8,2),
    rain_sum                  NUMERIC(8,2),
    temp_mean                 NUMERIC(6,2),
    humidity_mean             NUMERIC(6,2),
    cloud_cover_mean          NUMERIC(6,2),
    gdp_lka                   NUMERIC(20,2),
    lending_rate              NUMERIC(6,3),
    cbsl_pmi_construction     NUMERIC(6,2),
    govt_consumption          NUMERIC(20,2),
    is_sw_monsoon             SMALLINT,
    is_ne_monsoon             SMALLINT,
    is_dry_season             SMALLINT,
    is_sinhala_tamil_new_year SMALLINT,
    is_vesak                  SMALLINT,
    is_christmas_week         SMALLINT,
    post_holiday_lag_1        SMALLINT,
    post_holiday_lag_2        SMALLINT,
    is_year_end_quarter       SMALLINT,
    data_source               VARCHAR(20) DEFAULT 'augmented',
    UNIQUE (depot_id, week_start)
);

CREATE TABLE IF NOT EXISTS tc_forecasts (
    id               SERIAL PRIMARY KEY,
    depot_id         INTEGER REFERENCES tc_depots(depot_id),
    generated_at     TIMESTAMPTZ DEFAULT NOW(),
    as_of_date       DATE NOT NULL,
    horizon_weeks    SMALLINT NOT NULL,
    forecast_week    DATE NOT NULL,
    demand_forecast  NUMERIC(10,2) NOT NULL,
    model_version    VARCHAR(50),
    UNIQUE (depot_id, as_of_date, horizon_weeks)
);

CREATE TABLE IF NOT EXISTS tc_stock_levels (
    id              SERIAL PRIMARY KEY,
    depot_id        INTEGER REFERENCES tc_depots(depot_id),
    reported_at     TIMESTAMPTZ DEFAULT NOW(),
    week_start      DATE NOT NULL,
    stock_tonnes    NUMERIC(10,2) NOT NULL,
    reported_by     VARCHAR(100),
    UNIQUE (depot_id, week_start)
);

CREATE TABLE IF NOT EXISTS tc_purchase_orders (
    id                  SERIAL PRIMARY KEY,
    depot_id            INTEGER REFERENCES tc_depots(depot_id),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    week_start          DATE NOT NULL,
    recommended_qty     NUMERIC(10,2) NOT NULL,
    current_stock       NUMERIC(10,2),
    forecast_demand     NUMERIC(10,2),
    status              VARCHAR(20) DEFAULT 'pending',
    approved_by         VARCHAR(100),
    approved_at         TIMESTAMPTZ,
    UNIQUE (depot_id, week_start)
);

CREATE TABLE IF NOT EXISTS tc_alerts (
    id              SERIAL PRIMARY KEY,
    depot_id        INTEGER REFERENCES tc_depots(depot_id),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    alert_type      VARCHAR(50) NOT NULL,
    severity        VARCHAR(20) NOT NULL,
    message         TEXT,
    resolved        BOOLEAN DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS tc_sales_actuals (
    id              SERIAL PRIMARY KEY,
    depot_id        INTEGER REFERENCES tc_depots(depot_id),
    week_start      DATE NOT NULL,
    sales_tonnes    NUMERIC(10,2) NOT NULL,
    demand_tonnes   NUMERIC(10,2),
    notes           TEXT,
    entered_by      VARCHAR(100),
    entered_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_by      VARCHAR(100),
    updated_at      TIMESTAMPTZ,
    UNIQUE (depot_id, week_start)
);

CREATE TABLE IF NOT EXISTS tc_retrain_log (
    id                  SERIAL PRIMARY KEY,
    triggered_at        TIMESTAMPTZ DEFAULT NOW(),
    triggered_by        VARCHAR(100),
    trigger_reason      TEXT,
    rows_added          INTEGER,
    training_data_up_to DATE,
    mape_before         NUMERIC(6,3),
    mape_after          NUMERIC(6,3),
    new_model_version   VARCHAR(50),
    status              VARCHAR(20) DEFAULT 'pending',
    error_message       TEXT,
    mlflow_version      INTEGER,
    promoted            BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS tc_model_plots (
    id              SERIAL PRIMARY KEY,
    retrain_id      INTEGER REFERENCES tc_retrain_log(id),
    plot_type       VARCHAR(100) NOT NULL,
    depot_id        INTEGER REFERENCES tc_depots(depot_id),
    image_data      TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (retrain_id, plot_type, depot_id)
);
