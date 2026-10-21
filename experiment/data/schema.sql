CREATE TABLE zones (
    location_id   INTEGER PRIMARY KEY,  -- TLC taxi zone id
    borough       TEXT,                 -- 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'EWR'
    zone          TEXT,                 -- zone name, e.g. 'JFK Airport', 'Upper East Side North'
    service_zone  TEXT                  -- 'Yellow Zone', 'Boro Zone', 'Airports', 'EWR'
);

CREATE TABLE payment_types (
    payment_type_id  INTEGER PRIMARY KEY,
    name             TEXT               -- 'Credit card', 'Cash', 'No charge', 'Dispute', 'Unknown', 'Voided trip'
);

CREATE TABLE trips (
    trip_id               INTEGER PRIMARY KEY,
    vendor_id             INTEGER,      -- TPEP provider code
    pickup_datetime       TEXT,         -- 'YYYY-MM-DD HH:MM:SS', local time
    dropoff_datetime      TEXT,         -- 'YYYY-MM-DD HH:MM:SS', local time
    passenger_count       INTEGER,
    trip_distance         REAL,         -- miles
    rate_code_id          INTEGER,      -- 1 standard, 2 JFK, 3 Newark, 4 Nassau/Westchester, 5 negotiated, 6 group ride, 99 unknown
    pu_location_id        INTEGER REFERENCES zones(location_id),  -- pickup zone
    do_location_id        INTEGER REFERENCES zones(location_id),  -- dropoff zone
    payment_type          INTEGER REFERENCES payment_types(payment_type_id),
    fare_amount           REAL,         -- metered fare, USD
    tip_amount            REAL,         -- USD; cash tips are not recorded
    tolls_amount          REAL,         -- USD
    total_amount          REAL,         -- USD, total charged (excludes cash tips)
    congestion_surcharge  REAL,         -- USD
    airport_fee           REAL          -- USD
);

CREATE INDEX idx_trips_pu ON trips(pu_location_id);

CREATE INDEX idx_trips_do ON trips(do_location_id);

CREATE INDEX idx_trips_pickup ON trips(pickup_datetime);
