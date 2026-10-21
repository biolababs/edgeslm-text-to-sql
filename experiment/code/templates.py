"""Question/SQL templates for the taxi text-to-SQL set.

Each template has natural-language paraphrases and one gold SQL pattern.
Slots like {borough} are filled in two forms: a natural-language string for
the question and a SQL literal for the query.

Templates in DEFAULT_HELDOUT never appear in training. Each one needs a SQL
construct or combination the training templates do not use (HAVING, NOT IN,
scalar subquery, derived speed, borough inequality, date grouping), so the
held-out split measures generalisation and not template recall.

Yellow-taxi pickups are almost all in Manhattan, Queens, Brooklyn and the
Bronx, so borough slots have only ~4 values. Most borough templates
therefore also take an hour or day-of-week slot to get enough instances.
"""
import calendar
from dataclasses import dataclass, field

DEFAULT_HELDOUT = [
    "highest_avg_fare_zone_min_pickups",  # HAVING
    "cross_borough_trips_hour",           # pz.borough <> dz.borough
    "avg_speed_borough",                  # derived per-row quantity
    "busiest_date_borough",               # date() grouping
    "zones_without_pickups",              # anti-join
    "above_avg_fare_zone",                # scalar subquery
]


@dataclass
class Template:
    id: str
    questions: list
    sql: str
    slots: dict = field(default_factory=dict)  # slot name -> slot type
    ordered: bool = False      # row order matters when comparing results
    rank_sql: str = None       # ranking metric, DESC, LIMIT {n}; used to reject ties
    allow_zero: bool = False   # keep instances whose single answer is 0


PU = "JOIN zones pz ON t.pu_location_id = pz.location_id"
DO = "JOIN zones dz ON t.do_location_id = dz.location_id"
PAY = "JOIN payment_types p ON t.payment_type = p.payment_type_id"
HOUR = "CAST(strftime('%H', t.pickup_datetime) AS INTEGER)"
DOW = "strftime('%w', t.pickup_datetime)"
MINUTES = "(julianday(t.dropoff_datetime) - julianday(t.pickup_datetime)) * 1440"


# ---------------------------------------------------------------- slot values

def lit(s) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def hour_nl(h: int) -> str:
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


DOW_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def build_context(conn) -> dict:
    """Pull slot vocabularies from the database so every value actually occurs."""
    q = lambda sql: [r[0] for r in conn.execute(sql)]
    return {
        "boroughs": q(f"""SELECT pz.borough FROM trips t {PU}
                          GROUP BY pz.borough HAVING COUNT(*) >= 200 ORDER BY COUNT(*) DESC"""),
        "zones": q(f"""SELECT pz.zone FROM trips t {PU}
                       GROUP BY pz.zone HAVING COUNT(*) >= 100 ORDER BY COUNT(*) DESC"""),
        "dates": q("""SELECT date(pickup_datetime) AS d FROM trips
                      GROUP BY d HAVING COUNT(*) >= 1000 ORDER BY d"""),
    }


def _date_nl(iso: str) -> str:
    y, m, d = map(int, iso.split("-"))
    return f"{calendar.month_name[m]} {d}, {y}"


# slot type -> fn(rng, ctx) -> (question text, SQL literal)
SLOT_TYPES = {
    "borough": lambda rng, ctx: (b := rng.choice(ctx["boroughs"]), lit(b)),
    "zone": lambda rng, ctx: (z := rng.choice(ctx["zones"]), lit(z)),
    "hour": lambda rng, ctx: (hour_nl(h := rng.randrange(24)), str(h)),
    "dow": lambda rng, ctx: (DOW_NAMES[w := rng.randrange(7)], lit(w)),
    "date": lambda rng, ctx: (_date_nl(d := rng.choice(ctx["dates"])), lit(d)),
    "payment": lambda rng, ctx: (lambda n: (n.lower(), lit(n)))(rng.choice(["Credit card", "Cash"])),
    "miles": lambda rng, ctx: (str(m := rng.choice([1, 2, 3, 5, 10, 15, 20])), str(m)),
    "passengers": lambda rng, ctx: (str(n := rng.choice([1, 2, 3, 4])), str(n)),
    "k": lambda rng, ctx: (str(k := rng.choice([3, 5, 10])), str(k)),
    "min_pickups": lambda rng, ctx: (str(n := rng.choice([10, 25, 50, 100])), str(n)),
}


# ------------------------------------------------------------------ templates

TEMPLATES = [
    Template(
        "count_trips_borough",
        ["How many trips were picked up in {borough} on {dow}s?",
         "Count the taxi rides that started in {borough} on a {dow}.",
         "What is the number of {dow} trips with a pickup in {borough}?"],
        f"SELECT COUNT(*) FROM trips t {PU} WHERE pz.borough = {{borough}} AND {DOW} = {{dow}}",
        {"borough": "borough", "dow": "dow"},
    ),
    Template(
        "avg_tip_by_borough",
        ["What is the average tip for each pickup borough on {dow}s during the {hour} hour?",
         "Break down the average tip amount by pickup borough for {dow} trips in the {hour} hour."],
        f"SELECT pz.borough, AVG(t.tip_amount) FROM trips t {PU} "
        f"WHERE {DOW} = {{dow}} AND {HOUR} = {{hour}} GROUP BY pz.borough",
        {"dow": "dow", "hour": "hour"},
    ),
    Template(
        "busiest_pickup_zone_hour",
        ["Which pickup zone was the busiest during the {hour} hour?",
         "Where were taxis picked up most often during the {hour} hour?"],
        f"SELECT pz.zone FROM trips t {PU} WHERE {HOUR} = {{hour}} "
        f"GROUP BY pz.zone ORDER BY COUNT(*) DESC LIMIT 1",
        {"hour": "hour"},
        rank_sql=f"SELECT COUNT(*) AS m FROM trips t {PU} WHERE {HOUR} = {{hour}} "
                 f"GROUP BY pz.zone ORDER BY m DESC LIMIT {{n}}",
    ),
    Template(
        "longest_trip_per_payment_borough",
        ["For trips starting in {borough} during the {hour} hour, what is the longest trip "
         "distance for each payment type?",
         "Show the maximum trip distance per payment type for pickups in {borough} in the {hour} hour."],
        f"SELECT p.name, MAX(t.trip_distance) FROM trips t {PU} {PAY} "
        f"WHERE pz.borough = {{borough}} AND {HOUR} = {{hour}} GROUP BY p.name",
        {"borough": "borough", "hour": "hour"},
    ),
    Template(
        "avg_fare_zone",
        ["What is the average fare for trips picked up in {zone}?",
         "On average, how much was the fare for rides starting in {zone}?"],
        f"SELECT AVG(t.fare_amount) FROM trips t {PU} WHERE pz.zone = {{zone}}",
        {"zone": "zone"},
    ),
    Template(
        "count_borough_to_borough",
        ["How many trips went from {borough_a} to {borough_b} on {dow}s?",
         "Count the {dow} rides picked up in {borough_a} and dropped off in {borough_b}."],
        f"SELECT COUNT(*) FROM trips t {PU} {DO} "
        f"WHERE pz.borough = {{borough_a}} AND dz.borough = {{borough_b}} AND {DOW} = {{dow}}",
        {"borough_a": "borough", "borough_b": "borough", "dow": "dow"},
    ),
    Template(
        "top_k_dropoff_zones_from_borough",
        ["List the top {k} dropoff zones for {dow} trips that started in {borough}, most popular first.",
         "What are the {k} most common destination zones for rides from {borough} on {dow}s, in order?"],
        f"SELECT dz.zone FROM trips t {PU} {DO} WHERE pz.borough = {{borough}} AND {DOW} = {{dow}} "
        f"GROUP BY dz.zone ORDER BY COUNT(*) DESC LIMIT {{k}}",
        {"borough": "borough", "k": "k", "dow": "dow"},
        ordered=True,
        rank_sql=f"SELECT COUNT(*) AS m FROM trips t {PU} {DO} WHERE pz.borough = {{borough}} "
                 f"AND {DOW} = {{dow}} GROUP BY dz.zone ORDER BY m DESC LIMIT {{n}}",
    ),
    Template(
        "avg_duration_borough",
        ["What is the average trip duration in minutes for pickups in {borough} during the {hour} hour?",
         "How many minutes does a trip starting in {borough} in the {hour} hour take on average?"],
        f"SELECT AVG({MINUTES}) FROM trips t {PU} WHERE pz.borough = {{borough}} AND {HOUR} = {{hour}}",
        {"borough": "borough", "hour": "hour"},
    ),
    Template(
        "pct_payment_borough",
        ["What percentage of {dow} trips from {borough} were paid by {payment}?",
         "Of all rides picked up in {borough} on {dow}s, what percent used {payment}?"],
        f"SELECT 100.0 * SUM(CASE WHEN p.name = {{payment}} THEN 1 ELSE 0 END) / COUNT(*) "
        f"FROM trips t {PU} {PAY} WHERE pz.borough = {{borough}} AND {DOW} = {{dow}}",
        {"borough": "borough", "payment": "payment", "dow": "dow"},
    ),
    Template(
        "revenue_borough_dow",
        ["What was the total amount collected from trips picked up in {borough} on {dow}s?",
         "Sum the total charges for all rides that started in {borough} on a {dow}."],
        f"SELECT SUM(t.total_amount) FROM trips t {PU} WHERE pz.borough = {{borough}} AND {DOW} = {{dow}}",
        {"borough": "borough", "dow": "dow"},
    ),
    Template(
        "peak_hour_zone",
        ["In {zone}, which hour of the day (0-23) had the most pickups?",
         "What is the busiest pickup hour (0-23) in {zone}?"],
        f"SELECT {HOUR} AS hour FROM trips t {PU} WHERE pz.zone = {{zone}} "
        f"GROUP BY hour ORDER BY COUNT(*) DESC LIMIT 1",
        {"zone": "zone"},
        rank_sql=f"SELECT COUNT(*) AS m FROM trips t {PU} WHERE pz.zone = {{zone}} "
                 f"GROUP BY {HOUR} ORDER BY m DESC LIMIT {{n}}",
    ),
    Template(
        "count_passengers_miles",
        ["How many trips had more than {passengers} passengers and covered more than {miles} miles?",
         "Count rides with over {passengers} passengers that were longer than {miles} miles."],
        "SELECT COUNT(*) FROM trips t WHERE t.passenger_count > {passengers} AND t.trip_distance > {miles}",
        {"passengers": "passengers", "miles": "miles"},
    ),
    Template(
        "avg_tip_pct_hour_credit",
        ["Among credit card trips with a positive fare, what is the average tip as a percentage "
         "of the fare during the {hour} hour?",
         "For card-paid rides in the {hour} hour with a fare above zero, what is the mean tip "
         "percentage (tip divided by fare, times 100)?"],
        f"SELECT AVG(100.0 * t.tip_amount / t.fare_amount) FROM trips t {PAY} "
        f"WHERE p.name = 'Credit card' AND t.fare_amount > 0 AND {HOUR} = {{hour}}",
        {"hour": "hour"},
    ),
    Template(
        "distinct_dropoffs_from_zone",
        ["How many different dropoff location IDs were reached from {zone}?",
         "Count the distinct dropoff locations for trips picked up in {zone}."],
        f"SELECT COUNT(DISTINCT t.do_location_id) FROM trips t {PU} WHERE pz.zone = {{zone}}",
        {"zone": "zone"},
    ),
    Template(
        "airport_dropoffs_from_borough",
        ["How many {dow} trips from {borough} were dropped off at JFK Airport or LaGuardia Airport?",
         "Count rides picked up in {borough} on {dow}s that ended at JFK or LaGuardia."],
        f"SELECT COUNT(*) FROM trips t {PU} {DO} WHERE pz.borough = {{borough}} AND {DOW} = {{dow}} "
        f"AND dz.zone IN ('JFK Airport', 'LaGuardia Airport')",
        {"borough": "borough", "dow": "dow"},
    ),
    Template(
        "highest_avg_fare_zone_min_pickups",
        ["Considering only {dow} trips, among pickup zones in {borough} with at least "
         "{min_pickups} pickups, which has the highest average fare?",
         "On {dow}s, which {borough} pickup zone with {min_pickups} or more trips has the "
         "highest mean fare?"],
        f"SELECT pz.zone FROM trips t {PU} WHERE pz.borough = {{borough}} AND {DOW} = {{dow}} "
        f"GROUP BY pz.zone HAVING COUNT(*) >= {{min_pickups}} ORDER BY AVG(t.fare_amount) DESC LIMIT 1",
        {"borough": "borough", "min_pickups": "min_pickups", "dow": "dow"},
        rank_sql=f"SELECT AVG(t.fare_amount) AS m FROM trips t {PU} WHERE pz.borough = {{borough}} "
                 f"AND {DOW} = {{dow}} GROUP BY pz.zone HAVING COUNT(*) >= {{min_pickups}} "
                 f"ORDER BY m DESC LIMIT {{n}}",
    ),
    Template(
        "max_total_date",
        ["What was the largest total amount charged for a single trip picked up on {date}?",
         "What is the most expensive ride, by total amount, that started on {date}?"],
        "SELECT MAX(t.total_amount) FROM trips t WHERE date(t.pickup_datetime) = {date}",
        {"date": "date"},
    ),
    Template(
        "cross_borough_trips_hour",
        ["How many trips picked up during the {hour} hour ended in a different borough than "
         "they started in?",
         "Count rides starting in the {hour} hour whose dropoff borough differs from the "
         "pickup borough."],
        f"SELECT COUNT(*) FROM trips t {PU} {DO} WHERE {HOUR} = {{hour}} AND pz.borough <> dz.borough",
        {"hour": "hour"},
    ),
    Template(
        "avg_speed_borough",
        ["Computing each trip's speed in miles per hour and then averaging, what is the average "
         "speed of trips that started in {borough} during the {hour} hour?",
         "What is the mean per-trip speed (miles per hour) for rides picked up in {borough} "
         "in the {hour} hour?"],
        f"SELECT AVG(t.trip_distance / ({MINUTES} / 60)) FROM trips t {PU} "
        f"WHERE pz.borough = {{borough}} AND {HOUR} = {{hour}}",
        {"borough": "borough", "hour": "hour"},
    ),
    Template(
        "top_tip_borough",
        ["Which pickup borough had the highest average tip on {dow}s during the {hour} hour?",
         "On {dow}s in the {hour} hour, riders from which borough tipped the most on average?"],
        f"SELECT pz.borough FROM trips t {PU} WHERE {DOW} = {{dow}} AND {HOUR} = {{hour}} "
        f"GROUP BY pz.borough ORDER BY AVG(t.tip_amount) DESC LIMIT 1",
        {"dow": "dow", "hour": "hour"},
        rank_sql=f"SELECT AVG(t.tip_amount) AS m FROM trips t {PU} WHERE {DOW} = {{dow}} "
                 f"AND {HOUR} = {{hour}} GROUP BY pz.borough ORDER BY m DESC LIMIT {{n}}",
    ),
    Template(
        "trips_per_payment_borough",
        ["How many trips of each payment type started in {borough} on {dow}s?",
         "Give the number of {dow} rides per payment type for pickups in {borough}."],
        f"SELECT p.name, COUNT(*) FROM trips t {PU} {PAY} "
        f"WHERE pz.borough = {{borough}} AND {DOW} = {{dow}} GROUP BY p.name",
        {"borough": "borough", "dow": "dow"},
    ),
    Template(
        "busiest_date_borough",
        ["On which date were the most trips picked up in {borough} during the {hour} hour?",
         "Which date (YYYY-MM-DD) had the most {borough} pickups in the {hour} hour?"],
        f"SELECT date(t.pickup_datetime) AS day FROM trips t {PU} WHERE pz.borough = {{borough}} "
        f"AND {HOUR} = {{hour}} GROUP BY day ORDER BY COUNT(*) DESC LIMIT 1",
        {"borough": "borough", "hour": "hour"},
        rank_sql=f"SELECT COUNT(*) AS m FROM trips t {PU} WHERE pz.borough = {{borough}} "
                 f"AND {HOUR} = {{hour}} GROUP BY date(t.pickup_datetime) ORDER BY m DESC LIMIT {{n}}",
    ),
    Template(
        "avg_passengers_miles",
        ["What is the average passenger count for trips from {borough} longer than {miles} miles?",
         "For rides picked up in {borough} that went over {miles} miles, how many passengers "
         "were there on average?"],
        f"SELECT AVG(t.passenger_count) FROM trips t {PU} "
        f"WHERE pz.borough = {{borough}} AND t.trip_distance > {{miles}}",
        {"borough": "borough", "miles": "miles"},
    ),
    Template(
        "zones_without_pickups",
        ["How many taxi zones in {borough} had no pickups at all during the {hour} hour?",
         "Count the zones in {borough} where no trip was picked up in the {hour} hour."],
        "SELECT COUNT(*) FROM zones z WHERE z.borough = {borough} AND z.location_id NOT IN "
        f"(SELECT t.pu_location_id FROM trips t WHERE {HOUR} = {{hour}})",
        {"borough": "borough", "hour": "hour"},
        allow_zero=True,
    ),
    Template(
        "above_avg_fare_zone",
        ["How many trips from {zone} had a fare above the overall average fare of all trips?",
         "Count rides picked up in {zone} whose fare exceeded the average fare across every trip."],
        f"SELECT COUNT(*) FROM trips t {PU} WHERE pz.zone = {{zone}} "
        f"AND t.fare_amount > (SELECT AVG(fare_amount) FROM trips)",
        {"zone": "zone"},
    ),
    Template(
        "avg_tolls_borough_to_borough",
        ["What is the average toll amount for {dow} trips from {borough_a} to {borough_b}?",
         "On {dow}s, how much in tolls did rides from {borough_a} to {borough_b} pay on average?"],
        f"SELECT AVG(t.tolls_amount) FROM trips t {PU} {DO} "
        f"WHERE pz.borough = {{borough_a}} AND dz.borough = {{borough_b}} AND {DOW} = {{dow}}",
        {"borough_a": "borough", "borough_b": "borough", "dow": "dow"},
        allow_zero=True,
    ),
]

assert len({t.id for t in TEMPLATES}) == len(TEMPLATES)
assert set(DEFAULT_HELDOUT) <= {t.id for t in TEMPLATES}
