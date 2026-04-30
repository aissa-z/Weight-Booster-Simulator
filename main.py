from interface import run_app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDERS = ["booking", "expedia", "hotelscom"]
PROB_COLS = ["BOOKING_ADJ_PROB", "EXPEDIA_ADJ_PROB", "HOTELSCOM_ADJ_PROB"]

DATA_QUERY = """
SELECT
  c.timestamp,
  c.user_country,
  c.dest_country,
  c.product,
  c.medium,
  c.provider,
  IFF(b._id IS NOT NULL, 1, 0) AS booked,
  totalcommission,
  PARSE_JSON(c.ml)[0]:adjustedProbability::FLOAT AS top1_adj_prob,
  PARSE_JSON(c.ml)[1]:adjustedProbability::FLOAT AS top2_adj_prob,
  PARSE_JSON(c.ml)[2]:adjustedProbability::FLOAT AS top3_adj_prob,
  CASE WHEN PARSE_JSON(c.ml)[0]:provider::STRING = 'booking' THEN PARSE_JSON(c.ml)[0]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[1]:provider::STRING = 'booking' THEN PARSE_JSON(c.ml)[1]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[2]:provider::STRING = 'booking' THEN PARSE_JSON(c.ml)[2]:adjustedProbability::FLOAT
  END AS booking_adj_prob,
  CASE WHEN PARSE_JSON(c.ml)[0]:provider::STRING = 'expedia' THEN PARSE_JSON(c.ml)[0]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[1]:provider::STRING = 'expedia' THEN PARSE_JSON(c.ml)[1]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[2]:provider::STRING = 'expedia' THEN PARSE_JSON(c.ml)[2]:adjustedProbability::FLOAT
  END AS expedia_adj_prob,
  CASE WHEN PARSE_JSON(c.ml)[0]:provider::STRING = 'hotelscom' THEN PARSE_JSON(c.ml)[0]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[1]:provider::STRING = 'hotelscom' THEN PARSE_JSON(c.ml)[1]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[2]:provider::STRING = 'hotelscom' THEN PARSE_JSON(c.ml)[2]:adjustedProbability::FLOAT
  END AS hotelscom_adj_prob
FROM product.transformed.events_pubsub_clicks c
LEFT JOIN ENGINEERING.TRANSFORMED.MONGO_HUB_RELEASE_BOOKINGS b
  ON b.click_id = c.click_id
WHERE c.category = 'accommodation'
  AND c.is_wildcarded = TRUE
  AND c.is_biased = FALSE
  AND c.provider <> 'airbnb'
  AND c.timestamp BETWEEN '{start_date}' AND '{end_date}'
  AND c.user_country = '{country}'
"""


if __name__ == "__main__":
    run_app(PROVIDERS, PROB_COLS, DATA_QUERY)
