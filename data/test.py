import requests
import pprint

response = requests.get('https://data-api.coindesk.com/index/cc/v1/historical/days',
    params={"market":"cadli","instrument":"BTC-USD","aggregate":1,"fill":"true","apply_mapping":"true","response_format":"JSON","to_ts":1757670621,"groups":"ID,OHLC,VOLUME,MESSAGE","limit":2000,"api_key":"b000075c30bc6a4a6e66f7e41cb7b98f0aca76416cb278859ff034b19a97cf0e"},
    headers={"Content-type":"application/json; charset=UTF-8"}
)

json_response = response.json()
pprint.pprint(json_response)