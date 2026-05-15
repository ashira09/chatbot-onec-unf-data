#!/bin/bash
set -e

python -m src.bitrix.integration.init
python -m src.yandex_cloud.integration.init
python -m src.main