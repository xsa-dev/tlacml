download_data:
	mkdir ./data/ -p
	wget -P ../data https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_train.csv.zip
	wget -P ../data https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_test.csv.zip
	wget -P ../data https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/amer_sign2.png
	wget -P ../data https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/amer_sign3.png
	wget -P ../data https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/american_sign_language.PNG
	unzip ./data/sign_mnist_train.csv.zip -d ./data/
	unzip ./data/sign_mnist_test.csv.zip -d ./data/

start:
	uv run --active python main.py

dev:
	uv run --active python main.py --fast_dev_run True