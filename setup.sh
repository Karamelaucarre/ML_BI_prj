# Content for setup.sh
mkdir -p ~/.streamlit/
echo "[general]
email = \"talebk9@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
enableCORS=false
port = $PORT
" > ~/.streamlit/config.toml