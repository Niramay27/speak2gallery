# HTTPS setup for speak2gallery

Your original app already notes that the mic works on `localhost` and that remote access needs SSL. This patch adds direct HTTPS support to Flask plus clearer frontend guidance.

## What changed

- `app.py`
  - supports `SSL_CERT_FILE` + `SSL_KEY_FILE`
  - supports `SSL_ADHOC=1` for a self-signed development certificate
  - adds optional `FORCE_HTTPS=1` redirect support for reverse-proxy deployments
  - adds `Permissions-Policy` and `Strict-Transport-Security` headers
  - reports whether the current request is HTTPS in `/api/status`
- `static/index.html`
  - warns when the page is opened over insecure HTTP
  - blocks the mic button unless the context is secure or localhost
  - shows the expected `https://host:port` URL hint

## Fastest dev setup

Install dependencies:

```bash
pip install flask flask-cors cryptography --break-system-packages
```

Run with an auto-generated self-signed certificate:

```bash
cd /home/other/Niramay/DL/web_app
SSL_ADHOC=1 APP_PUBLIC_HOST=YOUR_SERVER_IP python app.py
```

Then open:

```text
https://YOUR_SERVER_IP:5000
```

Your browser will likely show a warning because the certificate is self-signed. Accept it once for local testing.

## Using your own certificate

```bash
cd /home/other/Niramay/DL/web_app
SSL_CERT_FILE=/path/to/fullchain.pem \
SSL_KEY_FILE=/path/to/privkey.pem \
APP_PUBLIC_HOST=your-domain.com \
python app.py
```

Open:

```text
https://your-domain.com:5000
```

## Production note

For a real deployment, the cleanest setup is usually:

- Flask app behind Nginx or Caddy
- TLS terminated at the reverse proxy
- `FORCE_HTTPS=1` enabled in Flask

That gives you proper certificates, automatic renewals, and cleaner ports.
