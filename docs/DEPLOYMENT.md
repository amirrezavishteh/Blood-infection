# Deployment guide

How to put the sepsis research simulator on a server so other people can open it in a browser. For running it on your own computer, see [Run locally](../README.md#run-locally) in the README.

> **Research use only.** Deploy this for research, teaching or demos, never for patient care. The score is a research score for the PhysioNet 2019 benchmark target, not a validated clinical probability. Do not connect it to a live hospital system.

## Choose a deployment path

| Path | Best for | What runs | Status in this repository |
|---|---|---|---|
| **A. Docker Compose** | One server, least setup | PostgreSQL, API + dashboard, worker as three containers | Files written (`infra/`); **not yet run**: Docker was unavailable on the development machine |
| **B. Linux server (systemd + nginx)** | Long-running server you manage | PostgreSQL, uvicorn API, worker, nginx with TLS | Unit and nginx files written (`infra/systemd`, `infra/nginx`); the same production configuration (PostgreSQL, 2 uvicorn workers, separate worker, bearer token) was **tested natively** end to end |

Both paths need the same three things:

1. **Code**: this repository.
2. **Data and a trained model**: built with the CLI, or copied from a machine where you already built them.
3. **Secrets**: a database password and an API token, kept out of git.

## Sizing

| Resource | Minimum | Notes |
|---|---|---|
| CPU | 4 cores | The pipeline used 32 cores here; fewer cores just take longer (the hospital-B evaluation took about 15 minutes per model on 32 cores) |
| RAM | 16 GB | Full feature tables are about 1.5 GB in memory during training and evaluation |
| Disk | 30 GB free | Raw data 255 MB, Parquet store and feature caches about 1 GB, plus model artifacts, official-scorer files and the database |
| Network | Outbound HTTPS to physionet.org | Only for the one-time download |

Serving the app alone (no training) is light: 2 cores and 4 GB are enough.

---

## Path A: Docker Compose (single server)

### A1. Install the prerequisites

Install Docker Engine with the Compose plugin ([docs.docker.com/engine/install](https://docs.docker.com/engine/install/)) and git. Then check both:

```bash
docker --version
docker compose version
```

### A2. Get the code

```bash
git clone https://github.com/amirrezavishteh/Blood-infection.git
cd Blood-infection
```

### A3. Set the secrets

```bash
cp infra/.env.example infra/.env
```

Edit `infra/.env`. Set `POSTGRES_PASSWORD` and `SEPSIS_API_TOKEN` to long random strings (for example `openssl rand -hex 24`). `infra/.env` is git-ignored.

### A4. Build the image

```bash
docker compose -f infra/docker-compose.yml --env-file infra/.env build
```

### A5. Prepare data and a model (choose one)

**Option 1: real PhysioNet data** (about 1.5 hours on 32 cores, mostly the download and the hospital-B evaluation). The CLI runs inside the same image, and the results land in `./data` and `./artifacts` on the host:

```bash
C="docker compose -f infra/docker-compose.yml --env-file infra/.env run --rm api python -m sepsis"
$C -v download --dataset physionet2019
$C validate --checksums
$C -v prepare
$C -v train --config configs/lightgbm_extended.yaml --run-id lgbm-ext
$C calibrate --run lgbm-ext
$C select-policy --run lgbm-ext --split a_validation
$C evaluate --run lgbm-ext --split a_test
$C evaluate --run lgbm-ext --split b_external
$C report
```

**Option 2: synthetic demo** (about 1 minute, no download). Uncomment the two `SEPSIS_*_DIR` lines in `infra/.env`, then:

```bash
docker compose -f infra/docker-compose.yml --env-file infra/.env run --rm api \
  python -m sepsis --data-dir /app/data/demo/data --artifact-dir /app/data/demo/artifacts demo
```

**Option 3: copy what you already built.** Copy your local `data/processed/` and `artifacts/` folders into the server's `data/` and `artifacts/`.

### A6. Start the services

```bash
docker compose -f infra/docker-compose.yml --env-file infra/.env up -d
docker compose -f infra/docker-compose.yml ps
curl http://127.0.0.1:8000/health/ready
```

`/health/ready` returns `"ready": true` once the database, the worker heartbeat and a model bundle are all available.

### A7. Put HTTPS in front

The API is published on `127.0.0.1:8000` only. Follow [B8](#b8-nginx-and-https) to add nginx with a TLS certificate, or use any reverse proxy you already run.

### A8. First use

Open `https://your-domain/`, enter the API token when the dashboard asks for it, then go to **Datasets**, choose **Import PhysioNet 2019**, wait for the status to become *ready*, and create a replay in the **Replay** tab.

---

## Path B: Linux server with systemd and nginx

These steps assume Ubuntu 24.04 and the domain `sepsis.example.org`. Replace the domain with yours.

### B1. Install system packages

```bash
sudo apt update
sudo apt install -y git python3.12-venv postgresql nginx certbot python3-certbot-nginx
```

The dashboard build needs Node 20 or newer. Ubuntu's own package is older, so install it from [NodeSource](https://github.com/nodesource/distributions), or build `apps/web/dist` on another machine and copy that folder over.

### B2. Create a service user and the folders

```bash
sudo useradd --system --create-home --home-dir /var/lib/sepsis --shell /usr/sbin/nologin sepsis
sudo mkdir -p /opt/sepsis /var/lib/sepsis/data /var/lib/sepsis/artifacts /etc/sepsis
sudo chown -R sepsis:sepsis /opt/sepsis /var/lib/sepsis
```

### B3. Install the application

```bash
sudo -u sepsis git clone https://github.com/amirrezavishteh/Blood-infection.git /opt/sepsis
cd /opt/sepsis
sudo -u sepsis python3.12 -m venv .venv
sudo -u sepsis .venv/bin/pip install -e ".[postgres]"
cd apps/web && sudo -u sepsis npm ci && sudo -u sepsis npm run build && cd ../..
```

### B4. Create the database

```bash
sudo -u postgres psql -c "CREATE USER sepsis WITH PASSWORD 'CHANGE_ME';"
sudo -u postgres psql -c "CREATE DATABASE sepsis OWNER sepsis;"
```

The tables are created automatically on first start.

### B5. Write the environment file

```bash
sudo cp infra/systemd/sepsis.env.example /etc/sepsis/sepsis.env
sudo nano /etc/sepsis/sepsis.env          # set the DB password and a long random SEPSIS_API_TOKEN
sudo chown root:sepsis /etc/sepsis/sepsis.env
sudo chmod 640 /etc/sepsis/sepsis.env
```

### B6. Prepare data and a model

Run the CLI as the service user with the same environment the services use:

```bash
run() { sudo -u sepsis bash -c "set -a; . /etc/sepsis/sepsis.env; set +a; cd /opt/sepsis; .venv/bin/python -m sepsis $*"; }
run -v download --dataset physionet2019
run validate --checksums
run -v prepare
run -v train --config configs/lightgbm_extended.yaml --run-id lgbm-ext
run calibrate --run lgbm-ext
run select-policy --run lgbm-ext --split a_validation
run evaluate --run lgbm-ext --split a_test
run evaluate --run lgbm-ext --split b_external
run report
```

Alternatively, copy `data/processed/` and `artifacts/` from a machine where you already ran the pipeline into `/var/lib/sepsis/data/processed/` and `/var/lib/sepsis/artifacts/`, then `chown -R sepsis:sepsis /var/lib/sepsis`.

### B7. Install and start the services

```bash
sudo cp infra/systemd/sepsis-api.service infra/systemd/sepsis-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sepsis-api sepsis-worker
systemctl status sepsis-api sepsis-worker
curl http://127.0.0.1:8000/health/ready
```

Logs:

```bash
journalctl -u sepsis-api -f
journalctl -u sepsis-worker -f
```

### B8. nginx and HTTPS

```bash
sudo cp infra/nginx/sepsis.conf /etc/nginx/sites-available/sepsis
sudo sed -i 's/sepsis.example.org/YOUR_DOMAIN/g' /etc/nginx/sites-available/sepsis
sudo ln -s /etc/nginx/sites-available/sepsis /etc/nginx/sites-enabled/
sudo certbot certonly --nginx -d YOUR_DOMAIN
sudo nginx -t && sudo systemctl reload nginx
```

The DNS A record for your domain must point to the server first. Open ports 80 and 443 only; port 8000 stays private:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

### B9. First use

Same as [A8](#a8-first-use).

---

## Operating the deployment

### Update the code

```bash
cd /opt/sepsis
sudo -u sepsis git pull
sudo -u sepsis .venv/bin/pip install -e ".[postgres]"
cd apps/web && sudo -u sepsis npm ci && sudo -u sepsis npm run build && cd ../..
sudo systemctl restart sepsis-api sepsis-worker
```

With Docker: `git pull`, then `docker compose -f infra/docker-compose.yml --env-file infra/.env up -d --build`.

### Add or switch a model

1. Train, calibrate, select a policy and evaluate the new run (B6).
2. Promote it: `run promote --runs OLD_RUN NEW_RUN`. This picks the higher validation AUPRC and writes `artifacts/promotion.json`.
3. New replays use the promoted model. Existing replays stay pinned to the model they started with, so their history never changes.

No restart is needed.

### Back up

| What | How |
|---|---|
| Replays, alerts, reviews, audit log | `sudo -u postgres pg_dump sepsis > sepsis-$(date +%F).sql` |
| Models and reports | Copy `/var/lib/sepsis/artifacts/` |
| Processed data | Rebuildable with `prepare`; back up `/var/lib/sepsis/data/raw/manifest.json` to prove which files were used |

### Security checklist

- [ ] `SEPSIS_API_TOKEN` is set to a long random value (every `/v1` call requires it; the dashboard asks once and keeps it in that browser).
- [ ] The API listens on `127.0.0.1` only; nginx terminates TLS; ports 80 and 443 are the only open ports.
- [ ] `infra/.env` and `/etc/sepsis/sepsis.env` are not in git and are readable only by root and the service user.
- [ ] Optional: nginx basic auth (commented out in `infra/nginx/sepsis.conf`) as a second layer.
- [ ] Only open data is deployed. Credentialed datasets (MIMIC-IV, eICU, HiRID, AmsterdamUMCdb) must stay inside environments their data-use agreements allow; never put them on a public server.
- [ ] The PhysioNet 2019 attribution (CC BY 4.0) stays visible on the Datasets screen.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/health/ready` shows `"worker": false` | Worker not running, or no heartbeat for 30 s | `systemctl status sepsis-worker` / `docker compose ps`; check its logs |
| `/health/ready` shows `"model": false` | No calibrated run with a frozen policy | Run B6 (or the demo), then reload |
| Dashboard asks for a token, then shows 401 | Wrong token | Clear the site data in the browser and enter the token again |
| Dataset stays *failed* | Processed files changed after `prepare`, or no split file | Rerun `prepare`; the error text names the file |
| Download stops with "N files failed" | Server-side timeouts at physionet.org | Run the same download command again; it resumes and fetches only missing files |
| Replay does not move after **Play** | Worker stopped | Start the worker; queued steps continue where they left off |
| Old dashboard after an update | Browser cache | Hard refresh (Ctrl+Shift+R); `index.html` is served with `no-cache`, so this should be rare |
