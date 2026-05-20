# ---- frontend build stage ------------------------------------------------
# Pre-builds templates/styles.css from the Tailwind source. Kept separate
# from the runtime image so we don't ship Node into production.
FROM node:22-alpine AS css-build
WORKDIR /build
COPY package.json tailwind.config.js ./
COPY templates ./templates
RUN npm install --no-audit --no-fund --silent \
    && npx tailwindcss -i ./templates/styles.src.css -o ./templates/styles.css --minify

# ---- runtime image -------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AEGIS_NO_BROWSER=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
# Overwrite the committed CSS with the freshly built one so the container
# never serves a stale stylesheet.
COPY --from=css-build /build/templates/styles.css ./templates/styles.css

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
