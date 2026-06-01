# --- Build stage: compile the Vite app to static assets ---------------------
FROM node:22-slim AS build

WORKDIR /app

# VITE_API_URL is baked into the bundle at build time (it's read in src/api.ts).
# Point it at the backend URL reachable from the *browser*, not the container.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# --- Runtime stage: serve the built assets with nginx ------------------------
FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
