# Stage 1 - build the React app
FROM node:20-alpine AS builder

WORKDIR /build

COPY tfg-arquitectura/frontend/package.json ./
RUN npm install

COPY tfg-arquitectura/frontend ./

# Bake in the API URL for production (nginx handles proxying on /api/)
ENV VITE_API_URL=""
RUN npm run build

# Stage 2 - nginx serving static files + reverse proxy
FROM nginx:1.27-alpine

COPY tfg-arquitectura/gateway/nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 80
