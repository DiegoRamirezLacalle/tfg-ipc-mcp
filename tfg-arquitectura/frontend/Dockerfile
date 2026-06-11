FROM node:20-alpine

WORKDIR /app

COPY tfg-arquitectura/frontend/package.json ./
RUN npm install

COPY tfg-arquitectura/frontend ./

EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
