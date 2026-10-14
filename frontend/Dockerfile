FROM node:20-alpine AS build 

WORKDIR /app 
COPY package*.json ./
RUN npm install 
COPY . .
RUN npm run build 

# Start a separate lightweight Nginx image to serve the production frontend.
FROM nginx:alpine 
# Copy the built frontend files from the build stage into Nginx's public web directory.
COPY --from=build /app/dist /usr/share/nginx/html 
# Document that Nginx serves the frontend on container port 80.
EXPOSE 80 

# Run Nginx in the foreground so it remains the container's main process.
CMD ["nginx", "-g", "daemon off;"]