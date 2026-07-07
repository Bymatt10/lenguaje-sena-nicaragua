FROM nginx:1.27-alpine

RUN rm -f /etc/nginx/conf.d/default.conf

COPY docker/nginx/nginx.conf /etc/nginx/conf.d/app.conf
COPY web/ /usr/share/nginx/html/

RUN addgroup -S -g 10003 nginx-static 2>/dev/null || true \
    && chown -R nginx:nginx /var/cache/nginx /var/log/nginx /usr/share/nginx/html

EXPOSE 80 443

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -q -O- http://127.0.0.1/health || exit 1