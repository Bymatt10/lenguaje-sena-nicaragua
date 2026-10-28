# Imagen de la web para la infraestructura de chepe-release (sin TLS propio:
# Cloudflare Tunnel + nginx-proxy-manager van delante). Los clips del curso no
# van en la imagen: se montan en /usr/share/nginx/html/{clips,demo}.
FROM nginx:1.27-alpine

RUN rm -f /etc/nginx/conf.d/default.conf
COPY docker/web/nginx.conf /etc/nginx/conf.d/lenguaje.conf
COPY web/ /usr/share/nginx/html/

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
