// Despliegue en chepe-vps con el mismo esquema que los demás servicios:
// imagen → registro local → chepe-release (verifica salud y revierte solo).
// Dos servicios: lenguaje-api (Flask) y lenguaje-web (nginx con web/).
pipeline {
    agent { label 'chepe-ci-rootless' }
    options {
        timestamps()
        ansiColor('xterm')
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
    }
    environment {
        APP_NAME = 'lenguaje'
        REGISTRY = '127.0.0.1:5055/chepe-ci'
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_SHA = sh(returnStdout: true, script: 'git rev-parse --short=7 HEAD').trim()
                    env.IMAGE_TAG = "${env.BUILD_NUMBER}-${env.GIT_SHA}"
                }
            }
        }
        stage('Docker Build') {
            parallel {
                stage('api') {
                    steps { sh 'docker build -t "$APP_NAME-api:$IMAGE_TAG" -f Dockerfile .' }
                }
                stage('web') {
                    steps { sh 'docker build -t "$APP_NAME-web:$IMAGE_TAG" -f docker/web/Dockerfile .' }
                }
            }
        }
        stage('Validate') {
            steps {
                // la API responde /health y rechaza contribuciones sin token
                sh '''
                    docker run --rm --network none -e LSN_SECRET=ci-only -e FLASK_ENV=production \
                        "$APP_NAME-api:$IMAGE_TAG" python -c "
import server
c = server.app.test_client()
assert c.get('/health').status_code == 200
assert c.post('/contribute', json={}).status_code == 401
print('api OK')"
                '''
                // nginx válido y el modelo + self-test presentes en la web
                sh '''
                    docker run --rm --network none "$APP_NAME-web:$IMAGE_TAG" sh -ec '
                        nginx -t
                        for f in index.html app.js style.css words.json lessons.json signs_anim.json \
                                 model/weights.bin model/weights_spec.json model/centroids.json \
                                 model/reference.json model/metrics.json; do
                            test -s "/usr/share/nginx/html/$f"
                        done
                        echo "web OK"
                    '
                '''
            }
        }
        stage('Publish images') {
            steps {
                sh '''
                    set -eu
                    for svc in api web; do
                        ref="$REGISTRY/$APP_NAME-$svc"
                        docker tag "$APP_NAME-$svc:$IMAGE_TAG" "$ref:$IMAGE_TAG"
                        docker push "$ref:$IMAGE_TAG"
                        docker image inspect "$ref:$IMAGE_TAG" --format '{{range .RepoDigests}}{{println .}}{{end}}' \
                            | awk -F@ -v repo="$ref" '$1 == repo {print $2}' > ".digest-$svc"
                        test "$(wc -l < ".digest-$svc")" -eq 1
                    done
                '''
            }
        }
        stage('Deploy & Verify') {
            steps {
                // primero la API: la web nueva puede depender de ella
                sh '/usr/local/bin/chepe-release lenguaje-api "$(cat .digest-api)" "$IMAGE_TAG"'
                sh '/usr/local/bin/chepe-release lenguaje-web "$(cat .digest-web)" "$IMAGE_TAG"'
            }
        }
    }
    post {
        success { echo 'lenguaje-api y lenguaje-web desplegados y verificados por chepe-release.' }
        cleanup { cleanWs() }
    }
}
