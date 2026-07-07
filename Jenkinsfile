pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '5'))
        ansiColor('xterm')
    }

    environment {
        REGISTRY       = 'ghcr.io'
        IMAGE_NAMESPACE = 'bymatt10'
        IMAGE_NAME      = 'lengua-lsp'
        DOCKER_BUILDKIT = '1'
        COMPOSE_PROJECT_NAME = 'lengua-lsp'
    }

    parameters {
        string(name: 'TAG', defaultValue: '', description: 'Tag de la imagen (vacío = número de build o branch)')
        choice(name: 'DEPLOY_ENV', choices: ['staging', 'production'], description: 'Entorno destino')
        booleanParam(name: 'DEPLOY', defaultValue: true, description: 'Desplegar tras construir')
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
                    env.GIT_BRANCH = sh(script: "git rev-parse --abbrev-ref HEAD", returnStdout: true).trim()
                    env.TAG = params.TAG?.trim() ?: "${env.GIT_BRANCH}-${env.BUILD_NUMBER}-${env.GIT_COMMIT_SHORT}"
                }
            }
        }

        stage('Lint') {
            parallel {
                stage('ruff') {
                    steps {
                        sh '''
                            python3 -m venv .venv-lint || true
                            . .venv-lint/bin/activate
                            pip install -q ruff==0.7.4
                            ruff check server.py validate_contrib.py logging_config.py auto_retrain.py tools/
                        '''
                    }
                }
                stage('docker compose config') {
                    steps {
                        sh 'docker compose config -q'
                    }
                }
            }
        }

        stage('Audit deps') {
            when { changeRequest() }
            steps {
                sh '''
                    python3 -m venv .venv-audit || true
                    . .venv-audit/bin/activate
                    pip install -q pip-audit==2.7.3
                    pip-audit -r requirements.txt || true
                '''
            }
        }

        stage('Build images') {
            parallel {
                stage('app') {
                    steps {
                        sh "docker build -t ${REGISTRY}/${IMAGE_NAMESPACE}/${IMAGE_NAME}/app:${TAG} -f Dockerfile ."
                    }
                }
                stage('nginx') {
                    steps {
                        sh "docker build -t ${REGISTRY}/${IMAGE_NAMESPACE}/${IMAGE_NAME}/nginx:${TAG} -f docker/nginx/Dockerfile ."
                    }
                }
                stage('cron') {
                    steps {
                        sh "docker build -t ${REGISTRY}/${IMAGE_NAMESPACE}/${IMAGE_NAME}/cron:${TAG} -f docker/cron/Dockerfile ."
                    }
                }
            }
        }

        stage('Push images') {
            when { anyOf { branch 'main'; branch 'master'; branch 'release/*'; tag pattern: 'v\\d+\\.\\d+\\.\\d+', comparator: 'REGEXP' } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'github-container-registry', usernameVariable: 'GH_USER', passwordVariable: 'GH_TOKEN')]) {
                    sh """
                        echo "\$GH_TOKEN" | docker login ${REGISTRY} -u \$GH_USER --password-stdin
                        for svc in app nginx cron; do
                            docker push ${REGISTRY}/${IMAGE_NAMESPACE}/${IMAGE_NAME}/\$svc:${TAG}
                            docker tag  ${REGISTRY}/${IMAGE_NAMESPACE}/${IMAGE_NAME}/\$svc:${TAG} ${REGISTRY}/${IMAGE_NAMESPACE}/${IMAGE_NAME}/\$svc:latest
                            docker push ${REGISTRY}/${IMAGE_NAMESPACE}/${IMAGE_NAME}/\$svc:latest
                        done
                    """
                }
            }
        }

        stage('Deploy') {
            when {
                allOf {
                    expression { return params.DEPLOY }
                    anyOf { branch 'main'; branch 'master'; tag pattern: 'v\\d+\\.\\d+\\.\\d+', comparator: 'REGEXP' }
                }
            }
            matrix {
                axes {
                    axis { name 'TARGET'; values 'staging', 'production' }
                }
                when { not { anyOf { branch 'main'; branch 'master' } } }
                stages { stage('skip') { steps { echo "No-op for ${TARGET}" } } }
            }
            steps {
                script {
                    def host = (params.DEPLOY_ENV == 'production')
                        ? 'lenguaje.chepeonline.com'
                        : 'staging.chepeonline.com'
                    def remoteDir = "/srv/${COMPOSE_PROJECT_NAME}"
                    def composeFile = (params.DEPLOY_ENV == 'production') ? 'docker-compose.yml' : 'docker-compose.staging.yml'

                    sshagent(credentials: ['vps-ssh-key']) {
                        sh """
                            ssh -o StrictHostKeyChecking=accept-new ${params.DEPLOY_ENV}@${host} "
                                set -e
                                cd ${remoteDir}
                                export TAG=${TAG}
                                export LSN_SECRET=\$(cat .env | grep '^LSN_SECRET=' | cut -d= -f2)
                                export LSN_SESSION_TTL=\$(cat .env | grep '^LSN_SESSION_TTL=' | cut -d= -f2)
                                export LSN_RETENTION_DAYS=\$(cat .env | grep '^LSN_RETENTION_DAYS=' | cut -d= -f2)

                                docker compose -f ${composeFile} -p ${COMPOSE_PROJECT_NAME}-${params.DEPLOY_ENV} pull
                                docker compose -f ${composeFile} -p ${COMPOSE_PROJECT_NAME}-${params.DEPLOY_ENV} up -d --remove-orphans
                                docker compose -f ${composeFile} -p ${COMPOSE_PROJECT_NAME}-${params.DEPLOY_ENV} ps
                                docker image prune -f
                            "
                        """
                    }
                }
            }
        }

        stage('Smoke test') {
            when {
                allOf {
                    expression { return params.DEPLOY }
                    anyOf { branch 'main'; branch 'master'; tag pattern: 'v\\d+\\.\\d+\\.\\d+', comparator: 'REGEXP' }
                }
            }
            steps {
                script {
                    def url = (params.DEPLOY_ENV == 'production')
                        ? 'https://lenguaje.chepeonline.com/health'
                        : 'https://staging.chepeonline.com/health'
                    sh """
                        sleep 10
                        for i in 1 2 3 4 5; do
                            if curl -fsS -o /dev/null -w '%{http_code}' ${url} | grep -q '^200$'; then
                                echo 'smoke test OK'
                                exit 0
                            fi
                            echo "intento \$i/5 falló; esperando 10s"
                            sleep 10
                        done
                        exit 1
                    """
                }
            }
        }
    }

    post {
        success {
            echo "Build ${env.TAG} OK"
        }
        failure {
            echo "Build ${env.TAG} falló"
        }
        always {
            sh '''
                docker image prune -f || true
                rm -rf .venv-lint .venv-audit || true
            '''
        }
        cleanup {
            cleanWs(deleteDirs: true, patterns: [[pattern: '.venv-*', type: 'INCLUDE']])
        }
    }
}