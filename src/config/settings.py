from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = "llm-agents"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    public_base_url: str = "http://localhost:8000"
    debug: bool = True
    max_licq_concurrent_transcriptions: int = 2
    max_fifo_concurrent_transcriptions: int = 1
    postgresql_host: str = "postgresql"
    postgresql_port: str = "5432"
    postgresql_db_name: str = "llm-agents"
    postgresql_user: str = "cluster_user"
    postgresql_password: str = ""
    db_pool_size: int = 2
    db_max_overflow: int = 3
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 1440
    environment: str = "default"
    otel_endpoint: str = "http://localhost:4317"
    crw_base: str = "http://localhost:3000"

    # rediss://default:<password>@<XXXXXX>.stackhero-network.com:<PORT_TLS>

    # local llama-server (OpenAI-compatible)
    llm_base_url: str = "http://localhost:8090/v1"
    llm_model: str = "./Qwen3.6-35B-A3B-UD-IQ4_NL.gguf"
    llm_api_key: str = "local"

    valkey_service_host: str = "http://localhost"
    valkey_service_port: int = 6379

    garage_endpoint: str = "garage.garage.svc.cluster.local:3900"
    garage_secret_key: str = "secretKey"
    garage_access_key: str = "accessKey"


    @property
    def database_url(self) -> str:
        return f"{self.postgresql_user}:{self.postgresql_password}@{self.postgresql_host}:{self.postgresql_port}/{self.postgresql_db_name}"

    @property
    def valkey_url(self) -> str:
        # taskiq-redis wants redis://host:port; strip any http(s):// scheme on the host
        host = self.valkey_service_host.split("://", 1)[-1]
        return f"redis://{host}:{self.valkey_service_port}"


settings = Settings()
