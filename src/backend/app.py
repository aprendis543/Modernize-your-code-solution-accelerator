"""Create and configure the FastAPI application."""
from contextlib import asynccontextmanager

from api.api_routes import router as backend_router

from common.config.config import app_config
from common.logger.app_logger import AppLogger

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from helper.azure_credential_utils import get_azure_credential

from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent  # pylint: disable=E0611

from sql_agents.agent_manager import clear_sql_agents, set_sql_agents
from sql_agents.agents.agent_config import AgentBaseConfig
from sql_agents.helpers.agents_manager import SqlAgents

import uvicorn
# from agent_services.agents_routes import router as agents_router

# Load environment variables
load_dotenv()

# OpenTelemetry tracing setup
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor

    resource = Resource.create({"service.name": "code-gen-accelerator-backend"})
    provider = TracerProvider(resource=resource)

    # Configure OTLP exporter to use the local collector endpoint
    otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4319")
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)

    trace.set_tracer_provider(provider)

    _otel_available = True
except Exception:
    # If OpenTelemetry packages are not installed or initialization fails,
    # continue without tracing to avoid breaking the application startup.
    _otel_available = False

# Configure logging
logger = AppLogger("app")

# Global variables for agents
sql_agents: SqlAgents = None
azure_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global sql_agents, azure_client

    # Startup
    try:
        logger.logger.info("Initializing SQL agents...")

        # Create Azure credentials and client
        creds = get_azure_credential(app_config.azure_client_id)
        azure_client = AzureAIAgent.create_client(
            credential=creds,
            endpoint=app_config.ai_project_endpoint
        )

        # Setup agent configuration with default conversion settings
        agent_config = AgentBaseConfig(
            project_client=azure_client,
            sql_from="informix",  # Default source dialect
            sql_to="tsql"         # Default target dialect
        )

        # Create SQL agents
        sql_agents = await SqlAgents.create(agent_config)

        # Set the global agents instance
        set_sql_agents(sql_agents)
        logger.logger.info("SQL agents initialized successfully.")

    except Exception as exc:
        logger.logger.error("Failed to initialize SQL agents: %s", exc)
        # Don't raise the exception to allow the app to start even if agents fail

    yield  # Application runs here

    # Shutdown
    try:
        if sql_agents:
            logger.logger.info("Application shutting down - cleaning up SQL agents...")
            await sql_agents.delete_agents()
            logger.logger.info("SQL agents cleaned up successfully.")

            # Clear the global agents instance
            await clear_sql_agents()

        if azure_client:
            await azure_client.close()

    except Exception as exc:
        logger.logger.error("Error during agent cleanup: %s", exc)


def create_app() -> FastAPI:
    """Create and return the FastAPI application instance."""
    app = FastAPI(title="Code Gen Accelerator", version="1.0.0", lifespan=lifespan)

    # Instrument FastAPI and outgoing HTTP requests if OpenTelemetry is available
    if _otel_available:
        try:
            FastAPIInstrumentor.instrument_app(app)
            RequestsInstrumentor().instrument()
        except Exception:
            # Non-fatal: continue even if instrumentation fails
            logger.logger.warning("OpenTelemetry instrumentation failed to initialize.")

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers with /api prefix
    app.include_router(backend_router, prefix="/api", tags=["backend"])
    # app.include_router(agents_router, prefix="/api/agents", tags=["agents"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
