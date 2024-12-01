from enum import Enum
from typing import Any, Dict, List, Mapping, Optional
from pydantic import BaseModel


class ServiceDefinition(BaseModel):
    class_path: str
    optional_params: List[str]
    required_params: List[str]
    default_params: Dict[str, Any]


class ServiceInfo(BaseModel):
    service_name: str
    requires_api_key: bool
    optional_params: List[str]
    required_params: List[str]


class ServiceFactory:
    _services: Dict[str, ServiceDefinition] = {}

    @classmethod
    def register_service(
        cls,
        service_class: str,
        service_name: str,
        default_params: Optional[Dict[str, Any]] = None,
        optional_params: Optional[List[str]] = None,
        required_params: Optional[List[str]] = None,
    ) -> None:
        if service_name in cls._services:
            raise ValueError(
                f"Service '{service_name}' is already registered"
            )

        cls._services[service_name] = ServiceDefinition(
            class_path=service_class,
            optional_params=optional_params or [],
            required_params=required_params or [],
            default_params=default_params or {},
        )

    @classmethod
    def get_service(
        cls,
        service_name: str,
        service_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        if service_name not in cls._services:
            raise ValueError(
                f"Service '{service_name}' is not registered"
            )

        service_info = cls._services[service_name]

        kwargs = service_info.default_params.copy()

        # Check required parameters
        for param in service_info.required_params:
            if (service_options is None or param not in service_options) and param not in kwargs:
                raise ValueError(
                    f"Required parameter '{param}' missing for service '{service_name}'"
                )
            if service_options and param in service_options:
                kwargs[param] = service_options[param]

        # Add optional parameters if provided
        for param in service_info.optional_params:
            if service_options and param in service_options:
                kwargs[param] = service_options[param]

        # Add any additional provided parameters that aren't in optional or required lists
        if service_options:
            for key, value in service_options.items():
                if key not in kwargs:
                    kwargs[key] = value

        # Instantiate the service
        module_name, class_name = service_info.class_path.rsplit(":", 1)
        module = __import__(module_name, fromlist=[class_name])
        service_class = getattr(module, class_name)

        return service_class(**kwargs)

    @classmethod
    def get_service_defintion(
        cls, service_name: str
    ) -> ServiceDefinition:
        if service_name not in cls._services:
            raise ValueError(f"Service '{service_name}' is not registered")
        return cls._services[service_name]

    @classmethod
    def get_available_service(cls) -> List[ServiceInfo]:
        services = ((name, service_dict) for name, service_dict in cls._services.items())

        return [
            ServiceInfo(
                service_name=name,
                optional_params=service.optional_params,
                required_params=service.required_params,
            )
            for name, service in services
        ]

# Audio amplifier service
ServiceFactory.register_service(
    "app.common.services.amplifier:AudioAmplifier",
    "amplifier",
    optional_params=["chunk_size", "sample_rate"],
)

# Audiogram reader service
ServiceFactory.register_service(
    "app.common.services.audiogram_reader:AudiogramReader",
    "audiogram_reader",
    required_params=["box_model_path", "symbol_model_path"]
)