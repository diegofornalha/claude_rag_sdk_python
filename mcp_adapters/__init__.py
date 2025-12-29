"""
MCP Adapters - Plugins opcionais para diferentes MCP servers

Cada adapter neste módulo é OPCIONAL e pode ser removido sem
afetar o funcionamento do sistema core.

Para adicionar um novo adapter:
1. Crie um arquivo (ex: my_adapter.py)
2. Implemente BaseMCPAdapter
3. Registre no registry (veja angular_cli.py como exemplo)

Para remover um adapter:
1. Delete o arquivo
2. Remova a importação abaixo
3. O sistema continua funcionando normalmente

Adapters disponíveis:
- angular_cli: Integração com Angular CLI MCP Server
"""

# Lista de adapters disponíveis
# Cada adapter é importado de forma segura (try/except)
# para não quebrar se o adapter for removido

_available_adapters: list[str] = []


def _try_register_angular_cli():
    """Tenta registrar o adapter Angular CLI."""
    try:
        from . import angular_cli  # noqa: F401

        _available_adapters.append("angular-cli")
    except ImportError:
        pass  # Adapter não disponível, ignora silenciosamente


def register_all_adapters():
    """
    Registra todos os adapters disponíveis.

    Deve ser chamado durante a inicialização da aplicação.
    Adapters que falharem ao importar são ignorados.
    """
    _try_register_angular_cli()
    # Adicionar futuros adapters aqui:
    # _try_register_react_devtools()
    # _try_register_vue_cli()


def get_available_adapters() -> list[str]:
    """Retorna lista de adapters que foram carregados com sucesso."""
    return _available_adapters.copy()


# Auto-registra adapters ao importar o módulo
register_all_adapters()
